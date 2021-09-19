import os
import argparse
import random
import numpy as np

import wandb
import torch
import esm
import sidechainnet
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB

import mp_nerf
from rgn2_replica.rgn2_trainers import train
from rgn2_replica.embedders import get_embedder
from rgn2_replica import set_seed, RGN2_Naive

VOCAB = VOCAB()

# params for prost
MIN_LEN = 0
MAX_LEN = 512


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RGN2 model')
    parser.add_argument("--device", help="Device ('cpu', 'cuda', 'cuda:0')", type=str, required=True)
    parser.add_argument("--run_name", help="Experiment name", type=str, required=True)

    parser.add_argument("--embedder_model", help="W & B project name", default='esm1b')
    parser.add_argument("--wb_proj", help="W & B project name", type=str, default=None)
    parser.add_argument("--wb_entity", help="W & B entity", type=str, default=None)
    parser.add_argument("--seed", help="Random seed", default=101)

    return parser.parse_args()


def load_dataloader(config):
    dataloaders = sidechainnet.load(casp_version=7, thinning=30, with_pytorch="dataloaders",
                                    batch_size=1, dynamic_batching=False)

    return dataloaders


def init_wandb_config(args):
    wandb.init(project=args.wb_proj, entity=args.wb_entity, name=args.run_name)

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.seed = args.seed
    config.device = args.device

    # model hyperparams
    config.embedder_model = args.embedder_model
    config.num_layers = 2
    config.emb_dim = 1280
    config.hidden = 1024
    config.mlp_hidden = [128, 4]  # 4 # 64
    config.act = "silu"  # "silu"
    config.layer_type = "LSTM"  # "LSTM"
    config.input_dropout = 0.5

    config.bidirectional = True  # True
    config.max_recycles_train = 3  #  set up to 1 to speed things
    config.angularize = False

    return config


def init_and_train(args):
    config = init_wandb_config(args)

    dataloaders = load_dataloader(config)
    print('loaded dataloaders')

    embedder = get_embedder(config, config.device)
    print('loaded embedder')

    run_train_schedule(dataloaders, embedder, config)


def run_train_schedule(dataloaders, embedder, config):
    device = torch.device(config.device)
    embedder = embedder.to(device)

    set_seed(config.seed)
    model = RGN2_Naive(layers=config.num_layers,
                       emb_dim=config.emb_dim+4,
                       hidden=config.hidden,
                       bidirectional=config.bidirectional,
                       mlp_hidden=config.mlp_hidden,
                       act=config.act,
                       layer_type=config.layer_type,
                       input_dropout=config.input_dropout,
                       angularize=config.angularize,
                       ).to(device)

    # 3. Log gradients and model parameters
    wandb.watch(model)

    steps = get_training_schedule()

    resume = False  #  only if declaring new optim
    for i, (batch_num, checkpoint, lr, batch_size, max_len, clip, loss_f, seed) in enumerate(steps):
        # reconfig batch otpions
        wandb.log({
            'learning_rate': lr,
            'batch_size': batch_size
        }, commit=False)

#         if i < 2:
#             continue
        # if i == 2: break # if i < 1: continue
        # if i == 5: break #  continue

        if i == 0 or resume:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        else:
            for g in optimizer.param_groups:
                g['lr'] = lr

        if seed is not None or resume:
            if not resume:
                set_seed(seed)
            get_prot_ = mp_nerf.utils.get_prot(
                dataloader_=dataloaders,
                vocab_=VOCAB,
                min_len=MIN_LEN, max_len=max_len,  # MAX_LEN,
                verbose=False, subset="train"
            )

        # train
        metrics_stuff = train(
            get_prot_=get_prot_,
            steps=batch_num,
            model=model,
            embedder=embedder,
            optim=optimizer,
            loss_f=loss_f,  # + 0.005 * metrics['drmsd'].mean()",
            clip=clip,
            accumulate_every=batch_size,
            log_every=4,
            seed=seed,
            recycle_func=lambda x: random.randint(1, config.max_recycles_train),  # 1
            wandbai=True,
        )

        metric = np.mean([x["drmsd"] for x in metrics_stuff[0][-5*batch_size:]])
        print("\nCheckpoint {0} @ {1}, pass @ {2}. Metrics mean train = {1}\n".format(
            i, checkpoint, metric, metrics_stuff[-1]
        ))

        # save
        save_path = "rgn2_models/"+wandb.run.name.replace("/", "_")+"@_{0}K.pt".format(
            sum(p[0] for p in steps[:i+1]) // 1000
        )
        torch.save(model.state_dict(), save_path)

        # ABORT OR CONTINUE: mean of last 5 batches below checkpoint
        if metric > checkpoint:
            print("ABORTING")
            print("Didn't pass checkpoint {0} @ drmsd = {1}, but instead drmsd = {2}".format(
                i, checkpoint, metric
            ))
            break

    # !mkdir rgn2_models
    os.makedirs('rgn2_models', exist_ok=True)
    save_path = "rgn2_models/"+wandb.run.name.replace("/", "_")+"@_{0}K.pt".format(
        sum(p[0] for p in steps[:i+1]) // 1000
    )
    torch.save(model.state_dict(), save_path)


def get_training_schedule():
    loss_f = "torsion_loss.mean()"
    loss_after = " + 0.2 * ( metrics['drmsd'].mean() / len(infer['seq']) )"  # metrics['rmsd'].mean() +

    #         steps, checkpoint, lr , bs , max_len, clip, loss_f
    return [[1000, 135, 1e-3, 8, MAX_LEN, None, loss_f, 42, ],
            [4000, 50, 1e-3, 16, MAX_LEN, None, loss_f, None, ],
            [10000, 50, 1e-3, 16, MAX_LEN, None, loss_f, None, ],
            [5000, 45, 1e-3, 16, MAX_LEN, 1., loss_f+loss_after+" * 0.05", None, ],
            [5000, 40, 1e-3, 32, MAX_LEN, 1., loss_f+loss_after+" * 0.1", None, ],
            [5000, 35, 1e-3, 32, MAX_LEN, 1., loss_f+loss_after+" * 0.175", None, ],
            # these ones add little value, it's mostly refinement
            [5000, 35, 1e-3, 32, MAX_LEN, 1., loss_f+loss_after+" * 0.25", None, ],
            [5000, 35, 1e-3, 32, MAX_LEN, 1., loss_f+loss_after+" * 0.375", None, ],
            [5000, 35, 1e-3, 32, MAX_LEN, 1., loss_f+loss_after+" * 0.5", None, ],
            [5000, 35, 1e-3, 32, MAX_LEN, 1., loss_f+loss_after+" * 0.75", None, ],
            [5000, 33.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 1.00", None, ],
            [5000, 33.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 1.50", None, ],
            [5000, 33.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 1.50", None, ],
            [5000, 33.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 2.00", None, ],
            [5000, 23.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 2.00", None, ],
            [5000, 23.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 2.50", None, ],
            [5000, 23.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 2.50", None, ],
            [5000, 23.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 3.00", None, ],
            [5000, 23.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 3.00", None, ],
            [5000, 23.75, 5e-4, 32, MAX_LEN, 1., loss_f+loss_after+" * 3.75", None, ],
            # till here
            [5000, 23.75, 5e-4, 64, MAX_LEN, 1., loss_f+loss_after+" * 4.50", None, ],
            [10000, 20, 1e-4, 64, MAX_LEN, 1., loss_f+loss_after+" * 6.25", None, ],
            [10000, 20, 1e-4, 64, MAX_LEN, 1., loss_f+loss_after+" * 10.00", None, ],
            [10000, 20, 1e-4, 64, MAX_LEN, 1., loss_f+loss_after+" * 20.00", None, ], ]


if __name__ == '__main__':
    args = parse_arguments()
    init_and_train(args)
