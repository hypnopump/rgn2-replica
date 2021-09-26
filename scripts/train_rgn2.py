import os
import json
import argparse
import random
import numpy as np

import wandb
import torch
import esm
import sidechainnet
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB

# IMPORTED ALSO IN LATER MODULES
VOCAB = VOCAB()

import mp_nerf
from rgn2_replica.rgn2_trainers import *
from rgn2_replica.embedders import *
from rgn2_replica import set_seed, RGN2_Naive




def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RGN2 model')
    # logging
    parser.add_argument("--device", help="Device ('cpu', cuda:0', ...)", type=str, required=True)
    parser.add_argument("--wb_proj", help="W & B project name", type=str, default=None)
    parser.add_argument("--wb_entity", help="W & B entity", type=str, default=None)
    parser.add_argument("--run_name", help="Experiment name", type=str, required=True)
    # run handling
    parser.add_argument("--resume_name", help="model path to load and resume", type=str, default=None)
    parser.add_argument("--resume_iters", help="num of iters to resume training at", type=int, default=0)
    # data params
    parser.add_argument("--min_len", help="Min seq len, for train", type=int, default=0)
    parser.add_argument("--min_len_valid", help="Min seq len, for valid", type=int, default=0)
    parser.add_argument("--max_len", help="Max seq len", type=int, default=512)
    parser.add_argument("--casp_version", help="SCN dataset version", type=int, default=12)
    parser.add_argument("--scn_thinning", help="SCN dataset thinning", type=int, default=90)
    parser.add_argument("--xray", help="only use xray structures", type=bool, default=0)
    parser.add_argument("--frac_true_torsions", help="Provide right torsions for some prots", type=bool, default=0)
    # model params
    parser.add_argument("--embedder_model", help="Embedding model to use", default='esm1b')
    parser.add_argument("--num_layers", help="num rnn layers", type=int, default=2)
    parser.add_argument("--emb_dim", help="embedding dimension", type=int, default=1280)
    parser.add_argument("--hidden", help="hidden dimension", type=int, default=1024)
    parser.add_argument("--act", help="hidden activation", type=str, default="silu")
    parser.add_argument("--layer_type", help="rnn layer type", type=str, default="LSTM")
    parser.add_argument("--input_dropout", help="input dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", help="bidirectionality", type=bool, default=0)
    parser.add_argument("--angularize", help="angularization units. 0 for reg", type=int, default=0)
    parser.add_argument("--num_recycles_train", type=int, default=3, 
                        help="number of recycling iters. set to 1 to speed training.",)
    # refiner params
    parser.add_argument("--refiner_args", help="args for refiner module", type=json.loads, default={})
    parser.add_argument("--seed", help="Random seed", default=42)

    return parser.parse_args()


def load_dataloader(args):
    dataloaders = sidechainnet.load(
        casp_version=args.casp_version, 
        thinning=args.scn_thinning, 
        with_pytorch="dataloaders",
        batch_size=1, dynamic_batching=False
    )

    return dataloaders


def save_as_txt(*items, path):
    if "/" in path: 
        folder = "/".join(path.split("/")[:-1])
        os.makedirs(folder, exist_ok=True)

    with open(path, "a") as f:
        for item in items: 
            try: 
                for line in item: 
                    f.write(str(line)+"\n")
            except Exception as e:
                print("Error in saving:", e) 
                f.write(str(item))

            f.write("\n")


def init_wandb_config(args):
    wandb.init(project=args.wb_proj, entity=args.wb_entity, name=args.run_name)

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.seed = args.seed
    config.device = args.device
    config.embedder_model = args.embedder_model
    config.scn_version = str(args.casp_version)+"-"+str(args.scn_thinning)
    config.min_len = args.min_len
    config.max_len = args.max_len
    config.xray = bool(args.xray)
    config.frac_true_torsions = bool(frac_true_torsions)

    # model hyperparams
    config.num_layers = args.num_layers
    config.emb_dim = args.emb_dim
    config.hidden = args.hidden
    config.mlp_hidden = [128, 4 if args.angularize == 0 else args.angularize]  # 4 # 64
    config.act = args.act  # "silu"
    config.layer_type = args.layer_type  # "LSTM"
    config.input_dropout = args.input_dropout

    config.bidirectional = bool(args.bidirectional)  # True
    config.max_recycles_train = args.num_recycles_train  #  set up to 1 to speed things
    config.angularize = bool(args.angularize)

    config.refiner_args = dict(refiner_args)
    
    return config


def init_and_train(args):
    config = init_wandb_config(args)

    dataloaders = load_dataloader(args)
    print('loaded dataloaders')

    embedder = get_embedder(config, config.device)
    print('loaded embedder')

    config = init_wandb_config(args)

    results = run_train_schedule(dataloaders, embedder, config, args)

    save_as_txt(
        [args, config, *results], 
        path = "rgn2_models/"+wandb.run.name.replace("/", "_")+"_logs.txt"
    )


def run_train_schedule(dataloaders, embedder, config, args):
    valid_log_acc = []
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
                       refiner_args=refiner_args,
                       ).to(device)
    
    if args.resume_name is not None: 
        model.load_state_dict(torch.load(args.resume_name))

    # 3. Log gradients and model parameters
    wandb.watch(model)

    steps = get_training_schedule(args)

    resume = True  # declare new optim
    for i, (batch_num, ckpt, lr, batch_size, max_len, clip, loss_f, seed) in enumerate(steps):
        # reconfig batch otpions
        wandb.log({
            'learning_rate': lr,
            'batch_size': batch_size
        }, commit=False)

        if sum([steps[j][0] for j in range(i)]) < args.resume_iters: continue

        if resume:
            if seed is not None:
                set_seed(seed)
            get_prot_ = mp_nerf.utils.get_prot(
                dataloader_=dataloaders,
                vocab_=VOCAB,
                min_len=config.min_len, max_len=max_len,  # MAX_LEN,
                verbose=False, subset="train", 
                xray_filter=config.xray,
            )

            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            resume = False
        else:
            for g in optimizer.param_groups:
                g['lr'] = lr

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
            config=config,
        )

        metric = np.mean([x["drmsd"] for x in metrics_stuff[0][-5*batch_size:]])
        print("\nCheckpoint {0} @ {1}, pass @ {2}. Metrics mean train = {1}\n".format(
            i, ckpt, metric, metrics_stuff[-1]
        ))

        # save
        os.makedirs('rgn2_models', exist_ok=True)
        save_path = "rgn2_models/"+wandb.run.name.replace("/", "_")+"@_{0}K.pt".format(
            sum(p[0] for p in steps[:i+1]) // 1000
        )
        torch.save(model.state_dict(), save_path)

        ## VALIDATING
        for valid_set in [10, 20, 30, 40, 50, 70, 90]:
            print("Validating "+str(valid_set))
            tic = time.time()
            get_prot_valid_ = mp_nerf.utils.get_prot( 
                dataloader_=dataloaders, 
                vocab_=VOCAB, # mp_nerf.utils.
                min_len=args.min_len_valid, max_len=max_len, 
                verbose=False, subset="valid-"+str(valid_set)
            )
            # get num of unique, full-masked proteins
            seqs = []
            for i, prot_args in enumerate(dataloaders["valid-"+str(valid_set)].dataset):
                # (id, int_seq, mask, ... , str_seq)
                length = len(prot_args[-1]) 
                if args.min_len_valid < length < max_len and sum( prot_args[2] ) == length:
                    seqs.append( prot_args[-1] )

            metrics_stuff_eval = predict(
                get_prot_= get_prot_valid_, 
                steps = len(set(seqs)), # 24 for MIN_LEN=70
                model = model,
                embedder = embedder, 
                return_preds = True,
                log_every = 4,
                accumulate_every = len(set(seqs)),
                seed = 0, # 42
                mode = "fast_test", # "test" # "test" is for AR, "fast_test" is for iterative
                recycle_func = lambda x: 10, # 5 # 3 # 2 
                wandbai = False,
            )
            preds_list_eval, metrics_list_eval, metrics_stats_eval = metrics_stuff_eval
            print("\n", "Eval Results:", sep="")
            for k,v in metrics_stats_eval.items():
                offset = " " * ( max(len(ki) for ki in metrics_stats_eval.keys()) - len(k) )
                print(k + offset, ":", v)
            print("\n")
            print("Time taken: ", time.time()-tic, "\n")
        
        # save logs to compare runs - wandb not enough
        valid_log_acc.append(metrics_stats_eval)

        # ABORT OR CONTINUE: mean of last 5 batches below ckpt
        if metric > ckpt:
            print("ABORTING")
            print("Didn't pass ckpt {0} @ drmsd = {1}, but instead drmsd = {2}".format(
                i, ckpt, metric
            ))
            break

    os.makedirs('rgn2_models', exist_ok=True)
    save_path = "rgn2_models/"+wandb.run.name.replace("/", "_")+"@_{0}K.pt".format(
        sum(p[0] for p in steps[:i+1]) // 1000
    )
    torch.save(model.state_dict(), save_path)

    ### TEST
    tic = time.time()
    get_prot_test_ = mp_nerf.utils.get_prot( 
        dataloader_=dataloaders, 
        vocab_=VOCAB, # mp_nerf.utils.
        min_len=args.min_len_valid, max_len=max_len, 
        verbose=False, subset="test"
    )
    # get num of unique, full-masked proteins
    seqs = []
    for i, prot_args in enumerate(dataloaders["test"].dataset):
        # (id, int_seq, mask, ... , str_seq)
        length = len(prot_args[-1]) 
        if args.min_len_valid < length < max_len and sum( prot_args[2] ) == length:
            seqs.append( prot_args[-1] )

    metrics_stuff_test = predict(
        get_prot_= get_prot_test_, 
        steps = len(set(seqs)), # 24 for MIN_LEN=70
        model = model,
        embedder = embedder, 
        return_preds = True,
        log_every = 4,
        accumulate_every = len(set(seqs)),
        seed = 0, # 42
        mode = "fast_test", # "test" # "test" is for AR, "fast_test" is for iterative
        recycle_func = lambda x: 10, # 5 # 3 # 2 
        wandbai = False,
    )
    preds_list_test, metrics_list_test, metrics_stats_test = metrics_stuff_test
    print("\n", "Test Results:", sep="")
    for k,v in metrics_stats_test.items():
        offset = " " * ( max(len(ki) for ki in metrics_stats_test.keys()) - len(k) )
        print(k + offset, ":", v)
    print("\n")
    print("Time taken: ", time.time()-tic, "\n")

    return metrics_stats_eval, valid_log_acc



def get_training_schedule(args):
    loss_f = " metrics['drmsd'].mean() / len(infer['seq']) " 

    #         steps, ckpt, lr , bs , max_len, clip, loss_f
    return [[32000, 135   , 1e-3, 16  , args.max_len, None, loss_f, 42  , ],
            [64000, 135   , 1e-3, 32  , args.max_len, None, loss_f, 42  , ],
            [32000, 135   , 1e-4, 32  , args.max_len, None, loss_f, 42  , ],]


if __name__ == '__main__':
    # # new run
    # nohup python rgn2-replica/scripts/train_rgn2.py --device cuda:3 --wb_entity hypnopump17 --wb_proj rgn2_replica \
    # --run_name RGN2X_vanillaLSTM_full_run --min_len_valid 0 --xray 1 > RGN2X_vanillaLSTM_full_run_logs.txt 2>&1 &

    # # continue
    # nohup python rgn2-replica/scripts/train_rgn2.py --device cuda:3 --wb_entity hypnopump17 --wb_proj rgn2_replica \
    # --resume_name rgn2_models/RGN2X_vanillaLSTM_full_run@_32K.pt --resume_iters 32000 \
    # --run_name RGN2X_vanillaLSTM_full_run --min_len_valid 0 --xray 1 > RGN2X_vanillaLSTM_full_run_logs.txt 2>&1 &

    args = parse_arguments()
    init_and_train(args)
