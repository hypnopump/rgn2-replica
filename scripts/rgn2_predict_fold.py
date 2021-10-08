# Author: Eirc Alcaide (@hypnopump)
from pathlib import Path
import json
import torch
# process
import argparse
# custom
import sidechainnet
from rgn2_replica import *
from rgn2_replica.embedders import get_embedder
from rgn2_replica.rgn2_utils import seqs_from_fasta
from rgn2_replica.rgn2_trainers import infer_from_seqs


def parse_arguments():
    # !python redictor.py --input proteins.fasta --model ../rgn2_models/baseline_run@_125K.pt --device 2
    parser = argparse.ArgumentParser('Predict with RGN2 model')
    input_group = parser.add_mutually_exclusive_group(required=True)

    # inputs
    parser.add_argument("--input", help="FASTA or MultiFASTA with protein sequences to predict")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for prediction")
    input_group.add_argument("--pdb_input", type=str, help="PDB file for refinement")

    # model params - same as training
    parser.add_argument("--embedder_model", help="Embedding model to use", default='esm1b')
    parser.add_argument("--num_layers", help="num rnn layers", type=int, default=2)
    parser.add_argument("--emb_dim", help="embedding dimension", type=int, default=1280)
    parser.add_argument("--hidden", help="hidden dimension", type=int, default=1024)
    parser.add_argument("--act", help="hidden activation", type=str, default="silu")
    parser.add_argument("--layer_type", help="rnn layer type", type=str, default="LSTM")
    parser.add_argument("--input_dropout", help="input dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", help="bidirectionality", type=bool, default=0)
    parser.add_argument("--angularize", help="angularization units. 0 for reg", type=int, default=0)
    parser.add_argument("--num_recycles_pred", type=int, default=10, 
                        help="number of recycling iters. set to 1 to speed inference.",)

    # rosetta
    parser.add_argument("--rosetta_refine", type=int, default=0, help="refine output with Rosetta. 0 for no refinement")
    parser.add_argument("--rosetta_relax", type=int, default=0, help="relax output with Rosetta. 0 for no relax.")
    parser.add_argument("--coord_constraint", type=float, default=1.0, help="constraint for Rosetta relax. higher=stricter.")
    parser.add_argument("--device", help="Device ('cpu', cuda:0', ...)", type=str, required=True)

    # outputs
    parser.add_argument("--output_path", type=str, default=None,  #  prot_id.fasta -> prot_id_0.fasta,
                        help="path for output .pdb files. Defaults to input name + seq num")
    # refiner params
    parser.add_argument("--af2_refine", type=int, default=0, help="refine output with AlphaFold2. 0 for no refinement")
    parser.add_argument("--refiner_args", help="args for refiner module", type=json.loads, default={})
    parser.add_argument("--seed", help="Random seed", default=101)

    args = parser.parse_args()
    args.bidirectional = bool(args.bidirectional)
    args.angularize = bool(args.angularize)
    args.refiner_args = dict(args.refiner_args)

    # mod parsed args
    if args.output_path is None:
        args.output_path = args.input.replace(".fasta", "_")

    return args


def load_model(args):
    mlp_hidden = [128, 4 if args.angularize == 0 else args.angularize]  # 4 # 64
    config = args
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
                       refiner_args=config.refiner_args,
                       ).to(args.device)

    model.load_my_state_dict(torch.load(args.model, map_location=args.device))
    model = model.eval()

    # # Load ESM-1b model
    embedder = get_embedder(args, args.device)

    return model.eval(), embedder


def predict(model, embedder, seq_list, seq_names, args):
    # batch wrapper
    pred_dict = {}
    num_batches = len(seq_list) // args.batch_size + \
        int(bool(len(seq_list) % args.batch_size))

    for i in range(num_batches):
        aux = infer_from_seqs(
            seq_list[args.batch_size*i: args.batch_size*(i+1)],
            model=model,
            embedder=embedder,
            recycle_func=lambda x: int(args.recycle),
            device=args.device
        )
        for k, v in aux.items():
            try:
                pred_dict[k] += v
            except KeyError:
                pred_dict[k] = v

    # save structures
    out_files = []
    for i, seq in enumerate(seq_list):
        struct_pred = sidechainnet.StructureBuilder(
            pred_dict["int_seq"][i].cpu(),
            crd=pred_dict["coords"][i].reshape(-1, 3).cpu()
        )
        out_files.append(args.output_path+str(i)+"_"+seq_names[i]+".pdb")
        struct_pred.to_pdb(out_files[-1])

        print("Saved", out_files[-1])

    return pred_dict, out_files


def refine(seq_list, pdb_files, args):
    # refine structs
    if args.rosetta_refine:
        result = rosetta_refine(seq_list, pdb_files, args)
    elif args.af2_refine:
        result = af2_refine(pdb_files, args)

    return result


def rosetta_refine(seq_list, pdb_files, args):
    from rgn2_replica.rgn2_refine import quick_refine, relax_refine

    for i, seq in enumerate(seq_list):
        # only refine
        if args.rosetta_relax == 0:
            quick_refine(
                in_pdb=pdb_files[i],
                out_pdb=pdb_files[i][:-4]+"_refined.pdb",
                min_iter=args.rosetta_refine
            )
        # refine and relax
        else:
            relax_refine(
                pdb_files[i],
                out_pdb=pdb_files[i][:-4]+"_refined_relaxed.pdb",
                min_iter=args.rosetta_refine,
                relax_iter=args.rosetta_relax,
                coord_constraint=args.coord_constraint,
            )
        print(pdb_files[i], "was refined successfully")


def af2_refine(pdb_files, args):
    from alphafold.relax import relax
    from alphafold.common import protein

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=20)

    out_dir = Path(args.output_path)
    relaxed_pdbs_files = []
    for pdb_file in pdb_files:
        pdb_file = Path(pdb_file)
        pdb_str = pdb_file.read_text()
        prot = protein.from_pdb_string(pdb_str)
        min_pdb, debug_data, violations = amber_relaxer.process(prot=prot)

        dest_file = out_dir / pdb_file.name
        dest_file.write_text(min_pdb)

        relaxed_pdbs_files.append(dest_file)

    return relaxed_pdbs_files


def predict_refine():
    args = parse_arguments()

    # get sequences
    if args.input is not None:
        seq_list, seq_names = seqs_from_fasta(args.input, names=True)
    else:
        seq_list, seq_names = None, None

    if args.model is not None:
        model, embedder = load_model(args)
        _, pdb_files = predict(model, embedder, seq_list, seq_names, args)
    else:
        pdb_files = [args.pdb_input]

    relaxed_pdbs = refine(seq_list, pdb_files, args)

    print('result files:')
    print(relaxed_pdbs)


if __name__ == "__main__":
    predict_refine()
