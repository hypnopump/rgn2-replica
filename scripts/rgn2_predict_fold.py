# Author: Eirc Alcaide (@hypnopump)
from pathlib import Path
import torch
# process
import argparse
# custom
import sidechainnet
from rgn2_replica import *
from rgn2_replica.rgn2_refine import *
from rgn2_replica.embedders import get_embedder
from rgn2_replica.rgn2_utils import seqs_from_fasta
from rgn2_replica.rgn2_trainers import infer_from_seqs


def parse_arguments():
    # !python redictor.py --input proteins.fasta --model ../rgn2_models/baseline_run@_125K.pt --device 2
    parser = argparse.ArgumentParser()

    # inputs
    parser.add_argument("--input", help="FASTA or MultiFASTA with protein sequences to predict")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for prediction")

    # model
    parser.add_argument("--model", type=str, help="Model file for prediction")
    parser.add_argument("--embedder_model", help="Embedding model to use", default='esm1b')
    parser.add_argument("--emb_dim", help="embedding dimension", type=int, default=1280)
    parser.add_argument("--num_layers", help="num rnn layers", type=int, default=2)
    parser.add_argument("--hidden", help="hidden dimension", type=int, default=1024)
    parser.add_argument("--bidirectional", help="bidirectionality", type=bool, default=0)
    parser.add_argument("--act", help="hidden activation", type=str, default="silu")
    parser.add_argument("--layer_type", help="rnn layer type", type=str, default="LSTM")
    parser.add_argument("--input_dropout", help="input dropout", type=float, default=0.5)
    parser.add_argument("--angularize", help="angularization units. 0 for reg", type=int, default=0)

    # refinement options
    parser.add_argument("--rosetta_refine", type=int, default=0, help="refine output with Rosetta. 0 for no refinement")
    parser.add_argument("--rosetta_relax", type=int, default=0, help="relax output with Rosetta. 0 for no relax.")
    parser.add_argument("--coord_constraint", type=float, default=1.0, help="constraint for Rosetta relax. higher=stricter.")
    parser.add_argument("--recycle", default=10, help="Recycling iterations")
    parser.add_argument("--device", default="cpu", help="['cpu', 'cuda:0', 'cuda:1', ...], cpu is slow!")

    # outputs
    parser.add_argument("--output_path", type=str, default=None,  #  prot_id.fasta -> prot_id_0.fasta,
                        help="path for output .pdb files. Defaults to input name + seq num")
    args = parser.parse_args()

    # mod parsed args
    if args.output_path is None:
        args.output_path = args.input.replace(".fasta", "_")

    return args


def load_model(args):
    mlp_hidden = [128, 4 if args.angularize == 0 else args.angularize]  # 4 # 64
    model = RGN2_Naive(
        layers=args.num_layers,
        emb_dim=args.emb_dim+4,
        hidden=args.hidden,
        bidirectional=args.bidirectional,
        mlp_hidden=mlp_hidden,
        act=args.act,
        layer_type=args.layer_type,
        input_dropout=args.input_dropout,
        angularize=args.angularize,
    ).to(args.device)

    model.load_state_dict(torch.load(args.model, map_location=args.device))

    return model.eval()


def predict(model, seq_list, seq_names, args):
    # Load ESM-1b model
    embedder = get_embedder(args, args.device)

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
        rosetta_refine(seq_list, pdb_files, args)
    else:
        af2_refine(pdb_files)

    print("All tasks done. Exiting...")


def rosetta_refine(seq_list, pdb_files, args):
    from rgn2_replica.rgn2_refine import *

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


def af2_refine(pdb_files):
    from alphafold.relax import relax
    from alphafold.common import protein
    
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=20)

    relaxed_pdbs = []
    for pdb_file in pdb_files:
        pdb_str = Path(pdb_file).read_text()
        prot = protein.from_pdb_string(pdb_str)
        min_pdb, debug_data, violations = amber_relaxer.process(prot=prot)

        relaxed_pdbs.append(min_pdb)

    return relaxed_pdbs


if __name__ == "__main__":
    args = parse_arguments()

    model = load_model(args)

    # get sequences
    seq_list, seq_names = seqs_from_fasta(args.input, names=True)

    pred_dict, pdb_files = predict(model, seq_list, seq_names, args)

    relaxed_pdbs = refine(seq_list, pdb_files, args)
