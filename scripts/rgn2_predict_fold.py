# Author: Eirc Alcaide (@hypnopump)
# process
import argparse
# custom
import sidechainnet
from rgn2_replica import *
from rgn2_replica.rosetta_refine import *
from rgn2_replica.utils import seqs_from_fasta
from rgn2_trainers import infer_from_seqs


if __name__ == "__main__": 
	# !python redictor.py --input proteins.fasta --model ../rgn2_models/baseline_run@_125K.pt --device 2
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("--input", help="FASTA or MultiFASTA with protein sequences to predict")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for prediction")
    # model
    parser.add_argument("--embedder_model", type=str, help="esm1b")
    parser.add_argument("--model", type=str, help="Model file for prediction")
    parser.add_argument("--rosetta_refine", type=int, default=0, help="refine output with Rosetta. 0 for no refinement")
    parser.add_argument("--rosetta_relax", type=int, default=0, help="relax output with Rosetta. 0 for no relax.")
    parser.add_argument("--coord_constraint", type=float, default=1.0, help="constraint for Rosetta relax. higher=stricter.")
    parser.add_argument("--recycle", default=10, help="Recycling iterations")
    parser.add_argument("--device", default="cpu", help="['cpu', 'cuda:0', 'cuda:1', ...], cpu is slow!")
    # outputs
    parser.add_argument("--output_path", type=str, default=None, # prot_id.fasta -> prot_id_0.fasta,
                        help="path for output .pdb files. Defaults to input name + seq num")
    args = parser.parse_args()
    # mod parsed args
    if args.output_path is None: 
        args.output_path = args.input.replace(".fasta", "_")

    # get sequences
    seq_list, seq_names = seqs_from_fasta(args.input, names=True)

    # predict structures
    model = RGN2_Naive(
        layers = 2, 
        emb_dim = args.emb_dim+4,
        hidden = 1024, 
        bidirectional = True, 
        mlp_hidden = [128, 4],
        act="silu", 
        layer_type="LSTM",
        input_dropout=0.5,
        angularize=False,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model))
    model = model.eval()
    # # Load ESM-1b model
    embedder = get_embedder(args, device)

    # batch wrapper
    pred_dict = {}
    num_batches = len(seq_list) // args.batch_size + \
                  int(bool(len(seq_list) % args.batch_size))

    for i in range( num_batches ):
        aux = infer_from_seqs(
            seq_list[args.batch_size*i : args.batch_size*(i+1)], 
            model = model, 
            embedder = embedder, 
            recycle_func=lambda x: int(args.recycle),
            device=args.device
        )
        for k,v in aux.items(): 
            try: pred_dict[k] += v
            except KeyError: pred_dict[k] = v

    # save structures
    out_files = []
    for i, seq in enumerate(seq_list): 
        struct_pred = sidechainnet.StructureBuilder(
            pred_dict["int_seq"][i].cpu(), 
            crd = pred_dict["coords"][i].reshape(-1, 3).cpu() 
        ) 
        out_files.append( args.output_path+str(i)+"_"+seq_names[i]+".pdb" )
        struct_pred.to_pdb( out_files[-1] )
 
        print("Saved", out_files[-1])

    # refine structs
    if args.rosetta_refine: 
        from typing import Optional

        for i, seq in enumerate(seq_list): 
            # only refine
            if args.rosetta_relax == 0: 
                quick_refine(
                    in_pdb = out_files[i],
                    out_pdb = out_files[i][:-4]+"_refined.pdb",
                    min_iter = args.rosetta_refine
                )
            # refine and relax
            else:
                relax_refine(
                    out_files[i], 
                    out_pdb=out_files[i][:-4]+"_refined_relaxed.pdb", 
                    min_iter = args.rosetta_refine, 
                    relax_iter = args.rosetta_relax,
                    coord_constraint = args.coord_constraint,
                )
            print(out_files[i], "was refined successfully")

    print("All tasks done. Exiting...")


