# Author: Eric Alcaide ( @hypnopump )

import time
import gc
import random 
import numpy as np

import torch
from einops import rearrange, repeat
from functools import partial

import mp_nerf
from rgn2_replica.utils import *
from rgn2_replica.rgn2_utils import *

# logging
WANDB = True
try: 
    import wandb
except: 
    WANDB = False
    print("Failed to import wandb. LOGGING NOT AVAILABLE")



def batched_inference(*args, model, embedder, batch_converter=None,
                             mode="test", device="cpu", recycle_func=lambda x: 1):
    """ Inputs: 
        * args: iterable of outputs from mp_nerf.utils.get_prot()
                ( seq, int_seq, true_coords, angles, padding_seq, mask, pid )
        * model: torch.nn.Module / extension. model to do inference on
        * embedder: func generator of NLP embeddings or torhc.nn.Embedding
        * mode: str. [~"train"~, "test"]. For either next pred, or full pred
                      Only works for test atm.
        * device: str or torch.device 
        * recycle_func: func. returns number of reycling iters
        Outputs: 
        * (B, L, 4)
    """
    # bacth tokens,masks and to device - Out: (B, L)
    batch_dim = len(args)
    max_seq_len = max(x[1].shape[-1] for x in args) 
    # create scaffolds
    int_seq = torch.ones(batch_dim, max_seq_len, dtype=torch.long) * 20 # padding tok
    mask = torch.zeros(*int_seq.shape, dtype=torch.bool)
    true_coords = torch.zeros(int_seq.shape[0], int_seq.shape[1]*14, 3, device=device)
    # fill scaffolds
    for i,arg in enumerate(args): 
        mask[i, :arg[1].shape[-1]] = arg[-2]
        int_seq[i, :arg[1].shape[-1]] = arg[1]
        true_coords[i, :arg[1].shape[-1]*14] = arg[2]
    
    mask = mask.bool().to(device)
    coords = rearrange(true_coords, 'b (l c) d -> b l c d', c=14)
    ca_trace = coords[..., 1, :]
    coords_rebuilt = mp_nerf.proteins.ca_bb_fold( ca_trace ) # beware extremes

    # calc angle labels
    angles_label_ = torch.zeros(*ca_trace.shape[:-1], 2, dtype=torch.float, device=device)
    for i, arg in enumerate(args): 
        length = arg[1].shape[-1]
        angles_label_[i, 1:length-1, 0] = mp_nerf.utils.get_cosine_angle( 
            ca_trace[i, :length-2 , :], 
            ca_trace[i, 1:length-1, :],
            ca_trace[i, 2:length  , :],
        )
        angles_label_[i, 2:length-1, 1] = mp_nerf.utils.get_dihedral( 
            ca_trace[i, :length-3 , :], 
            ca_trace[i, 1:length-2, :],
            ca_trace[i, 2:length-1, :],
            ca_trace[i, 3:length  , :],
        ) 
    points_label = mp_nerf.ml_utils.angle_to_point_in_circum(angles_label_) # (B, L, 2, 2)

    # include angles of previous AA as input
    points_input = points_label.clone()
    points_input = torch.cat([
        points_input[..., :1, :, :], points_input[..., :-1, :, :]
    ], dim=-3)
    angles_input =  rearrange(points_input, "... c d -> ... (c d)")

    # EMBEDD
    if isinstance(embedder, torch.nn.Embedding): 
        embedds = embedder(int_seq.to(device))
    else: 
        embedds = get_esm_embedd(int_seq, embedd_model=embedder, batch_converter=batch_converter)
    
    embedds = torch.cat([
        embedds, 
        # don't pass angles info - just 0 at start (sin=0, cos=1)
        angles_input*0 + points_label[:, :1], 
    ], dim=-1)
    
    # PREDICT
    if mode == "train": 
        # get angles
        preds, r_iters = model.forward(embedds, mask=mask,
                                       recycle=recycle_func(None))      # (B, L, 4)
    elif mode == "test": 
        preds, r_iters = model.predict_fold(embedds, mask=mask,
                                            recycle=recycle_func(None)) # (B, L, 4)   
    elif mode == "fast_test":  
        embedds[:, :, -4:] = embedds[:, :, -4:] * 0. # zero out angle features
        preds, r_iters = model.forward(embedds, mask=mask,
                                       recycle=recycle_func(None))     # , inter_recycle=True

    points_preds = rearrange(preds, '... (a d) -> ... a d', a=2)       # (B, L, 2, 2)
    
    # POST-PROCESS
    points_preds, ca_trace_pred, frames_preds, wrapper_pred = rgn2_replica.rgn2.pred_post_process(
        points_preds, mask=mask 
    )

    # get frames for for later fape
    bb_ca_trace_rebuilt, frames_labels = mp_nerf.proteins.ca_from_angles( 
        points_label.reshape(points_label.shape[0], -1, 4) # (B, L, 2, 2) -> (B, L, 4)
    ) 
    

    return (
        {
            "seq": arg[0],
            "int_seq": arg[1], 
            "angles": arg[2],
            "padding_seq": arg[3],
            "mask": arg[4],
            "pid": arg[5],
            # labels
            "true_coords": true_coords[i:i+1, :arg[1].shape[-1]*14], # (1, (L C), 3)
            "coords": coords[i:i+1, :arg[1].shape[-1]],              # (1, L, C, 3)
            "ca_trace": ca_trace[i:i+1, :arg[1].shape[-1]],          # (1, L, 3)
            "points_label": points_label[i:i+1, :arg[1].shape[-1]],  # (1, L, 4)
            "frames_labels": frames_labels[i, :arg[1].shape[-1]],    # (L, 3, 3)
            # inputs
            "points_input": angles_input[i:i+1, :arg[1].shape[-1]],  # (1, L, 4)
            # preds
            "wrapper_pred": wrapper_pred[i:i+1, :arg[1].shape[-1]],  # (1, L, C, 3)
            "ca_trace_pred": ca_trace_pred[i:i+1, :arg[1].shape[-1]],# (1, L, C, 3) 
            "points_preds": points_preds[i:i+1, :arg[1].shape[-1]],  # (1, L, 4)
            "frames_preds": frames_preds[i, :arg[1].shape[-1]],      # (L, 3, 3)
            # (iters, L, 4) - only if available
            "r_iters": r_iters[i, :, :arg[1].shape[-1]] if len(r_iters.shape) > 2 else r_iters[i], 
        } for i,arg in enumerate(args)
    )


def inference(*args, model, embedder, batch_converter=None, 
                     mode="train", device="cpu", recycle_func=lambda x: 1):
    """ Inputs: 
        * args: output from mp_nerf.utils.get_prot()
        * model: torch.nn.Module / extension. model to do inference on
        * embedder: func generator of NLP embeddings or torhc.nn.Embedding
        * mode: str. ["train", "test", "fast_test"]. For either next pred, 
                     or full pred. "test" does AR, "fast_test" does iterative
                     refinement (good approximation and 10x faster)
        * device: str or torch.device 
        * recycle_func: func. returns number of reycling iters
        Outputs: 
        * output_dict
    """
    seq, int_seq, true_coords, angles, padding_seq, mask, pid = args

    int_seq = int_seq.unsqueeze(0)
    mask = mask.bool().to(device)
    coords = rearrange(true_coords, '(l c) d -> () l c d', c=14).to(device)
    ca_trace = coords[..., 1, :]
    coords_rebuilt = mp_nerf.proteins.ca_bb_fold( ca_trace ) 
    # mask for thetas and chis
    angles_mask = torch.ones(*ca_trace.shape[:-1], 2, dtype=torch.bool, device=device)
    angles_mask[..., 1:-1, :] = False
    angles_label_ = torch.zeros(*ca_trace.shape[:-1], 2, dtype=torch.float, device=device)
    angles_label_[..., 1:-1, 0] = mp_nerf.utils.get_cosine_angle( 
        ca_trace[..., :-2 , :], 
        ca_trace[..., 1:-1, :],
        ca_trace[..., 2:  , :],
    )
    angles_label_[..., 2:-1, 1] = mp_nerf.utils.get_dihedral( 
        ca_trace[..., :-3 , :], 
        ca_trace[..., 1:-2, :],
        ca_trace[..., 2:-1, :],
        ca_trace[..., 3:  , :],
    ) 
    points_label = mp_nerf.ml_utils.angle_to_point_in_circum(angles_label_) # (B, L, 2, 2)

    # include angles of previous AA as input
    points_input = points_label.clone()
    points_input = torch.cat([
        points_input[..., :1, :, :], points_input[..., :-1, :, :]
    ], dim=-3)
    angles_input =  rearrange(points_input, "... c d -> ... (c d)")

    # PREDICT
    if isinstance(embedder, torch.nn.Embedding): 
        embedds = embedder(int_seq)
    else: 
        embedds = get_esm_embedd(int_seq, embedd_model=embedder, batch_converter=batch_converter)
    
    embedds = torch.cat([
        embedds, 
        # don't pass angles info - just 0 at start (sin=0, cos=1)
        angles_input*0 + points_label[:, :1], 
    ], dim=-1)
    
    if mode == "train": 
        preds, r_iters = model.forward(embedds, mask=mask,
                                       recycle=recycle_func(None))       # (B, L, 4)
    elif mode == "test": 
        preds, r_iters = model.predict_fold(embedds, mask=mask,
                                            recycle=recycle_func(None))  # (B, L, 4)
    elif mode == "fast_test": 
        embedds[:, :, -4:] = embedds[:, :, -4:] * 0. # zero out angle features
        preds, r_iters = model.forward(embedds, mask=mask,
                                       recycle=recycle_func(None))     # , inter_recycle=True
        
    points_preds = rearrange(preds, '... (a d) -> ... a d', a=2)       # (B, L, 2, 2)

    # post-process
    points_preds, ca_trace_pred, frames_preds, wrapper_pred = rgn2_replica.rgn2.pred_post_process(
        points_preds, mask=mask 
    )

    # get frames for for later fape
    bb_ca_trace_rebuilt, frames_labels = mp_nerf.proteins.ca_from_angles( 
        points_label.reshape(1, -1, 4) # (B, L, 2, 2) -> (B, L, 4)
    ) 


    return {
        "seq": seq,
        "int_seq": int_seq, 
        "angles": angles,
        "padding_seq": padding_seq,
        "mask": mask,
        "pid": pid, 
        # labels
        "true_coords": true_coords,  # (B, (L C), 3)
        "coords": coords,            # (B, L, C, 3)
        "ca_trace": ca_trace, 
        "points_label": points_label,
        "frames_labels": frames_labels, # (L, 3, 3)
        # inputs
        "points_input": angles_input,
        # preds
        "wrapper_pred": wrapper_pred,   # (1, L, C, 3) 
        "ca_trace_pred": ca_trace_pred, # (1, L, C, 3) 
        "points_preds": points_preds, 
        "frames_preds": frames_preds,   # (L, 3, 3)
        "r_iters": r_iters, 
    }


def predict(get_prot_, steps, model, embedder, batch_converter=None, return_preds=True,
         accumulate_every=1, log_every=None, seed=None, wandbai=False, 
         recycle_func=lambda x: 1, mode="test"):
    """ Performs a batch prediction. 
        Can return whole list of preds or just metrics.
        Inputs: 
        * get_prot_: mp_nerf.utils.get_prot() iterator 
        * steps: int. number of steps to predict
        * model: torch model
        * embedder: callable to get NLP embeddings from
        * vocab_:
        * return_preds: bool. whether to return predictions as well
        * accumulate_every: int. batch size. 
        * log_every: int or None. print on screen every X batches.
        * seed: 
        * wandbai: 
        * recycle_func: func. number of recycle iters per sample
        * mode: str. one of "test" (ar prediction) or "fast_test"
                     good on-the-ground approx if recycle ~ 10.
        Outputs: 
        * preds_list: (steps, dict) list
        * metrics_list: (steps, dict) list
    """
    model = model.eval()
    device = next(model.parameters()).device

    preds_list, metrics_list = [], []

    b = 0
    tic = time.time()
    while b < (steps//accumulate_every): 
        if b == 0 and seed is not None:
            set_seed(seed)
        
        # get + predict
        with torch.no_grad():
            prots = [ next(get_prot_) for i in range(accumulate_every) ]
            infer_batch = batched_inference(
                *prots, 
                model=model, embedder=embedder, batch_converter=batch_converter, 
                mode=mode, device=device, recycle_func=recycle_func
            )
        # calculate metrics
        for infer in infer_batch: 
            # calc loss terms || baselines for next-term: torsion=2, fape=0.95
            torsion_loss = mp_nerf.ml_utils.torsion_angle_loss(
                pred_points=infer["points_preds"][:, :-1], # (B, L-2, 2, 2) 
                true_points=infer["points_label"][:, :-1], # (B, L-2, 2, 2)
            )

            # violation loss btween calphas - L1
            dist_mat = torch.cdist(infer["wrapper_pred"][:, :, 1], 
                                   infer["wrapper_pred"][:, :, 1],) + \
                       torch.eye(infer["wrapper_pred"][0, :, 1].shape[0], 
                                 device=device).unsqueeze(0) * 5
            viol_loss = -torch.minimum(dist_mat - 3.75, torch.zeros_like(dist_mat)) 
            
            # calc metrics
            log_dict = {
                "torsion_loss": torsion_loss.mean().item(),
                "viol_loss": viol_loss.mean().item()           
            }
            metrics = mp_nerf.proteins.get_protein_metrics(
                true_coords=infer["coords"], pred_coords=infer["wrapper_pred"], detach=True
            )
            log_dict.update({
                k:v.mean().item() for k,v in metrics.items() if "wrap" not in k
            })

            # record
            metrics_list.append( log_dict )
            if wandbai and WANDB:
                wandb.log(metrics_list[-1])

            if return_preds: 
                # pass all to cpu - free mem in the gpu
                for k,v in infer.items(): 
                    if isinstance(v, torch.Tensor): 
                        infer[k] = v.cpu()
                preds_list.append( infer )
                # free mem - slow
                del infer
                gc.collect()

        # log
        if log_every and (b-1) % log_every == 0: 
            tac = time.time()
            print("Batch {0}/{1}, metrics_last_ex = {2}. Took: {3} seconds".format(
                (b-1) // log_every,  steps // log_every, log_dict, np.round(tac-tic, decimals=3)
            ))
            tic = tac

        # go to next bacth
        b += 1

    metrics_stats = { "eval_"+k : \
                      np.mean([ metrics[k] for metrics in metrics_list ]) \
                      for k in metrics_list[0].keys()
                    }

    return preds_list, metrics_list, metrics_stats


def train(get_prot_, steps, model, embedder, optim, batch_converter=None, loss_f=None, 
          clip=None, accumulate_every=1, log_every=None, seed=None, wandbai=False, 
          recycle_func=lambda x: 1): 
    """ Performs a batch prediction. 
        Can return whole list of preds or just metrics.
        Inputs: 
        * get_prot_: mp_nerf.utils.get_prot() iterator 
        * steps: int. number of steps to predict
        * embedder: callable to get NLP embeddings from
        * optim: torch.Opim for training.
        * loss_f: str or None. custom expression for the loss
        * clip: float or None. Gradient clipping
        * accumulate_every: int. effective batch size for backprop
        * log_every: int or None. print every X number of batches.
        * seed: int or None.
        * wandbai: bool. whether to log to W&B
        * recycle_func: func. number of recycle iters per sample
        Outputs: 
        * preds_list: (steps, dict) list
        * metrics_list: (steps, dict) list
    """
    model = model.train()
    device = next(model.parameters()).device

    metrics_list = []
    b = 0
    loss = 0.
    tic = time.time()
    while b < (steps//accumulate_every): # steps: # 
        if b == 0 and seed is not None: 
            set_seed(seed)

        # get data + predict
        prots = [ next(get_prot_) for i in range(accumulate_every) ]
        infer_batch = batched_inference(
            *prots, 
            model=model, embedder=embedder, batch_converter=batch_converter, 
            mode="train", device=device, recycle_func=recycle_func
        )

        # calculate metrics
        loss_batch = 0
        for i, infer in enumerate(infer_batch):
            # calc loss terms 
            torsion_loss = mp_nerf.ml_utils.torsion_angle_loss(
                pred_points=infer["points_preds"][:, :-1], # (B, L-2, 2, 2) 
                true_points=infer["points_label"][:, :-1], # (B, L-2, 2, 2)
            )

            # violation loss btween calphas - L1
            dist_mat = torch.cdist(infer["wrapper_pred"][:, :, 1], 
                                   infer["wrapper_pred"][:, :, 1],) + \
                       torch.eye(infer["wrapper_pred"][0, :, 1].shape[0], 
                                 device=device).unsqueeze(0) * 5
            viol_loss = -torch.minimum(dist_mat - 3.78, torch.zeros_like(dist_mat)) 
            
            # calc metrics
            log_dict = {
                "torsion_loss": torsion_loss.mean().item(),
                "viol_loss": viol_loss.mean().item()           
            }
            metrics = mp_nerf.proteins.get_protein_metrics(
                true_coords=infer["coords"], pred_coords=infer["wrapper_pred"], detach=False
            )
            log_dict.update({k:v.mean().item() for k,v in metrics.items() if "wrap" not in k})

            # calc loss
            prev_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            if isinstance(loss_f, str):
                loss += eval(loss_f)
            else: 
                loss += torsion_loss.mean() + metrics["drmsd"].mean() # + 
                
            # record
            log_dict["loss"] = loss.item() - prev_loss
            metrics_list.append( log_dict )
            if wandbai and WANDB:
                wandb.log(metrics_list[-1])

            
        # clip gradients - p.44 AF2 methods section
        # update weights
        (loss/accumulate_every).mean().backward() # retain_graph=True
        if clip: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) 
        optim.step()
        optim.zero_grad()
        loss = 0.

        # log
        if (b-1) % log_every == 0:
            tac = time.time()
            print("Batch {0}/{1}, metrics_last_ex = {2}. Took: {3} seconds".format(
                (b-1), # // accumulate_every, 
                steps // accumulate_every, 
                {k: np.mean([x[k] for x in metrics_list[-log_every:]]) \
                    for k in log_dict.keys()}, 
                np.round(tac-tic, decimals=3)
            ))
            tic = tac 

        # go to next bacth
        b += 1

    metrics_stats = { "train_"+k : \
                      np.mean([ metrics[k] for metrics in metrics_list ]) \
                      for k in metrics_list[0].keys()
                    }

    return metrics_list, metrics_stats


#############################
## MAKING REAL PREDICTIONS ##
#############################

def infer_from_seqs(seq_list, model, embedder, batch_converter, 
                    recycle_func=lambda x: 10, device="cpu"): 
    """ Infers structures for a sequence of proteins. 
        Inputs: 
        * seq_list: list of str. Protein sequences in FASTA format
        * model: torch.nn.Module pytorch model
        * embedder: torch.nn.Module pytorch model. 
        * batch_converter: function to prepare input tokens for the embedder. 
        * recycle_func: func -> int. number of recycling iterations. a lower value 
            makes prediction faster. Past 10, improvement is marginal.
        * device: str or torch.device. Device for inference. CPU is slow. 
        Outputs: dict of
        * coords: list of torch.FloatTensor. Each of shape (L, 14, 3)
        * int_seq: list of torch.LongTensor. Each of shape (L, )
    """
    batch_dim = len(seq_list)
    lengths = [len(x) for x in seq_list]
    max_seq_len = max(lengths)
    # group in batch - init tokens to padding tok
    int_seq = torch.ones(batch_dim, max_seq_len, dtype=torch.long)*21
    for i, seq in enumerate(seq_list): 
        int_seq[i, :lengths[i]] = torch.tensor([ 
            mp_nerf.kb_proteins.AAS2INDEX[aa] for aa in seq 
        ])
    int_seq = int_seq.to(device)
    mask = int_seq != 21 # tokens to predict

    # get embeddings
    embedds = get_esm_embedd(int_seq, embedd_model=embedder, batch_converter=batch_converter)
    embedds = torch.cat([
        embedds, 
        torch.zeros_like(embedds[..., -4:])
    ], dim=-1)
    # don't pass angles info - just 0 at start (sin=0, cos=1)
    embedds[:, :, [1, 3]] = 1.
    # pred
    with torch.no_grad(): 
        preds, r_iters = model.forward(embedds, mask=None, recycle=recycle_func(None))  
    points_preds = rearrange(preds, '... (a d) -> ... a d', a=2)       # (B, L, 2, 2)
    
    # POST-PROCESS
    points_preds, ca_trace_pred, frames_preds, wrapper_pred = rgn2_replica.rgn2.pred_post_process(
        points_preds, seq_list=seq_list, mask=mask
    )

    return {
        # (L, 14, 3)
        "coords": [ wrapper_pred[i, :lengths[i]] for i in range(batch_dim) ], 
        # (L, )
        "int_seq": [ int_seq[i, :lengths[i]] for i in range(batch_dim) ],
        # (L, 2, 2)
        "points_preds": [ points_preds[i:i+1, :lengths[i]] for i in range(batch_dim) ],
        # (L, 3, 3)
        "frames_preds": [ frames_preds[i, :lengths[i]] for i in range(batch_dim)] ,
    }


