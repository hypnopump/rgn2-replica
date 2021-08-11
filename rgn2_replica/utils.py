# Author: Eric Alcaide ( @hypnopump ) 
import random
import math
import torch
import numpy as np


def chunk_permute(seq):
    """ Permutes a chunk from the sequence. 
        Inputs, Outputs: 
        * seq: (N,) tensor. 
    """
    x = random.randint(2, 10) 
    step = math.ceil( seq.shape[0] / x )
    seq_ = seq.clone()
    perms = []
    for i in range(x):
        chunk = seq_[i*step:(i+1)*step].numpy()
        np.random.shuffle(chunk)
        perms.append( torch.from_numpy(chunk) )

    return torch.cat( perms , dim=0).to(seq.device)


def masked_lang(seq, mask_tok=99, prop_len=0.15, lam=2.5): 
    """ Masks contiguous positions for masked language modelling
        Inputs: 
        * seq: (N,) tensor
        * mask_tok: int. mask token.
        * prop_len: float. proportion of the length to mask
        * lam: float. lambda for the poisson distribution.
        Outputs: (N,)
    """
    seq_ = seq.clone()
    # set mask features
    clump_size = int( np.random.poisson(lam=lam) + 1 )
    n_mask = int( seq.shape[0] * prop_len )
    # get maskable idxs
    idxs = list(range(seq.shape[0] - clump_size))
    # do contiguous mask iteratively
    for i in range( int(n_mask/clump_size) ):
        choice = np.random.choice(idxs)
        seq_[choice:choice+clump_size] = mask_tok
        # eliminate contiguous positions
        for j in range(clump_size): 
            idxs.remove(choice+j)

    return seq_


def mask_seq(seq, mask_tok=99, prop_len=0.15, lam=2.5): 
    """ Masks a sequence as described in paper - page 16
        https://www.biorxiv.org/content/10.1101/2021.08.02.454840v1.full.pdf
        Inputs, Outputs: 
        * seq: (N,) tensor
    """
    p = random.random()
    # chunk permutation
    if p < 0.3: 
        if p < 0.3*0.35: seq = chunk_permute(seq) # modify
        else: pass                                # unmodified
    # masked language modelling
    else: 
        # normal mask or clumping depending on prob - regulate lambda
        lam_eff = 0 if p < ( 0.3 + 0.7 * (1 - 0.3) ) else lam
        seq = masked_lang(seq, mask_tok=mask_tok, 
        				  prop_len=prop_len, lam=lam_eff)
    return seq












