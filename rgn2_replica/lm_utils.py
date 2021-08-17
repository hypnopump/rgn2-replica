# Author: Eric Alcaide ( @hypnopump ) 
import random
import math
import torch
import numpy as np

# utils specific for the LM

def chunk_permute(seq):
    """ Permutes a chunk from the sequence. 
        Inputs: 
        * seq: (N,) tensor. 
        Outputs: 
        * seq: (N,) tensor
        * labels: (N,) long tensor (idx of each chunk)
    """
    x = random.randint(2, 10) 
    step = math.ceil( seq.shape[0] / x )
    seq_ = seq.clone()

    perms, labels = [], []
    for i in range(x):
        chunk = seq_[i*step:(i+1)*step].numpy()
        np.random.shuffle(chunk)
        perms.append( torch.from_numpy(chunk) )
        labels.append( torch.ones_like(perms[-1]) * i )

    perms = torch.cat( perms , dim=0).to(seq.device)
    labels = torch.cat( labels , dim=0).to(seq.device)
    return perms, labels


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
        Inputs:  
        * seq: (N,) tensor
        Outputs: 
        * seq: (N,) tensor 
        * chunk_permte: bool (indicates seq has been chunk-permutted)
        * labels: (N,) tensor. Chunk belonging of each AA
    """
    p = random.random()
    labels = None
    # chunk permutation
    if p < 0.3: 
        # modify (prob=0.35) or unchanged
        if p < 0.3*0.35: seq, labels = chunk_permute(seq) 
        else: pass                              
    # masked language modelling
    else: 
        # normal mask or clumping depending on prob - regulate lambda
        lam_eff = 0 if p < ( 0.3 + 0.7 * (1 - 0.3) ) else lam
        seq = masked_lang(seq, mask_tok=mask_tok, 
                          prop_len=prop_len, lam=lam_eff)
    # create chunk labels
    if labels is None: 
        labels = torch.zeros_like(seq) 

    return seq, p < 0.3*0.35, labels
