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
    step = math.ceil( len(seq) / x )
    seq_ = seq.clone()
    perms = []
    for i in range(x):
        chunk = seq_[i*step:(i+1)*step].numpy()
        np.random.shuffle(chunk)
        perms.append( torch.from_numpy(chunk, device=seq.device) )

    return torch.cat( perms , dim=0)