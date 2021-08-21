# Author: Eric Alcaide ( @hypnopump ) 
import random
import math
import torch
import numpy as np


# random hacks - device utils for pyTorch - saves transfers
to_cpu = lambda x: x.cpu() if x.is_cuda else x
to_device = lambda x, device: x.to(device) if x.device != device else x

# system-wide utility functions

def expand_dims_to(t, length):
    """ Expands up to N dimensions. Different from AF2 (inspo drawn):
    	* Only works for torch Tensors
    	* Expands to `t`, NOT `adds t dims`
        https://github.com/lucidrains/alphafold2/blob/main/alphafold2_pytorch/utils.py#L63
        Ex: 
        >>> expand_dims_to( torch.eye(8), length = 3) # (1, 8, 8)
        >>> expand_dims_to( torch.eye(8), length = 1) # (8, 8)
    """
    if not length - len(t.shape) > 0:
        return t
    return t.reshape(*((1,) * length - len(t.shape)), *t.shape) 


def set_seed(seed, verbose=False): 
    try: random.seed(seed)
    except: "Could not set `random` module seed"

    try: np.random.seed(seed)
    except: "Could not set `np.random` module seed"

    try: torch.manual_seed(seed)
    except:"Could not set `torch.manual_seed` module seed"
    
    if verbose: 
        print("Seet seed to {0}".format(seed))






