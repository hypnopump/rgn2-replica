# Author: Eric Alcaide ( @hypnopump ) 
import random
import math
import torch
import numpy as np

# system-wide utility functions

def set_seed(seed, verbose=False): 
    try: random.seed(seed)
    except: "Could not set `random` module seed"

    try: np.random.seed(seed)
    except: "Could not set `np.random` module seed"

    try: torch.manual_seed(seed)
    except:"Could not set `torch.manual_seed` module seed"
    
    if verbose: 
        print("Seet seed to {0}".format(seed))






