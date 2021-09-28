# Author: Eric Alcaide

import torch
import numpy as np 


# random hacks

# to_pi_minus_pi(4) = -2.28  # to_pi_minus_pi(-4) = 2.28  # rads to pi-(-pi)
to_zero_two_pi = lambda x: ( x + (2*np.pi) * ( 1 + torch.floor_divide(x.abs(), 2*np.pi) ) ) % (2*np.pi)
def to_pi_minus_pi(x): 
    zero_two_pi = to_zero_two_pi(x)
    return torch.where(
        zero_two_pi < np.pi, zero_two_pi, -(2*np.pi - zero_two_pi)
    )

@torch.jit.script 
def cdist(x,y):
    """ robust cdist - drop-in for pytorch's. 
        Inputs: 
        * x, y: (B, N, D)
    """
    return torch.pow( 
        x.unsqueeze(-3) - y.unsqueeze(-2), 2 
    ).sum(dim=-1).clamp(min=1e-7).sqrt()

# data utils
def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, 
             verbose=True, subset="train", xray_filter=False, full_mask=True):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training. 
        Inputs: 
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
        * subset: str. which subset to load proteins from. 
        * xray_filter: bool. whether to return only xray structures.
        * mask_tol: bool or int. bool: whether to return seqs with unknown coords.
                    int: number of minimum label positions
        Outputs: (cleaned, without padding)
        (seq_str, int_seq, coords, angles, padding_seq, mask, pid)
    """
    if xray_filter: 
        raise NotImplementedError

    while True:
        for b,batch in enumerate(dataloader_[subset]):
            for i in range(batch.int_seqs.shape[0]):
                # skip too short
                if batch.int_seqs[i].shape[0] < min_len:
                    continue

                # strip padding - matching angles to string means
                # only accepting prots with no missing residues (mask is 0)
                padding_seq = (batch.int_seqs[i] == 20).sum().item()
                padding_mask = -(batch.msks[i] - 1).sum().item() # find 0s

                if (full_mask and padding_seq == padding_mask) or \
                    (full_mask is not True and batch.int_seqs[i].shape[0] - full_mask > 0):
                    # check for appropiate length
                    real_len = batch.int_seqs[i].shape[0] - padding_seq
                    if max_len >= real_len >= min_len:
                        # strip padding tokens
                        seq = batch.str_seqs[i] # seq is already unpadded - see README at scn repo 
                        int_seq = batch.int_seqs[i][:-padding_seq or None]
                        angles  = batch.angs[i][:-padding_seq or None]
                        mask    = batch.msks[i][:-padding_seq or None]
                        coords  = batch.crds[i][:-padding_seq*14 or None]

                        if verbose:
                            print("stopping at sequence of length", real_len)
                        
                        yield seq, int_seq, coords, angles, padding_seq, mask, batch.pids[i]
                    else:
                        if verbose:
                            print("found a seq of length:", batch.int_seqs[i].shape,
                                  "but oustide the threshold:", min_len, max_len)
                else:
                    if verbose:
                        print("paddings not matching", padding_seq, padding_mask)
                    pass
    return None
    

######################
## structural utils ##
######################

def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


def get_cosine_angle(c1, c2, c3, eps=1e-7): 
    """ Returns the angle in radians. Uses cosine formula
        Not all angles are possible all the time.
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    return torch.acos( (u1*u2).sum(dim=-1)  / (u1.norm(dim=-1)*u2.norm(dim=-1) + eps))


def get_angle(c1, c2, c3):
    """ Returns the angle in radians.
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # dont use acos since norms involved. 
    # better use atan2 formula: atan2(cross, dot) from here: 
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # add a minus since we want the angle in reversed order - sidechainnet issues
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1), 
                        -(u1*u2).sum(dim=-1) ) 


def kabsch_torch(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (D, N) - usually (3, N)
    """
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t())
    # Optimal rotation matrix via SVD - warning! W must be transposed
    if int(torch.__version__.split(".")[1]) < 8:
        V, S, W = torch.svd(C.detach())
        W = W.t()
    else: 
        V, S, W = torch.linalg.svd(C.detach()) 
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_


def rmsd_torch(X, Y):
    """ Assumes x,y are both (batch, d, n) - usually (batch, 3, N). """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )


def drmsd_torch(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    X_ = X.transpose(-1, -2)
    Y_ = Y.transpose(-1, -2)
    x_dist = cdist(X_, X_) # (B, N, N)
    y_dist = cdist(Y_, Y_) # (B, N, N)

    return torch.sqrt( torch.pow(x_dist-y_dist, 2).mean(dim=(-1, -2)).clamp(min=1e-7) )


def ensure_chirality(coords_wrapper, use_backbone=True): 
    """ Ensures protein agrees with natural distribution 
        of chiral bonds (ramachandran plots).
        Reflects ( (-1)*Z ) the ones that do not. 
        Inputs: 
        * coords_wrapper: (B, L, C, 3) float tensor. First 3 atoms
                          in C should be N-CA-C
        * use_backbone: bool. whether to use the backbone (better, more robust) 
                              if provided, or just use c-alphas. 
        Ouputs: (B, L, C, 3)
    """
    
    # detach gradients for angle calculation - mirror selection
    coords_wrapper_ = coords_wrapper.detach()
    mask = coords_wrapper_.abs().sum(dim=(-1, -2)) != 0.

    # if BB present: use bb dihedrals
    if coords_wrapper[:, :, 0].abs().sum() != 0. and use_backbone:
        # compute phis for every protein in the batch
        phis = get_dihedral(
            coords_wrapper_[:, :-1, 2], # C_{i-1}
            coords_wrapper_[:, 1: , 0], # N_{i}
            coords_wrapper_[:, 1: , 1], # CA_{i}
            coords_wrapper_[:, 1: , 2], # C_{i}
        )

        # get proportion of negatives
        props = [(phis[i, mask[i, :-1]] > 0).float().mean() for i in range(mask.shape[0])]

        # fix mirrors by (-1)*Z if more (+) than (-) phi angles
        corrector = torch.tensor([ [1, 1, -1 if p > 0.5 else 1]  # (B, 3)
                                   for p in props ], dtype=coords_wrapper.dtype)

        return coords_wrapper * corrector.to(coords_wrapper.device)[:, None, None, :]
    else: 
        return coords_wrapper




