import torch
from torch import sin, cos, atan2, acos
from functools import wraps
from typing import Optional
from einops import rearrange
import torch.nn.functional as F

def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype = torch.get_default_dtype())
        return fn(t)
    return inner

@cast_torch_tensor
def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

@cast_torch_tensor
def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

"""
below are fape-related utilities, translated from AF2 original implementation
"""

def rots_mul_vecs(rots, vecs, cross_mode=False):
    if cross_mode:
        return torch.einsum("b n i j, b m j -> b n m i", rots, vecs)
    else:
        return torch.einsum("b n i j, b n j ->b n i", rots, vecs)

def invert_rigids(rigids):
    rots, trans = rigids
    inv_rots = rearrange(rots, "b n i j -> b n j i")
    t = rots_mul_vecs(inv_rots, trans)
    inv_trans = -t
    return inv_rots, inv_trans

def rigid_mul_vecs(rigids, vecs, cross_mode=False):
    rots, trans = rigids
    trans = trans.unsqueeze(1)
    return rots_mul_vecs(rots, vecs, cross_mode) + trans

def frame_aligned_point_error(
        pred_frames, pred_positions, target_frames, target_positions,
        frames_mask=None, positions_mask=None,
        length_scale=10.0, l1_clamp_distance: Optional[float] = None, epsilon=1e-4):

    # Compute array of predicted/target positions in the predicted frames.
    local_pred_pos = rigid_mul_vecs(invert_rigids(pred_frames), pred_positions, cross_mode=True)
    local_target_pos = rigid_mul_vecs(invert_rigids(target_frames), target_positions, cross_mode=True)

    # Compute errors between the structures.
    error_dist = torch.sqrt((local_pred_pos - local_target_pos).pow(2) + epsilon)

    if l1_clamp_distance:
        error_dist = torch.clamp(error_dist, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    # todo: implement masking
    if frames_mask is not None and positions_mask is not None:
        normed_error *= frames_mask.unsqueeze(-1)
        normed_error *= positions_mask.unsqueeze(-2)

        normalization_factor = torch.sum(frames_mask, dim=-1) * torch.sum(positions_mask, dim=-1)

        return torch.sum(normed_error, dim=(-2, -1)) / (epsilon + normalization_factor)
    else:
        return torch.mean(normed_error)  # mean of all elements
