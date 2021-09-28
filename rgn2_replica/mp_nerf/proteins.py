# science
# diff / ml
# module
import torch.nn.functional as F
from rgn2_replica.mp_nerf.utils import *
from rgn2_replica.mp_nerf.ml_utils import *
from rgn2_replica.mp_nerf.massive_pnerf import *
from rgn2_replica.mp_nerf.kb_proteins import *
from einops import rearrange, repeat


def scn_cloud_mask(seq, coords=None, strict=False):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * seqs: (length) iterable of 1-letter aa codes of a protein
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        * strict: bool. whther to discard the next points after a missing one 
        Outputs: (length, 14) boolean mask 
    """ 
    if coords is not None:
        start = (( rearrange(coords, 'b (l c) d -> b l c d', c=14) != 0 ).sum(dim=-1) != 0).float()
        # if a point is 0, the following are 0s as well
        if strict:
            for b in range(start.shape[0]):
                for pos in range(start.shape[1]):
                    for chain in range(start.shape[2]):
                        if start[b, pos, chain].item() == 0:
                            start[b, pos, chain:] *= 0
        return start
    return torch.tensor([SUPREME_INFO[aa]['cloud_mask'] for aa in seq])


def scn_bond_mask(seq):
    """ Inputs: 
        * seqs: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 14) maps point to bond length
    """ 
    return torch.tensor([SUPREME_INFO[aa]['bond_mask'] for aa in seq])


def scn_angle_mask(seq, angles=None, device=None):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        * angles: (length, 12). [phi, psi, omega, b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca), 6_scn_torsions]
        Outputs: (L, 14) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    device = angles.device if angles is not None else torch.device("cpu")
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    torsion_mask_use = "torsion_mask" if angles is not None else "torsion_mask_filled"
    # get masks
    theta_mask   = torch.tensor([SUPREME_INFO[aa]['theta_mask'] for aa in seq], dtype=precise).to(device)
    torsion_mask = torch.tensor([SUPREME_INFO[aa][torsion_mask_use] for aa in seq], dtype=precise).to(device)
    
    # adapt general to specific angles if passed
    if angles is not None: 
        # fill masks with angle values
        theta_mask[:, 0] = angles[:, 4] # ca_c_n
        theta_mask[1:, 1] = angles[:-1, 5] # c_n_ca
        theta_mask[:, 2] = angles[:, 3] # n_ca_c
        # backbone_torsions
        torsion_mask[:, 0] = angles[:, 1] # n determined by psi of previous
        torsion_mask[1:, 1] = angles[:-1, 2] # ca determined by omega of previous
        torsion_mask[:, 2] = angles[:, 0] # c determined by phi
        # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L313
        torsion_mask[:, 3] = angles[:, 1] - np.pi

        # add torsions to sidechains - no need to modify indexes due to torsion modification
        # since extra rigid modies are in terminal positions in sidechain
        to_fill = torsion_mask != torsion_mask # "p" fill with passed values
        to_pick = torsion_mask == 999          # "i" infer from previous one
        for i,aa in enumerate(seq):
            # check if any is nan -> fill the holes
            number = to_fill[i].long().sum()
            torsion_mask[i, to_fill[i]] = angles[i, 6:6+number]

            # pick previous value for inferred torsions
            for j, val in enumerate(to_pick[i]):
                if val:
                    torsion_mask[i, j] = torsion_mask[i, j-1] - np.pi # pick values from last one.

            # special rigid bodies anomalies: 
            if aa == "I": # scn_torsion(CG1) - scn_torsion(CG2) = 2.13 (see KB)
                torsion_mask[i, 7] += torsion_mask[i, 5]
            elif aa == "L": 
                torsion_mask[i, 7] += torsion_mask[i, 6]


    torsion_mask[-1, 3] += np.pi 
    return torch.stack([theta_mask, torsion_mask], dim=0)


def scn_index_mask(seq):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 11, 3) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    idxs = torch.tensor([SUPREME_INFO[aa]['idx_mask'] for aa in seq])
    return rearrange(idxs, 'l s d -> d l s')


def scn_rigid_index_mask(seq, c_alpha=None): 
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein 
        * c_alpha: part of the chain to compute frames on.
        Outputs: (3, Length * Groups). indexes for 1st, 2nd and 3rd point 
                  to construct frames for each group. 
    """
    maxi = 1 if c_alpha else None

    return torch.cat([torch.tensor(SUPREME_INFO[aa]['rigid_idx_mask'])[:maxi] if i==0 else \
                      torch.tensor(SUPREME_INFO[aa]['rigid_idx_mask'])[:maxi] + 14*i \
                      for i,aa in enumerate(seq)], dim=0).t()


def build_scaffolds_from_scn_angles(seq, angles=None, coords=None, device="auto"):
    """ Builds scaffolds for fast access to data
        Inputs: 
        * seq: string of aas (1 letter code)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        * coords: (L, 3) sidechainnet coords. builds the mask with those instead
                  (better accuracy if modified residues present).
        Outputs:
        * cloud_mask: (L, 14 ) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    # auto infer device and precision
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    if device == "auto":
        device = angles.device if angles is not None else device

    if coords is not None: 
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else: 
        cloud_mask = scn_cloud_mask(seq)

    cloud_mask = cloud_mask.bool().to(device)
    
    point_ref_mask = scn_index_mask(seq).long().to(device)
     
    angles_mask = scn_angle_mask(seq, angles).to(device, precise)
     
    bond_mask = scn_bond_mask(seq).to(device, precise)
    # return all in a dict
    return {"cloud_mask":     cloud_mask, 
            "point_ref_mask": point_ref_mask,
            "angles_mask":    angles_mask,
            "bond_mask":      bond_mask }


#############################
####### ENCODERS ############
#############################


def modify_angles_mask_with_torsions(seq, angles_mask, torsions): 
    """ Modifies a torsion mask to include variable torsions. 
        Inputs: 
        * seq: (L,) str. FASTA sequence
        * angles_mask: (2, L, 14) float tensor of (angles, torsions)
        * torsions: (L, 4) float tensor (or (L, 5) if it includes torsion for cb)
        Outputs: (2, L, 14) a new angles mask
    """
    c_beta = torsions.shape[-1] == 5 # whether c_beta torsion is passed as well
    start = 4 if c_beta else 5
    # get mask of to-fill values
    torsion_mask = torch.tensor([SUPREME_INFO[aa]["torsion_mask"] for aa in seq]).to(torsions.device) # (L, 14)
    torsion_mask = torsion_mask != torsion_mask # values that are nan need replace
    # undesired outside of margins
    torsion_mask[:, :start] = torsion_mask[:, start+torsions.shape[-1]:] = False

    angles_mask[1, torsion_mask] = torsions[ torsion_mask[:, start:start+torsions.shape[-1]] ]
    return angles_mask


def modify_scaffolds_with_coords(scaffolds, coords):
    """ Gets scaffolds and fills in the right data.
        Inputs: 
        * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
        * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
        Outputs: corrected scaffolds
    """


    # calculate distances and update: 
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1) # N
    scaffolds["bond_mask"][ :, 1] = torch.norm(coords[ :, 1] - coords[:  , 0], dim=-1) # CA
    scaffolds["bond_mask"][ :, 2] = torch.norm(coords[ :, 2] - coords[:  , 1], dim=-1) # C
    # O, CB, side chain
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # get indexes
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][:, :, i-3] # (3, L, 11) -> 3 * (L, 11)
        # correct distances
        scaffolds["bond_mask"][:, i] = torch.norm(coords[:, i] - coords[selector, idx_c], dim=-1)
        # get angles
        scaffolds["angles_mask"][0, :, i] = get_angle(coords[selector, idx_b], 
                                                      coords[selector, idx_c], 
                                                      coords[:, i])
        # handle C-beta, where the C requested is from the previous aa
        if i == 4:
            # for 1st residue, use position of the second residue's N
            first_next_n     = coords[1, :1] # 1, 3
            # the c requested is from the previous residue
            main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]# (L-1), 3
            # concat
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(coords_a,
                                                         coords[selector, idx_b], 
                                                         coords[selector, idx_c], 
                                                         coords[:, i])
    # correct angles and dihedrals for backbone 
    scaffolds["angles_mask"][0, :-1, 0] = get_angle(coords[:-1, 1], coords[:-1, 2], coords[1: , 0]) # ca_c_n
    scaffolds["angles_mask"][0, 1:,  1] = get_angle(coords[:-1, 2], coords[1:,  0], coords[1: , 1]) # c_n_ca
    scaffolds["angles_mask"][0,  :,  2] = get_angle(coords[:,   0], coords[ :,  1], coords[ : , 2]) # n_ca_c
    
    # N determined by previous psi = f(n, ca, c, n+1)
    scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0])
    # CA determined by omega = f(ca, c, n+1, ca+1)
    scaffolds["angles_mask"][1,  1:, 1] = get_dihedral(coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1])
    # C determined by phi = f(c-1, n, ca, c)
    scaffolds["angles_mask"][1,  1:, 2] = get_dihedral(coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2])

    return scaffolds


##################################
####### MAIN FUNCTION ############
##################################


def protein_fold(cloud_mask, point_ref_mask, angles_mask, bond_mask,
                 device=torch.device("cpu"), hybrid=False):
    """ Calcs coords of a protein given it's
        sequence and internal angles.
        Inputs: 
        * cloud_mask: (L, 14) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, 14, L) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom

        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    # automatic type (float, mixed, double) and size detection
    precise = bond_mask.dtype
    length  = cloud_mask.shape[0]
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device, dtype=precise)

    # do first AA
    coords[0, 1] = coords[0, 0] + torch.tensor([1, 0, 0], device=device, dtype=precise) * BB_BUILD_INFO["BONDLENS"]["n-ca"] 
    coords[0, 2] = coords[0, 1] + torch.tensor([torch.cos(np.pi - angles_mask[0, 0, 2]),
                                                torch.sin(np.pi - angles_mask[0, 0, 2]),
                                                0.], device=device, dtype=precise) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    
    # starting positions (in the x,y plane) and normal vector [0,0,1]
    init_a = repeat(torch.tensor([1., 0., 0.], device=device, dtype=precise), 'd -> l d', l=length)
    init_b = repeat(torch.tensor([1., 1., 0.], device=device, dtype=precise), 'd -> l d', l=length)
    # do N -> CA. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 1]
    coords[1:, 1] = mp_nerf_torch(init_a,
                                   init_b, 
                                   coords[:, 0], 
                                   bond_mask[:, 1], 
                                   thetas, dihedrals)[1:]
    # do CA -> C. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 2]
    coords[1:, 2] = mp_nerf_torch(init_b,
                                   coords[:, 0],
                                   coords[:, 1],
                                   bond_mask[:, 2],
                                   thetas, dihedrals)[1:]
    # do C -> N
    thetas, dihedrals = angles_mask[:, :, 0]
    coords[:, 3] = mp_nerf_torch(coords[:, 0],
                                   coords[:, 1],
                                   coords[:, 2],
                                   bond_mask[:, 0],
                                   thetas, dihedrals)

    #########
    # sequential pass to join fragments
    #########
    # part of rotation mat corresponding to origin - 3 orthogonals
    mat_origin  = get_axis_matrix(init_a[0], init_b[0], coords[0, 0], norm=False)
    # part of rotation mat corresponding to destins || a, b, c = CA, C, N+1
    # (L-1) since the first is in the origin already 
    mat_destins = get_axis_matrix(coords[:-1, 1], coords[:-1, 2], coords[:-1, 3])

    # get rotation matrices from origins
    # https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    rotations  = torch.matmul(mat_origin.t(), mat_destins)
    rotations /= torch.norm(rotations, dim=-1, keepdim=True)

    # do rotation concatenation - do for loop in cpu always - faster
    rotations = rotations.cpu() if coords.is_cuda and hybrid else rotations
    for i in range(1, length-1):
        rotations[i] = torch.matmul(rotations[i], rotations[i-1])
    rotations = rotations.to(device) if coords.is_cuda and hybrid else rotations
    # rotate all
    coords[1:, :4] = torch.matmul(coords[1:, :4], rotations)
    # offset each position by cumulative sum at that position
    coords[1:, :4] += torch.cumsum(coords[:-1, 3], dim=0).unsqueeze(-2)


    #########
    # parallel sidechain - do the oxygen, c-beta and side chain
    #########
    for i in range(3,14):
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = coords[(level_mask.nonzero().view(-1) - 1), idx_a] # (L-1), 3
            # if first residue is not glycine, 
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0].item():
                coords_a[0] = coords[1, 1]
        else:
            coords_a = coords[level_mask, idx_a]

        coords[level_mask, i] = mp_nerf_torch(coords_a, 
                                              coords[level_mask, idx_b],
                                              coords[level_mask, idx_c],
                                              bond_mask[level_mask, i], 
                                              thetas, dihedrals)
    
    return coords, cloud_mask


def sidechain_fold(wrapper, cloud_mask, point_ref_mask, angles_mask, bond_mask,
                   device=torch.device("cpu"), c_beta=False):
    """ Calcs coords of a protein given it's sequence and internal angles.
        Inputs: 
        * wrapper: (L, 14, 3). coords container with backbone ([:, :3]) and optionally
                               c_beta ([:, 4])
        * cloud_mask: (L, 14) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, 14, L) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
        * c_beta: whether to place cbeta

        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    precise = wrapper.dtype

    # parallel sidechain - do the oxygen, c-beta and side chain
    for i in range(3,14):
        # skip cbeta if arg is set
        if i == 4 and not isinstance(c_beta, str):
            continue
        # prepare inputs
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = wrapper[(level_mask.nonzero().view(-1) - 1), idx_a] # (L-1), 3
            # if first residue is not glycine, 
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0].item():
                coords_a[0] = wrapper[1, 1]
        else:
            coords_a = wrapper[level_mask, idx_a]

        wrapper[level_mask, i] = mp_nerf_torch(coords_a, 
                                               wrapper[level_mask, idx_b],
                                               wrapper[level_mask, idx_c],
                                               bond_mask[level_mask, i], 
                                               thetas, dihedrals)
    
    return wrapper, cloud_mask


##############################
####### XTENSION  ############
##############################


# inspired by: https://www.biorxiv.org/content/10.1101/2021.08.02.454840v1
def ca_from_angles(angles, bond_len=3.80): 
    """ Builds a C-alpha trace from a set of 2 angles (theta, chi). 
        Inputs: 
        * angles: (B, L, 4): float tensor. (cos, sin) · (theta, chi)
                  angles in point-in-unit-circumference format. 
        Outputs: (B, L, 3) coords for c-alpha trace
    """
    device = angles.device
    length = angles.shape[-2]
    frames = [ torch.repeat_interleave(
                torch.eye(3, device=device, dtype=torch.float).unsqueeze(0),
                angles.shape[0], 
                dim=0
             )]

    rot_mats = torch.stack([ 
        torch.stack([  angles[...,0] * angles[...,2], angles[...,0] * angles[...,3], -angles[...,1]    ], dim=-1),
        torch.stack([ -angles[...,3]                , angles[...,2]                ,  angles[...,0]*0. ], dim=-1),
        torch.stack([  angles[...,1] * angles[...,2], angles[...,1] * angles[...,3],  angles[...,0]    ], dim=-1),
    ], dim=-2)  # (B, L, 3, 3)

    # iterative update of frames - skip last frame. 
    for i in range(length-1): 
        frames.append( rot_mats[:, i] @ frames[i] ) # could do frames[-1] as well
    frames = torch.stack(frames, dim=1) # (B, L, 3, 3)
        
    ca_trace = bond_len * frames[..., -1, :].cumsum(dim=-2) # (B, L, 3)

    return ca_trace, frames


# inspired by: https://github.com/psipred/DMPfold2/blob/master/dmpfold/network.py#L139
def ca_bb_fold(ca_trace): 
    """ Calcs a backbone given the coordinate trace of the CAs. 
        Inputs: 
        * ca_trace: (B, L, 3) float tensor with CA coordinates. 
        Outputs: (B, L, 14, 3) (-N-CA(-CB-...)-C(=O)-)
    """
    wrapper = torch.zeros(ca_trace.shape[0], ca_trace.shape[1]+2, 14, 3, device=ca_trace.device)
    wrapper[:, 1:-1, 1] = ca_trace
    # Place dummy extra Cα atoms on extremenes to get the required vectors
    vecs = ca_trace[ :, [0, 2, -1, -3] ] - ca_trace[ :, [1, 1, -2, -2] ] # (B, 4, 3)
    wrapper[:,  0, 1] = ca_trace[:,  0] + 3.80 * F.normalize(torch.cross(vecs[:, 0], vecs[:, 1]), dim=-1)
    wrapper[:, -1, 1] = ca_trace[:, -1] + 3.80 * F.normalize(torch.cross(vecs[:, 2], vecs[:, 3]), dim=-1)
    
    # place N and C term
    vec_ca_can = wrapper[:, :-2, 1] - wrapper[:, 1:-1, 1]
    vec_ca_cac = wrapper[:, 2: , 1] - wrapper[:, 1:-1, 1]
    mid_ca_can = (wrapper[:, 1:, 1] + wrapper[:, :-1, 1]) / 2
    cross_vcan_vcac = F.normalize(torch.cross(vec_ca_can, vec_ca_cac, dim=-1), dim=-1)
    wrapper[:, 1:-1, 0] = mid_ca_can[:, :-1] - vec_ca_can / 7.5 + cross_vcan_vcac / 3.33
    # placve all C but last, which is special
    wrapper[:, 1:-2, 2] = (mid_ca_can[:, :-1] + vec_ca_can / 8 - cross_vcan_vcac / 2.5)[:, 1:]
    wrapper[:,   -2, 2] = mid_ca_can[:, -1, :] - vec_ca_cac[:, -1, :] / 8 + cross_vcan_vcac[:, -1, :] / 2.5

    return wrapper[:, 1:-1]



############################
####### METRICS ############
############################


def get_protein_metrics(
        true_coords,
        pred_coords, 
        cloud_mask = None,
        return_aligned = True, 
        detach = None 
    ): 
    """ Calculates many metrics for protein structure quality.
        Aligns coordinates.
        Inputs: 
        * true_coords: (B, L, 14, 3) unaligned coords (B = 1)
        * pred_coords: (B, L, 14, 3) unaligned coords (B = 1)
        * cloud_mask: (B, L, 14) bool. gotten from pred_coords if not passed
        * return_aligned: bool. whether to return aligned structs.
        * detach: bool. whether to detach inputs before compute. saves mem
        Outputs: dict (k,v)
    """
    metric_dict = {
        # "rmsd": RMSD,
        # "drmsd": dRMSD,
        # not implemented yet
        # "gdt_ts": partial(GDT, mode="TS"),
        # "gdt_ha": partial(GDT, mode="HA"),
        # "tmscore": tmscore_torch,
        # "lddt": lddt_torch,
    }

    if detach:
        true_coords = true_coords.detach()
        pred_coords = pred_coords.detach()

    # clone so originals are not modified
    true_coords = true_coords.clone()
    pred_coords = pred_coords.clone()
    cloud_mask = pred_coords.abs().sum(dim=-1).bool() * \
                 true_coords.abs().sum(dim=-1).bool() # 1, L, 14
    chain_mask = cloud_mask.sum(dim=-1).bool() # 1, L

    true_aligned, pred_aligned = kabsch_torch(
        pred_coords[cloud_mask].t(), true_coords[cloud_mask].t()
    )
    # no need to rebuild true coords since unaffected by kabsch
    true_coords[cloud_mask] = true_aligned.t()
    pred_coords[cloud_mask] = pred_aligned.t()

    # compute metrics
    outputs = {}
    for k,f in metric_dict.items(): 
        # special. works only on ca trace
        if k == "tmscore": 
            ca_trace = true_coords[:, :, 1].transpose(-1, -2)
            ca_pred_trace = pred_coords[:, :, 1].transpose(-1, -2)
            outputs[k] = f(ca_trace, ca_pred_trace)
        # special. works on full prot
        elif k == "lddt": 
            outputs[k] = f(true_coords[:, chain_mask[0]], pred_coords[:, chain_mask[0]], cloud_mask=cloud_mask)
        # special. needs batch dim
        elif "gdt" in k: 
            outputs[k] = f(true_aligned[None, ...], pred_aligned[None, ...])
        else: 
            outputs[k] = f(true_aligned, pred_aligned)

    if return_aligned:
        outputs.update({
            "pred_align_wrap": pred_coords, 
            "true_align_wrap": true_coords,
        })

    return outputs

