from typing import Optional
import pyrosetta

"""
NOTE: Remember to initialize PyRosetta before using these functions

Example:
    import pyrosetta
    pyrosetta.init("-mute all")

If you need to see Rosetta outputs, remove '-mute all'
"""


def get_fa_min_mover(
        max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """ Create full-atom minimization mover
        Inputs: 
        * max_iter: int. Maximum number of iterations for MinMover
    """
    # Create full-atom score function with terms for fixing bad bond lengths
    sf = pyrosetta.create_score_function('ref2015_cst')
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0)

    # Allow movement of backbone, side chains, and chain breaks
    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    # Create MinMover acting in cartesian space
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(max_iter)
    min_mover.cartesian(True)

    return min_mover


def get_fa_relax_mover(
        max_iter: int = 200) -> pyrosetta.rosetta.protocols.moves.Mover:
    """ Create full-atom relax mover
        Inputs: 
        * max_iter: int. Maximum number of iterations for FastRelax
    """
    # Create full-atom score function
    sf = pyrosetta.create_score_function('ref2015_cst')

    # Allow movement of backbone, side chains, and chain breaks
    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    # Create FastRelax mover acting in dualspace (cartesian and internal space)
    relax_mover = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax_mover.set_scorefxn(sf)
    relax_mover.max_iter(max_iter)
    relax_mover.dualspace(True)
    relax_mover.set_movemap(mmap)
    relax_mover.ramp_down_constraints(True)

    return relax_mover


def quick_refine(in_pdb: str, out_pdb: Optional[str] = None, min_iter: int = 1000):
    """ PyRosetta protocol for minimization refinement of protein structure
        Inputs: 
        * in_pdb: str. Path to PDB file to be refined
        * out_pdb: str. Path to save refined PDB file
        * min_iter: int. Maximum number of iterations for MinMover
    """
    if out_pdb is None:
        out_pdb = in_pdb

    # Load input PDB into pose
    pose = pyrosetta.pose_from_pdb(in_pdb)

    # Create movers
    cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
    cst_mover.cst_sidechain(False)
    min_mover = get_fa_min_mover(min_iter)
    idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()

    # Refine structure
    cst_mover.apply(pose)
    min_mover.apply(pose)
    idealize_mover.apply(pose)

    # Save refined structure to PDB
    pose.dump_pdb(out_pdb)

def relax_refine(in_pdb: str, out_pdb: Optional[str] = None, min_iter: int = 1000, relax_iter: int = 200):
    """ PyRosetta protocol for relaxation and minimization of protein structure
        Inputs: 
        * in_pdb: str. Path to PDB file to be refined
        * out_pdb: str. Path to save refined PDB file
        * min_iter: int. Maximum number of iterations for MinMover
        * relax_iter: int. Maximum number of iterations for FastRelax
    """
    if out_pdb is None:
        out_pdb = in_pdb

    # Load input PDB into pose
    pose = pyrosetta.pose_from_pdb(in_pdb)

    # Create movers
    cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
    cst_mover.cst_sidechain(False)
    min_mover = get_fa_min_mover(min_iter)
    relax_mover = get_fa_relax_mover(relax_iter)
    idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()

    # Refine structure
    cst_mover.apply(pose)
    min_mover.apply(pose)
    relax_mover.apply(pose)
    min_mover.apply(pose)
    idealize_mover.apply(pose)

    # Save refined structure to PDB
    pose.dump_pdb(out_pdb)