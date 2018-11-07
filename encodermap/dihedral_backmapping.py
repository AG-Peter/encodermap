from math import pi
import MDAnalysis as md
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction


def _expand_universe(universe, length):
    coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(),
                                       universe.atoms).run().results
    coordinates = np.tile(coordinates, (length, 1, 1))
    universe.load_new(coordinates, format=MemoryReader)


def _set_dihedral(dihedral, atoms, angle):
    current_angle = dihedral.dihedral.value()
    head = atoms[dihedral[2].id:]
    vec = dihedral[2].position - dihedral[1].position
    head.rotateby(angle-current_angle, vec, dihedral[2].position)


def dihedral_backmapping(pdb_path, dihedral_trajectory, rough_n_points=-1):
    """
    Takes a pdb file with a peptide and creates a trajectory based on the dihedral angles given.
    It simply rotates around the dihedral angle axis. In the result side-chains might overlap but the backbone should
    turn out quite well.

    :param pdb_path: (str)
    :param dihedral_trajectory:
        array-like of shape (traj_length, number_of_dihedrals)
    :param rough_n_points: (int) a step_size to select a subset of values from dihedral_trajectory is calculated by
        max(1, int(len(dihedral_trajectory) / rough_n_points)) with rough_n_points = -1 all values are used.
    :return: (MDAnalysis.Universe)
    """
    step_size = max(1, int(len(dihedral_trajectory) / rough_n_points))
    dihedral_trajectory = dihedral_trajectory[::step_size]

    uni = md.Universe(pdb_path)
    protein = uni.select_atoms("protein")

    dihedrals = []

    for residue in protein.residues:
        phi = residue.phi_selection()
        if phi:
            dihedrals.append(phi)

    for residue in protein.residues:
        psi = residue.psi_selection()
        if psi:
            dihedrals.append(psi)

    _expand_universe(uni, len(dihedral_trajectory))

    for dihedral_values, step in zip(dihedral_trajectory, uni.trajectory):
        for dihedral, value in zip(dihedrals, dihedral_values):
            _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
    return uni
