from math import pi, cos, sin
import MDAnalysis as md
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from .misc import rotation_matrix
import tensorflow as tf


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


def straight_tetrahedral_chain(n_atoms=None, bond_lengths=None):
    dx = cos(70.63 / 180 * pi)
    dy = sin(70.63 / 180 * pi)

    if n_atoms and not bond_lengths:
        coordinates = np.zeros((n_atoms, 3), dtype=np.float32)
        indices = np.repeat(np.arange(int(n_atoms / 2) + 1), 2)
        coordinates[:, 0] = (indices[1:n_atoms + 1] + dx * indices[0:n_atoms])
        coordinates[:, 1] = dy * indices[0:n_atoms]

    elif (bond_lengths and not n_atoms) or n_atoms == len(bond_lengths)+1:
        n_bonds = len(bond_lengths)
        n_atoms = n_atoms or n_bonds+1

        dxs = bond_lengths * np.tile([1, dx], int(n_atoms/2))[:n_bonds]
        dys = bond_lengths * np.tile([0, dy], int(n_atoms/2))[:n_bonds]

        coordinates = np.zeros((n_atoms, 3), dtype=np.float32)
        coordinates[1:, 0] = np.cumsum(dxs)
        coordinates[1:, 1] = np.cumsum(dys)

    else:
        raise ValueError("input not compatible")
    return coordinates


def chain_in_plane(lengths, angles):

    prev_angle = tf.zeros(angles.shape[0])
    xs = [tf.zeros((lengths.shape[0]))]
    ys = [tf.zeros((lengths.shape[0]))]
    sign = 1

    for i in range(angles.shape[1]):
        prev_angle = tf.Print(prev_angle, [prev_angle])
        xs.append(xs[-1] + lengths[:, i] * tf.cos(prev_angle))
        ys.append(ys[-1] + lengths[:, i] * tf.sin(prev_angle) * sign)
        prev_angle = pi - angles[:, i] - prev_angle
        sign *= -1

    xs.append(xs[-1] + lengths[:, i+1] * tf.cos(prev_angle))
    ys.append(ys[-1] + lengths[:, i+1] * tf.sin(prev_angle) * sign)

    cartesians = tf.stack([tf.stack(xs, axis=1), tf.stack(ys, axis=1)], axis=2)

    return cartesians


def dihedrals_to_cartesian_tf(dihedrals, cartesian=None, central_atom_indices=None, no_omega=False):

    if not tf.is_numeric_tensor(dihedrals):
        dihedrals = tf.convert_to_tensor(dihedrals)
    if len(dihedrals.get_shape()) == 1:
        one_d = True
        dihedrals = tf.expand_dims(dihedrals, 0)
    else:
        one_d = False

    n = int(dihedrals.shape[-1])
    dihedrals = -dihedrals

    if cartesian is None:
        cartesian = tf.constant(straight_tetrahedral_chain(n + 3))
    cartesian = tf.tile(tf.expand_dims(cartesian, axis=0), [tf.shape(dihedrals)[0], 1, 1])

    if central_atom_indices is None:
        cai = list(range(n+3))
    else:
        cai = central_atom_indices

    for i in range(n):
        if not no_omega:
            j = i
        else:
            j = i + int((i+1)/2)
        axis = cartesian[:, cai[j+2]] - cartesian[:, cai[j+1]]
        axis /= tf.norm(axis, axis=1, keepdims=True)
        rotated = cartesian[:, cai[j+2]:cai[j+2]+1] + \
            tf.matmul(cartesian[:, cai[j+2]+1:] - cartesian[:, cai[j+2]:cai[j+2]+1],
                      rotation_matrix(axis, dihedrals[:, i]))
        cartesian = tf.concat([cartesian[:, :cai[j+2]+1], rotated], axis=1)

    if one_d:
        cartesian = tf.squeeze(cartesian)
    return cartesian
