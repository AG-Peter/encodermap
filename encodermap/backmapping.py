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
    batch_size = tf.shape(angles)[0]

    prev_angle = tf.zeros((batch_size))
    xs = [tf.zeros((batch_size))]
    ys = [tf.zeros((batch_size))]
    sign = 1

    for i in range(angles.shape[1]):
        xs.append(xs[-1] + lengths[:, i] * tf.cos(prev_angle))
        ys.append(ys[-1] + lengths[:, i] * tf.sin(prev_angle) * sign)
        prev_angle = pi - angles[:, i] - prev_angle
        sign *= -1

    xs.append(xs[-1] + lengths[:, i+1] * tf.cos(prev_angle))
    ys.append(ys[-1] + lengths[:, i+1] * tf.sin(prev_angle) * sign)

    xs = tf.stack(xs, axis=1)
    ys = tf.stack(ys, axis=1)
    cartesians = tf.stack([xs, ys, tf.zeros(tf.shape(xs))], axis=2)

    return cartesians


def dihedrals_to_cartesian_tf_old(dihedrals, cartesian=None, central_atom_indices=None, no_omega=False):

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
    if len(cartesian.get_shape()) == 2:
        cartesian = tf.tile(tf.expand_dims(cartesian, axis=0), [tf.shape(dihedrals)[0], 1, 1])

    if central_atom_indices is None:
        cai = list(range(cartesian.shape[1]))
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

    return cartesian


def dihedrals_to_cartesian_tf(dihedrals, cartesian):

    if not tf.is_numeric_tensor(dihedrals):
        dihedrals = tf.convert_to_tensor(dihedrals)

    n = int(dihedrals.shape[-1])

    if len(cartesian.get_shape()) == 2:
        cartesian = tf.tile(tf.expand_dims(cartesian, axis=0), [tf.shape(dihedrals)[0], 1, 1])

    split = int(int(cartesian.shape[1])/2)

    cartesian_right = cartesian[:, split-1:]
    dihedrals_right = dihedrals[:, split-1:]

    cartesian_left = cartesian[:, split+1::-1]
    dihedrals_left = dihedrals[:, split-2::-1]

    new_cartesian_right = dihedral_to_cartesian_tf_one_way(dihedrals_right, cartesian_right)
    new_cartesian_left = dihedral_to_cartesian_tf_one_way(dihedrals_left, cartesian_left)

    new_cartesian = tf.concat([new_cartesian_left[:, ::-1], new_cartesian_right[:, 3:]], axis=1)

    return new_cartesian


def dihedral_to_cartesian_tf_one_way(dihedrals, cartesian):
    n = int(dihedrals.shape[-1])
    dihedrals = -dihedrals

    rotated = cartesian[:, 1:]
    collected_cartesians = [cartesian[:, 0:1]]
    for i in range(n):
        collected_cartesians.append(rotated[:, 0:1])
        axis = rotated[:, 1] - rotated[:, 0]
        axis /= tf.norm(axis, axis=1, keepdims=True)
        offset = rotated[:, 1:2]
        rotated = offset + tf.matmul(rotated[:, 1:] - offset, rotation_matrix(axis, dihedrals[:, i]))
    collected_cartesians.append(rotated)
    collected_cartesians = tf.concat(collected_cartesians, axis=1)
    return collected_cartesians


# def dihedral_to_cartesian_tf_one_way2(dihedrals, cartesian):
#     n = int(dihedrals.shape[-1])
#     dihedrals = -dihedrals
#
#     n_batch = tf.shape(cartesian)[0]
#
#     new_cartesians = tf.Variable(np.zeros((256, int(cartesian.shape[1]), 3), dtype=np.float32), trainable=False)
#     new_cartesians = new_cartesians[:n_batch].assign(cartesian)
#
#     for i in range(n):
#         axis = new_cartesians[:n_batch, i + 2] - new_cartesians[:n_batch, i + 1]
#         axis /= tf.norm(axis, axis=1, keepdims=True)
#         new_cartesians[:n_batch, i + 3:].assign(new_cartesians[:n_batch, i + 2:i + 3] +
#                                          tf.matmul(new_cartesians[:n_batch, i + 3:] - new_cartesians[:n_batch, i + 2:i + 3],
#                                                    rotation_matrix(axis, dihedrals[:, i])))
#     return new_cartesians[:n_batch]

# def dihedrals_to_cartesian_tf(dihedrals, cartesian):
#
#     if not tf.is_numeric_tensor(dihedrals):
#         dihedrals = tf.convert_to_tensor(dihedrals)
#
#     n = int(dihedrals.shape[-1])
#     dihedrals = -dihedrals
#
#     if len(cartesian.get_shape()) == 2:
#         cartesian = tf.tile(tf.expand_dims(cartesian, axis=0), [tf.shape(dihedrals)[0], 1, 1])
#
#     for i in range(n):
#         axis = cartesian[:, i + 2] - cartesian[:, i + 1]
#         axis /= tf.norm(axis, axis=1, keepdims=True)
#         rotated = cartesian[:, i + 2:i + 2 + 1] + \
#                   tf.matmul(cartesian[:, i + 3:] - cartesian[:, i + 2:i + 3],
#                             rotation_matrix(axis, dihedrals[:, i]))
#         cartesian = tf.concat([cartesian[:, :i + 3], rotated], axis=1)
#
#     return cartesian


def guess_sp2_atom(cartesians, atom_names, bond_partner, angle_to_previous, bond_length):
    assert cartesians.shape[1] == len(atom_names)
    added_cartesians = []
    for i in range(1, len(atom_names)):
        if atom_names[i] == bond_partner:
            prev_vec = cartesians[:, i - 1] - cartesians[:, i]
            try:
                next_vec = cartesians[:, i + 1] - cartesians[:, i]
            except ValueError:
                next_vec = cartesians[:, i - 2] - cartesians[:, i]

            perpendicular_axis = tf.cross(prev_vec, next_vec)
            perpendicular_axis /= tf.norm(perpendicular_axis, axis=1, keepdims=True)
            bond_vec = tf.matmul(tf.expand_dims(prev_vec, 1), rotation_matrix(perpendicular_axis, angle_to_previous))
            bond_vec = bond_vec[:, 0, :]
            bond_vec *= bond_length / tf.norm(bond_vec, axis=1, keepdims=True)
            added_cartesians.append(cartesians[:, i] + bond_vec)
    added_cartesians = tf.stack(added_cartesians, axis=1)
    return added_cartesians


def guess_amide_H(cartesians, atom_names):
    return guess_sp2_atom(cartesians, atom_names, "N", 123/180*pi, 1.10)


def guess_amide_O(cartesians, atom_names):
    return guess_sp2_atom(cartesians, atom_names, "C", 121/180*pi, 1.24)


def merge_cartesians(central_cartesians, central_atom_names, H_cartesians, O_cartesians):
    cartesian = [central_cartesians[:, 0]]
    h_i = 0
    o_i = 0
    for i in range(1, len(central_atom_names)):
        atom_name = central_atom_names[i]
        cartesian.append(central_cartesians[:, i])
        if atom_name == "N":
            cartesian.append(H_cartesians[:, h_i])
            h_i += 1
        elif atom_name == "C":
            cartesian.append(O_cartesians[:, o_i])
            o_i += 1
    cartesian = tf.stack(cartesian, axis=1)
    assert cartesian.shape[1] == central_cartesians.shape[1] + H_cartesians.shape[1] + O_cartesians.shape[1]
    return cartesian
