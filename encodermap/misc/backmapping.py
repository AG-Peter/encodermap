# -*- coding: utf-8 -*-
# encodermap/misc/backmapping.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade, Tobias Lemke
#
# Encodermap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
# This package is distributed in the hope that it will be useful to other
# researches. IT DOES NOT COME WITH ANY WARRANTY WHATSOEVER; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# See <http://www.gnu.org/licenses/>.
################################################################################
"""Backmapping functions largely based upon encodermap_tf1's nackmapping an martini-tools backwards.py

    ToDo:
        * Using Quaternions in Tensorflow rotation matrices could be accelerated?
        * Multi Top.

"""

##############################################################################
# Imports
##############################################################################

import copy
from math import pi

import numpy as np
import tensorflow as tf

from .._optional_imports import _optional_import
from ..misc import transformations as trans
from ..misc.errors import BadError

##############################################################################
# Optional Imports
##############################################################################


md = _optional_import("mdtraj")
mda = _optional_import("MDAnalysis")
AnalysisFromFunction = _optional_import(
    "MDAnalysis", "analysis.base.AnalysisFromFunction"
)
MemoryReader = _optional_import("MDAnalysis", "coordinates.memory.MemoryReader")


##############################################################################
# Globals
##############################################################################


__all__ = ["backbone_hydrogen_oxygen_crossproduct"]


##############################################################################
# Public Functions
##############################################################################


def split_and_reverse_dihedrals(x):
    """Splits dihedrals in BackMapping model into left (reversed) and right part.
    These dihedrals are then used to bring the chain_in_plane into 3D.

    Args:
        x (tf.Tensor): The dihedrals with shape (None, n_reisudes * 3 - 3)

    Examples:
        >>> from encodermap.misc.backmapping import split_and_reverse_dihedrals
        >>> import numpy as np

        >>> # create dihedrals for protein with 3 resiudes, i.e. 3*3 - 3  = 6 central dihedral angles
        >>> # single sample will be used -> shape = (1, 6)
        >>> np.random.seed(20)
        >>> dihedrals = np.random.random((1, 6)) * 2 * np.pi
        >>> print(dihedrals)
        [[3.69533481 5.64050171 5.60165278 5.12605805 0.22550092 4.34644107]]

        >>> dihedrals_left, dihedrals_right = split_and_reverse_dihedrals(dihedrals)
        >>> print(dihedrals_left, dihedrals_right)
        [[5.60165278 5.64050171 3.69533481]] [[5.12605805 0.22550092 4.34644107]]

    """
    if tf.executing_eagerly():
        middle = int(x.shape[1] / 2)
        if tf.math.equal(tf.math.mod(x.shape[1], 2), 0):
            return x[:, middle - 1 :: -1], x[:, middle:]
        else:
            return x[:, middle::-1], x[:, middle + 1 :]
    else:
        middle = int(x.shape[1] / 2)
        cond = tf.math.equal(tf.math.mod(x.shape[1], 2), 0)
        return tf.cond(
            cond,
            true_fn=lambda: (x[:, middle - 1 :: -1], x[:, middle:]),
            false_fn=lambda: (x[:, middle::-1], x[:, middle + 1 :]),
        )


def split_and_reverse_cartesians(x):
    """Splits cartesians and returns a left (reversed) right part.

    Because dihedrals are made up from 4 atoms, three atoms are
    identical in the left and right part of the list. This holds true:
    left[0] = right[2]
    left[1] = right[1]
    left[2] = right[0]

    Args:
        x (tf.Tensor): The cartesians with shape (None, n_reisudes * 3, 3)

    Examples:
        >>> from encodermap.misc.backmapping import split_and_reverse_cartesians
        >>> import numpy as np

        >>> # create cartesians for protein with 3 resiudes, i.e. 9
        >>> # single sample will be used -> shape = (1, 9, 3)
        >>> np.random.seed(20)
        >>> cartesians = np.random.random((1, 9, 3)) * 10

        >>> cartesians_left, cartesians_right = split_and_reverse_cartesians(cartesians)

        >>> print(cartesians_left.shape, cartesians_right.shape)
        (1, 6, 3) (1, 6, 3)

        >>> print(cartesians_left[:,0] == cartesians_right[:,2])
        [[ True  True  True]]

        >>> print(cartesians_left[:,1] == cartesians_right[:,1])
        [[ True  True  True]]

        >>> print(cartesians_left[:,2] == cartesians_right[:,0])
        [[ True  True  True]]



    """
    middle = int(x.shape[1] / 2)
    return x[:, middle + 1 :: -1], x[:, middle - 1 :]


def dihedrals_to_cartesian_tf_layers(dihedrals, cartesians):
    """Calculates dihedrals to cartesians in Graph/Layer execution.

    Args:
        dihedrals (tf.Tensor): The dihedrals of shape (None, n_resides * 3 - 3)
        cartesians (tf.Tensor): The cartesians of shaoe (None, n_residues * 3, 3).

    """

    if len(cartesians.get_shape()) == 2:
        # if a single line of cartesians is passed it is repeated to match the number of dihedrals
        cartesians = Lambda(
            lambda x: tf.tile(tf.expand_dims(x[0], axis=0), [tf.shape(x[1])[0], 1, 1])
        )((cartesians, dihedrals))

    # split and reverse so that the center of the molecule stays in the 2D plane
    # and the left and right ends curl into the 3rd dimension
    cartesians_left, cartesians_right = split_and_reverse_cartesians(cartesians)
    dihedrals_left, dihedrals_right = split_and_reverse_dihedrals(dihedrals)

    new_cartesians_left = dihedral_to_cartesian_tf_one_way_layers(
        dihedrals_left, cartesians_left, int(dihedrals.shape[1] / 2)
    )
    new_cartesians_right = dihedral_to_cartesian_tf_one_way_layers(
        dihedrals_right, cartesians_right, int(dihedrals.shape[1] / 2)
    )

    new_cartesians = tf.concat(
        [new_cartesians_left[:, ::-1], new_cartesians_right[:, 3:]], axis=1
    )

    return new_cartesians


def mdtraj_backmapping(
    top,
    dihedrals,
    sidechain_dihedrals=None,
    trajs=None,
    fake_dihedrals=False,
    verify_every_rotation=True,
    angle_type="radian",
):
    """Uses MDTraj and Christoph Gohlke's transformations.py to rotate the bonds in the provided topology.

    Input currently only in angles.

    General procedure:
        * Decide on which topology to use (if different topologies are in the TrajEnsemble class the `dihedrals` and
            `sidechain_dihedrals` arrays need to be altered so that the correct dihedrals are used. Because EncoderMap
            is trained on a full input `dihedrals` and `sidechain_dihedrals` contain the dihedrals for the topology
            in `TrajEnsemble` with the most of such angles. Some SingleTraj classes in TrajEnsemble might not contain all these
            angles if for example an amino acid has been modified the mutant contains more sidechain dihedrals than the
            wt. So the correct sidechain dihedrals for the wildtype need to be selected.
        * Get the indices of the far sides of the rotations. The graph is gradually broken apart and the longer
            subgraphs are kept.
        * Extend the trajectory. The lengths of dihedrals and sidechain_dihedrals should match. The frame given by top
            will be duplicated len(dihedrals)-times.
        * Get the current angles. We know what the final angles should be, but now how far to rotate the bonds. This
            can be done by getting the difference between current and target angle.
        * Rotate the bonds. Using Christoph Gohlke's transformations.py, the rotation matrix is constructed and
            the array is padded with zeros to resemble an array of quaternions.

    Args:
        top (str): The topology file to use.
        dihedrals (np.ndarray): The dihedrals to put onto the trajectory. len(dihedrals) is number of frames of
            output trajectory. dihedrals.shape[1] needs to be the same as the number of dihedrals in the topology.
        sidechain_dihedrals (Union[np.ndarray, None], optional): The sidechain dihedrals to put onto the trajectory.
            If None is provided, the sidechains are kept like they were in the topology. Defaults to None.
        trajs (Union[em.TrajEnsemble, None], optional): Encodermap TrajEnsemble class. Can accelerate loading of
            current dihedral angles. Also checks of provided topology is part of trajs. Defaults to None.
        fake_dihedrals (bool, optional): Whether to fake dihedrals. For debugging. Defaults to False.
        verify_every_rotation (bool, optional): Whether the rotation succeeded.
        angle_type (bool, optional): Whether input is in degrees. Input will be converted to radians. Defaults to False.

    Raises:
        Exception: If the input seems like it is in degrees.
        Exception: If top is not part of the TrajEnsemble class provided in argument `trajs`.

    Returns:
        mdtraj.Trajectory: An MDTraj trajectory with the correct dihedrals/side-dihedrals.

    """
    import networkx as nx

    # change the angles
    if angle_type == "radian":
        pass
    elif angle_type == "degree":
        dihedrals = np.deg2rad(dihedrals)
        sidechain_dihedrals = np.deg2rad(sidechain_dihedrals)
    else:
        raise Exception("Argument `angle_type` must be either 'radian' or 'degree'.")

    # make sure the input has the same shape along the "frame" axis, that will be created.
    if sidechain_dihedrals is not None:
        assert len(dihedrals) == len(sidechain_dihedrals)

    # decide on what to do with trajs. If it was supplied it can
    # either be TrajEnsemble or SingleTraj
    if trajs is not None:
        # TrajEnsemble
        if hasattr(trajs, "n_trajs"):
            # The topology in `top` should also be present in traj
            if not top in trajs._top_files:
                raise Exception(
                    f"Provided topology is not part of TrajEnsemble object. Possible tops are {trajs._top_files}"
                )

            # Deciding which traj to use
            ind = [top == i for i in trajs._top_files].index(True)
            traj = trajs[ind]
            print(
                f"Using trajectory {traj} as parent for backmapping. Because its "
                f"topoloy file ({traj.top_file}) matches the file provided as "
                f"argument `top` ({top})."
            )
        # SingleTraj

        else:
            traj = trajs

        # load the CVs to use the indices
        traj.load_CV("all")
        inp_traj = md.load(traj.top_file)
    else:
        # either build info Single and load the CVs, which is currently broken for pdbs
        from ..trajinfo import info_single

        try:
            traj = info_single.SingleTraj(top)
            traj.load_CV("all")
            inp_traj = md.load(top)
        except OSError as e:
            # this is a weird PyEMMA error that keeps happening
            if "REMARK" in e.__str__():
                traj = md.load(top)
                inp_traj = md.load(top)
                angles = ["psi", "omega", "phi"]
                dihedrals_ = [
                    getattr(md, f"compute_{a}")(traj)[0].tolist() for a in angles
                ]
                results = [None] * (
                    len(dihedrals_[0]) + len(dihedrals_[0]) + len(dihedrals_[2])
                )
                results[::3] = dihedrals_[0]
                results[1::3] = dihedrals_[1]
                results[2::3] = dihedrals_[2]
                dih_indices = np.array(results)
            else:
                raise e

    # get indices of atoms for rotations
    g = inp_traj.top.to_bondgraph()
    # nx.draw(g, pos=nx.spring_layout(g))
    if not nx.is_connected(g):
        raise BadError(
            f"MDTraj parsed the topology at {top} and found it disconnected. Changing dihedrals in multiple "
            f"chains is currently not possible. If you are sure your protein is just one chain you can try "
            f"the MDAnalysis backmapping backend or provide a topology of the file with manually fixed bonds."
        )

    # get near and far sides
    # dih indices are four atoms
    # bond indices are the two atoms in the middle giving the axis of rotation
    if hasattr(traj, "_CVs"):
        if "central_dihedrals" in traj._CVs.attrs:
            dih_indices = np.asarray(traj._CVs.attrs["central_dihedrals"])

    # at this point dih_bond_indices has been defined.
    # either via PyEMMA featurizer or the compute_phi/omega/psi methods of mdtraj
    dih_bond_indices = dih_indices[:, 1:3]

    # filter out the Proline angles
    dih_bond_atoms = np.dstack(
        [
            [traj.top.atom(a).__str__() for a in dih_bond_indices[:, 0]],
            [traj.top.atom(a).__str__() for a in dih_bond_indices[:, 1]],
        ]
    )[0]
    indices = np.arange(len(dih_bond_indices)).tolist()
    for i, bond in enumerate(dih_bond_atoms):
        if "PRO" in bond[0] and "PRO" in bond[1] and "N" in bond[0] and "CA" in bond[1]:
            indices.remove(i)

    dih_indices = dih_indices[indices]
    dih_bond_indices = dih_bond_indices[indices]
    dihedrals = dihedrals[:, indices]
    dih_near_sides, dih_far_sides = _get_far_and_near_networkx(
        g, dih_bond_indices, inp_traj.top
    )

    if sidechain_dihedrals is not None:
        if "side_dihedrals" not in traj._CVs.attrs:
            try:
                traj.load_CV("all")
            except OSError as e:
                pass
            raise NotImplementedError(
                f"This traj produces some error with PyEMMA: {e} " ""
            )

        side_indices = np.asarray(traj._CVs.attrs["side_dihedrals"])
        side_bond_indices = side_indices[:, 1:3]
        # filter out the Proline angles
        side_bond_atoms = np.dstack(
            [
                [traj.top.atom(a).__str__() for a in side_bond_indices[:, 0]],
                [traj.top.atom(a).__str__() for a in side_bond_indices[:, 1]],
            ]
        )[0]
        indices = np.arange(len(side_bond_indices)).tolist()
        for i, bond in enumerate(side_bond_atoms):
            if (
                "PRO" in bond[0]
                and "PRO" in bond[1]
                and "CA" in bond[0]
                and "CB" in bond[1]
            ):
                indices.remove(i)
            if (
                "PRO" in bond[0]
                and "PRO" in bond[1]
                and "CB" in bond[0]
                and "CG" in bond[1]
            ):
                indices.remove(i)

        side_indices = side_indices[indices]
        side_bond_indices = side_bond_indices[indices]
        sidechain_dihedrals = sidechain_dihedrals[:, indices]

        side_near_sides, side_far_sides = _get_far_and_near_networkx(
            g, side_bond_indices, inp_traj.top
        )

    # extend the traj
    for i in range(len(dihedrals)):
        if i == 0:
            out_traj = copy.deepcopy(inp_traj)
        else:
            out_traj = out_traj.join(inp_traj)

    if fake_dihedrals:
        print("Faking dihedrals for testing purposes.")
        # dihedrals = np.vstack([current_angles for i in range(len(dihedrals))])
        # dihedrals[:, 0] = np.linspace(-170, 170, len(dihedrals))
        dihedrals = np.dstack(
            [np.linspace(-170, 170, len(dihedrals)) for i in range(dihedrals.shape[1])]
        ).squeeze()
        sidechain_dihedrals = np.dstack(
            [
                np.linspace(-170, 170, len(sidechain_dihedrals))
                for i in range(sidechain_dihedrals.shape[1])
            ]
        ).squeeze()

    # adjust the torsions
    new_xyz = copy.deepcopy(out_traj.xyz)
    for i in range(dihedrals.shape[0]):
        for j in range(dihedrals.shape[1]):
            # central_dihedrals
            near_side = dih_near_sides[j]
            far_side = dih_far_sides[j]
            dihedral = dih_indices[j]
            bond = dih_bond_indices[j]

            # define inputs
            target_angle = dihedrals[i, j]
            current_angle = _dihedral(new_xyz[i], dihedral)[0][0]
            angle = target_angle - current_angle
            direction = np.diff(new_xyz[i, bond], axis=0).flatten()
            pivot_point = new_xyz[i, bond[0]]

            # perform rotation
            rotmat = trans.rotation_matrix(angle, direction, pivot_point)
            padded = np.pad(
                new_xyz[i][far_side],
                ((0, 0), (0, 1)),
                mode="constant",
                constant_values=1,
            )
            new_xyz[i][far_side] = rotmat.dot(padded.T).T[:, :3]

            if i == 0 and j == 0 and verify_every_rotation:
                dih_indexes = traj._CVs.attrs["central_dihedrals"][j]
                s = f"Near and far side for dihedral {[str(traj.top.atom(x)) for x in dih_indexes]} are:"
                s += (
                    f"\nNear: {[str(traj.top.atom(x)) for x in near_side]}, {near_side}"
                )
                s += f"\nFar: {[str(traj.top.atom(x)) for x in dih_far_sides[j][:12]]}..., {dih_far_sides[j][:12]}..."
                s += f"\nRotation around bond {[str(traj.top.atom(x)) for x in bond]}, {bond}."
                s += f"\nPositions of near side before rotation are\n{out_traj.xyz[i][near_side]}."
                s += f"\nPositions of near side after rotation aren\n{new_xyz[i][near_side]}"
                print(s)

            # verify
            if verify_every_rotation:
                _ = _dihedral(new_xyz[i], dihedral)[0][0]
                if not np.isclose(_, target_angle, atol=1e-3):
                    s = (
                        f"Adjusting dihedral angle for atoms {[str(traj.top.atom(x)) for x in dihedral]} failed with a tolerance of 1e-4."
                        f"\nTarget angle was {target_angle} {angle_type}, but rotation yieled angle with {_} {angle_type}."
                        f"\nCurrent angle was {current_angle}. To reach target angle is a rotation of {angle} {angle_type} was carried out."
                        f"\nRotation axis was vector from {traj.top.atom(bond[0])} to {traj.top.atom(bond[1])}"
                        f"\nOnly these atoms should have been affected by rotation: {far_side}"
                        "\nBut somehow this method still crashed. Maybe these prints will help."
                    )
                    raise BadError(s)

        if sidechain_dihedrals is not None:
            for j in range(sidechain_dihedrals.shape[1]):
                # central_dihedrals
                near_side = side_near_sides[j]
                far_side = side_far_sides[j]
                dihedral = dih_indices[j]
                bond = side_indices[j]

                # define inputs
                target_angle = sidechain_dihedrals[i, j]
                current_angle = np.rad2deg(_dihedral(new_xyz[i], dihedral))[0][0]
                angle = target_angle - current_angle
                direction = np.diff(new_xyz[i, bond], axis=0).flatten()
                pivot_point = new_xyz[i, bond[0]]

                # perform rotation
                rotmat = trans.rotation_matrix(angle, direction, pivot_point)
                padded = np.pad(
                    new_xyz[i][far_side],
                    ((0, 0), (0, 1)),
                    mode="constant",
                    constant_values=1,
                )
                new_xyz[i][far_side] = rotmat.dot(padded.T).T[:, :3]

    # overwrite traj and return
    out_traj.xyz = new_xyz
    return out_traj


def _get_far_and_near_networkx(bondgraph, edge_indices, top=None):
    """Returns near and far sides for a list of edges giving the indices of the two atoms at which the structure is broken.

    Args:
        bondgraph (networkx.classes.graph.Graph): The bondgraph describing the protein.
        edge_indices (np.ndarray): The edges the graph will be broken at.

    Returns:
        tuple: A tuple containing the following:
            near_sides (list of np.ndarray): List of integer arrays giving the near sides. len(near_sides) == len(edge_indices).
            far_sides (list of np.ndarray): Same as near sides, but this time the far sides.

    """
    import networkx as nx
    from networkx.algorithms.components.connected import connected_components

    near_sides = []
    far_sides = []
    for i, edge in enumerate(edge_indices):
        G = nx.convert_node_labels_to_integers(bondgraph).copy()
        G.remove_edge(*edge)
        components = [*connected_components(G)]
        if len(components) != 2:
            if top is None:
                raise Exception(
                    f"Splitting the topology of the trajectory at the edge "
                    f"{edge} does not work. Provide a topology to see, "
                    "which atoms are affected"
                )
            else:
                raise Exception(
                    f"Splitting at edge {edge} does not work. Here are the "
                    f"atoms: {top.atom(edge[0])} and {top.atom(edge[1])}."
                )

        if edge[1] in components[0] and edge[0] in components[1]:
            components = components[::-1]
        assert len(components) == 2, print(
            f"Protein might be cyclic or contain more than 1 chain. {len(components)}"
        )
        assert edge[0] in components[0] and edge[1] in components[1], print(
            "Finding near and far sides failed."
        )
        subgraph = G.subgraph(components[-1]).copy()
        far_sides.append(np.asarray(subgraph.nodes))
        subgraph = G.subgraph(components[0]).copy()
        near_sides.append(np.asarray(subgraph.nodes))
    return near_sides, far_sides


def _dihedral(xyz, indices):
    """Returns current dihedral angle between positions.

    Adapted from MDTraj.

    Args:
        xyz (np.ndarray). This function only takes a xyz array of a single frame and uses np.expand_dims()
            to make that fame work with the `_displacement` function from mdtraj.
        indices (Union[np.ndarray, list]): List of 4 ints describing the dihedral.

    """
    indices = np.expand_dims(np.asarray(indices), 0)
    xyz = np.expand_dims(xyz, 0)
    ix10 = indices[:, [0, 1]]
    ix21 = indices[:, [1, 2]]
    ix32 = indices[:, [2, 3]]

    b1 = _displacement(xyz, ix10)
    b2 = _displacement(xyz, ix21)
    b3 = _displacement(xyz, ix32)

    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 *= (b2 * b2).sum(-1) ** 0.5
    p2 = (c1 * c2).sum(-1)

    return np.arctan2(p1, p2, None)


def _displacement(xyz, pairs):
    "Displacement vector between pairs of points in each frame"
    value = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    assert value.shape == (
        xyz.shape[0],
        pairs.shape[0],
        3,
    ), "v.shape %s, xyz.shape %s, pairs.shape %s" % (
        str(value.shape),
        str(xyz.shape),
        str(pairs.shape),
    )
    return value


def dihedral_to_cartesian_tf_one_way_layers(dihedrals, cartesian, n):
    dihedrals = -dihedrals

    rotated = cartesian[:, 1:]
    collected_cartesians = [cartesian[:, 0:1]]
    for i in range(n):
        collected_cartesians.append(rotated[:, 0:1])
        axis = rotated[:, 1] - rotated[:, 0]
        axis /= tf.norm(axis, axis=1, keepdims=True)
        offset = rotated[:, 1:2]
        rotated = offset + tf.matmul(
            rotated[:, 1:] - offset, rotation_matrix(axis, dihedrals[:, i])
        )
    collected_cartesians.append(rotated)
    collected_cartesians = tf.concat(collected_cartesians, axis=1)
    return collected_cartesians


def backbone_hydrogen_oxygen_crossproduct(backbone_positions):
    assert backbone_positions.shape[2] % 3 == 0  # C, CA, N atoms, multiple of three
    pass


def guess_sp2_atom(cartesians, indices, angle_to_previous, bond_length):
    added_cartesians = []
    for i in indices:
        prev_vec = cartesians[:, i - 1] - cartesians[:, i]
        try:
            next_vec = cartesians[:, i + 1] - cartesians[:, i]
        except tf.errors.InvalidArgumentError:
            next_vec = cartesians[:, i - 2] - cartesians[:, i]

        perpendicular_axis = tf.linalg.cross(prev_vec, next_vec)
        perpendicular_axis /= tf.norm(perpendicular_axis, axis=1, keepdims=True)
        bond_vec = tf.matmul(
            tf.expand_dims(prev_vec, 1),
            rotation_matrix(perpendicular_axis, angle_to_previous),
        )
        bond_vec = bond_vec[:, 0, :]
        bond_vec *= bond_length / tf.norm(bond_vec, axis=1, keepdims=True)
        added_cartesians.append(cartesians[:, i] + bond_vec)
    added_cartesians = tf.stack(added_cartesians, axis=1)
    return added_cartesians


def guess_amide_H(cartesians, N_indices):
    return guess_sp2_atom(cartesians, N_indices[1::], 123 / 180 * pi, 1.10)


def guess_amide_O(cartesians, C_indices):
    return guess_sp2_atom(cartesians, C_indices, 121 / 180 * pi, 1.24)


def rotation_matrix(axis_unit_vec, angle):
    angle = tf.expand_dims(tf.expand_dims(angle, axis=-1), axis=-1)
    i = tf.expand_dims(tf.eye(3), 0)
    zeros = tf.zeros(tf.shape(axis_unit_vec)[0])
    cross_prod_matrix = tf.convert_to_tensor(
        [
            [zeros, -axis_unit_vec[:, 2], axis_unit_vec[:, 1]],
            [axis_unit_vec[:, 2], zeros, -axis_unit_vec[:, 0]],
            [-axis_unit_vec[:, 1], axis_unit_vec[:, 0], zeros],
        ]
    )
    cross_prod_matrix = tf.transpose(cross_prod_matrix, [2, 0, 1])
    r = tf.cos(angle) * i
    r += tf.sin(angle) * cross_prod_matrix
    axis_unit_vec = tf.expand_dims(axis_unit_vec, 2)
    r += (1 - tf.cos(angle)) * tf.matmul(
        axis_unit_vec, tf.transpose(axis_unit_vec, [0, 2, 1])
    )
    return r


def merge_cartesians(
    central_cartesians, N_indices, O_indices, H_cartesians, O_cartesians
):
    cartesian = [central_cartesians[:, 0]]
    h_i = 0
    o_i = 0
    for i in range(1, central_cartesians.shape[1]):
        cartesian.append(central_cartesians[:, i])
        if i in N_indices[1::]:
            cartesian.append(H_cartesians[:, h_i])
            h_i += 1
        elif i in O_indices:
            cartesian.append(O_cartesians[:, o_i])
            o_i += 1
    cartesian = tf.stack(cartesian, axis=1)
    assert (
        cartesian.shape[1]
        == central_cartesians.shape[1] + H_cartesians.shape[1] + O_cartesians.shape[1]
    )
    return cartesian


def dihedral_backmapping(
    pdb_path, dihedral_trajectory, rough_n_points=-1, sidechains=None
):
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
    if sidechains is not None:
        sidechain_dihedral_trajectory = sidechains[::step_size]

    uni = mda.Universe(pdb_path, format="PDB")
    protein = uni.select_atoms("protein")

    dihedrals = []
    sidechain_dihedrals = []

    for residue in protein.residues:
        psi = residue.psi_selection()
        if psi:
            dihedrals.append(psi)

    for residue in protein.residues:
        omega = residue.omega_selection()
        if omega:
            dihedrals.append(omega)

    for residue in protein.residues:
        phi = residue.phi_selection()
        if phi:
            dihedrals.append(phi)

    if sidechains is not None:
        for residue in protein.residues:
            chi1 = residue.chi1_selection()
            if chi1:
                sidechain_dihedrals.append(chi1)

        for residue in protein.residues:
            if "chi2" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi2_selection())

        for residue in protein.residues:
            if "chi3" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi3_selection())

        for residue in protein.residues:
            if "chi4" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi4_selection())

        for residue in protein.residues:
            if "chi5" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi5_selection())

    if sidechains is not None:
        if sidechain_dihedral_trajectory.shape[1] == len(sidechain_dihedrals) * 2:
            sidechain_dihedral_trajectory = sidechain_dihedral_trajectory[:, ::2]

    _expand_universe(uni, len(dihedral_trajectory))

    if sidechains is None:
        for dihedral_values, step in zip(dihedral_trajectory, uni.trajectory):
            for dihedral, value in zip(dihedrals, dihedral_values):
                _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
    else:
        for dihedral_values, sidechain_dihedral_values, step in zip(
            dihedral_trajectory, sidechain_dihedral_trajectory, uni.trajectory
        ):
            for dihedral, value in zip(dihedrals, dihedral_values):
                _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
            for dihedral, value in zip(sidechain_dihedrals, sidechain_dihedral_values):
                _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
    return uni


def _set_dihedral(dihedral, atoms, angle):
    current_angle = dihedral.dihedral.value()
    head = atoms[dihedral[2].id :]
    vec = dihedral[2].position - dihedral[1].position
    head.rotateby(angle - current_angle, vec, dihedral[2].position)


def _expand_universe(universe, length):
    coordinates = (
        AnalysisFromFunction(lambda ag: ag.positions.copy(), universe.atoms)
        .run()
        .results
    )["timeseries"]
    coordinates = np.tile(coordinates, (length, 1, 1))
    universe.load_new(coordinates, format=MemoryReader)
