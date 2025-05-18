# -*- coding: utf-8 -*-
# encodermap/misc/rotate.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
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
"""Helpers to apply rotations to molecular coordinates.

"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union, overload

# Third Party Imports
import numpy as np
import transformations as trans
from optional_imports import _optional_import

# Encodermap imports
from encodermap.trajinfo.trajinfo_utils import _delete_bond


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    import networkx as nx
    from mdtraj.core.topology import Atom


################################################################################
# Globals
################################################################################


__all__: list[str] = ["mdtraj_rotate"]


################################################################################
# Functions
################################################################################


def arbitrary_dihedral(
    pos: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes the dihedral angles of a position array with shape (n_frames, 4).

    Args:
        pos (np.ndarray): The positions between which to calculate the dihedrals.
        out (np.ndarray): A location into which the result is stored. If provided,
            it must have a shape that the inputs broadcast to. If not provided
            or None, a freshly-allocated array is returned. A tuple (possible
            only as a keyword argument) must have length equal to the number of outputs.

    Returns:
        np.ndarray: The dihedral angles in radians.

    """
    p0 = pos[:, 0]
    p1 = pos[:, 1]
    p2 = pos[:, 2]
    p3 = pos[:, 3]

    b1 = -1.0 * (p1 - p0)
    b2 = p2 - p1
    b3 = p3 - p2

    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 *= (b2 * b2).sum(-1) ** 0.5
    p2 = (c1 * c2).sum(-1)

    return np.arctan2(p1, p2, out)


def mdtraj_rotate(
    traj: md.Trajectory,
    angles: np.ndarray,
    indices: np.ndarray,
    deg: bool = False,
    check_cyclic_backbone: bool = True,
    verify_every_rotation: bool = False,
    drop_proline_angles: bool = False,
    delete_sulfide_bridges: bool = True,
) -> md.Trajectory:
    """Uses MDTraj and Christoph Gohlke's transformations.py to set bond
    rotations provided traj.

    Input can be in radian (set `deg` to False) or degree (set `deg` to True).

    General procedure:
        * Carry out some checks. Shapes of input need to be correct. `traj`
            needs to have a single frame and not be of a cyclic protein nor
            contain multiple chains.
        * Get the indices of the near and far side of the rotations. Every
            dihedral angle is indexed by 4 atoms. The rotational axis is located
            between the central two atoms (dihedral[1:3]).
        * Extend the trajectory. The lengths of dihedrals and sidechain_dihedrals
            should match. The frame given by top will be duplicated
            len(dihedrals)-times.
        * Get the current angles. We know what the final angles should be, but
            now how far to rotate the bonds. This can be done by getting the
            difference between current and target angle.
        * Rotate the bonds. Using Christoph Gohlke's transformations.py,
            the rotation matrix is constructed and the array is padded with
            zeros to resemble an array of quaternions.

    Args:
        traj (mdtraj.Trajectory): The trajectory to use. Needs to have only one frame.
        angles (list[list[float]], np.ndarray): The values the angles should
            assume after the rotations. This arg can either be a nested list
            with floats or (better) a numpy array with the shape angles.shape =
            (n_new_frames, n_indexed_dihedrals). Here, angles.shape[0] defines
            how many frames the output trajectory is going to have and angles.shape[1]
            should be similar to the number of dihedrals you want to rotate around.
            A shape of (4, 2) would indicate that two dihedrals are going to be
            used for rotation and the output trajectory is going to have 4 frames.
        indices (list[list[int]], np.ndarray): A list of ints indexing the
            dihedrals to be rotated around. Naturally indices.shape[1] needs
            to be 4. Additionally indices.shape[0] needs to be the same as
            angles.shape[1]. indices indexes the angles along axis 1 and angles
            sets the values of those angles along axis 0.
        deg (bool, optional): Whether argument `angles` is in deg.
            Defaults to False.
        check_cyclic_backbone (bool): Whether the backbone should be
            checked for being cyclic. Rotating around a backbone angle for a
            cyclic protein is not possible and thus an Exception is raised.
            However, rotation around sidechain dihedrals is still possible.
            If you are sure you want to rotate sidechain dihedrals set this
            argument to False to prevent the cyclic backbone check.
            Defaults to True.
        verify_every_rotation (bool): Whether the rotation succeeded.
        drop_proline_angles (bool): Whether to automatically drop proline
            angles and indices.
        delete_sulfide_bridges (bool): Whether to automatically remove bonds from
            between cysteine residues.

    Raises:
        Exception: If the input seems like it is in degrees, but `deg` is False.
        Exception: If `traj` contains more than 1 frame.
        Exception: If traj is not fully connected.
        Exception: If shapes of `angles` and `indices` mismatches.
        Exception: If shape[1] of `indices` is not 4.
        Exception: If backbone is cyclic and check_cyclic_backbone is True.
        Exception: If the first rotation does not reach a tolerance of 1e-3.

    Returns:
        mdtraj.Trajectory: An MDTraj trajectory with applied rotations.

    Examples:
        >>> import mdtraj as md
        >>> import numpy as np

        >>> # load an arbitrary protein from the pdb
        >>> traj = md.load_pdb('https://files.rcsb.org/view/1GHC.pdb')
        >>> print(traj.n_frames)
        14

        >>> # traj has multiple frames so we remove all but one
        >>> traj = traj[0]

        >>> # Get indices of psi_angles
        >>> psi_indices, old_values = md.compute_psi(traj)

        >>> # set every psi angle to be either 0 or 180 deg
        >>> angles = np.full((len(psi_indices), 2), [0, 180]).T

        >>> # create the new traj with the desired rotations
        >>> out_traj = mdtraj_rotate(traj, angles, psi_indices, deg=True)
        >>> print(out_traj.n_frames)
        2

        >>> # check values
        >>> _, new_values = md.compute_psi(out_traj)
        >>> print(np.abs(np.rad2deg(new_values[0, :2]).round(0))) # prevent rounding inconsistencies
        [0. 0.]
        >>> print(np.abs(np.rad2deg(new_values[1, :2]).round(0))) # prevent rounding inconsistencies
        [180. 180.]

    """
    # Third Party Imports
    import networkx as nx

    if deg:
        angles = np.deg2rad(angles)
    else:
        if np.any(angles > 2 * np.pi):
            print(
                "Some of your input for `angles` is larger than 2pi. "
                "This suggests, that you provided `angles` in deg not in rad. "
                "Please set the argument `deg` to True to transpose the values "
                "of `angles`."
            )

    # make np arrays
    indices = np.asarray(indices).astype(int)
    angles = np.asarray(angles)

    # if only one rotation/index is given expand dims
    if angles.ndim == 1:
        angles = np.expand_dims(angles, -1)
    if indices.ndim == 1:
        indices = np.expand_dims(indices, 0)

    # check whether traj has only one frame:
    if traj.n_frames > 1:
        raise Exception(
            f"The provided `traj` has {traj.n_frames}. "
            f"Please provide a traj with only 1 frame."
        )
    traj = deepcopy(traj)

    # check if shape of indices and dihedrals is consistent
    if angles.shape[1] != indices.shape[0]:
        raise Exception(
            f"Shapes of `angles` and `indices` mismatch. Shape[1] of `angles` is "
            f"{angles.shape[1]}, which indicates that you also want to rotate "
            f"around {angles.shape[1]} dihedral(s), but indices indexes "
            f"{indices.shape[0]} dihedral angle(s). The shapes of the inputs "
            f"need to match. indices.shape[0] == angles.shape[1]"
        )

    # check whether 4 atoms are indexed
    if indices.shape[1] != 4:
        raise Exception(
            f"The shape of `indices` needs to be (n_dihedrals, 4), meaning, that "
            f"a dihedral angle is defined by 4 atoms. Your `indices` argument` "
            f"has shape {indices.shape} which is not allowed."
        )

    # check whether structure is whole
    g = traj.top.to_bondgraph()
    if not nx.is_connected(g):
        raise Exception(
            "Structure is disjoint and not fully connected. This can be caused by"
            "multiple problems: Topology was not parsed correctly "
            "(nonstandard residues), traj contains multiple proteins (chains), "
            "or water molecules. You can add the bonds with "
            "`mdtraj.Topology.add_bond()`, if you know what bonds exactly are missing."
        )

    # check whether is cyclic
    backbone = traj.atom_slice(traj.top.select("backbone"))
    try:
        edges = nx.algorithms.cycles.find_cycle(backbone.top.to_bondgraph())
    except nx.NetworkXNoCycle:
        pass
    else:
        if check_cyclic_backbone:
            raise Exception(
                "The Backbone of your protein is cyclic. You can not rotate "
                "around the backbone dihedral (Ramachandran) angles. "
                "You can still rotate around sidechain angles. For that set the "
                "`check_cyclic_backbone` argument of this function to "
                "True to silence this warning."
            )

    # remove proline angles from angles and indices
    if drop_proline_angles:
        offending_atoms = set(traj.top.select("resname PRO and (name CA or name N)"))
        slice = []
        for i, ind in enumerate(indices):
            ind = ind[1:3]
            if len(offending_atoms.intersection(set(ind))) == 2:
                slice.append(i)
        slice = np.array(slice)
        angles = np.delete(angles, slice, axis=1)
        indices = np.delete(indices, slice, axis=0)
        assert not offending_atoms.issubset(set(indices[:, 1:3].flatten()))

    # delete sulfide bridges from the topology
    if delete_sulfide_bridges:
        for i, (a, b) in enumerate(traj.top.bonds):
            if (
                a.element.symbol == "S"
                and b.element.symbol == "S"
                and a.residue.index != b.residue.index
            ):
                traj.top = _delete_bond(traj.top, (a, b))
        g = traj.top.to_bondgraph()

    # get near and far sides
    dih_near_sides, dih_far_sides = _get_near_and_far_networkx(
        g, indices[:, 1:3], top=traj.top
    )

    # extend the traj
    for i in range(len(angles)):
        if i == 0:
            out_traj = copy.deepcopy(traj)
        else:
            out_traj = out_traj.join(traj)

    # adjust the torsions
    new_xyz = copy.deepcopy(out_traj.xyz)
    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            # central_dihedrals
            near_side = dih_near_sides[j]
            far_side = dih_far_sides[j]
            dihedral = indices[j]
            bond = dihedral[1:3]

            # define inputs
            target_angle = angles[i, j]
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
                _ = np.rad2deg(_dihedral(new_xyz[i], dihedral))[0][0]
                if not np.isclose(_, target_angle, atol=1e-3):
                    s = f"Adjusting dihedral angle for atoms {[str(traj.top.atom(x)) for x in dihedral]} failed with a tolerance of 1e-4."
                    s += f"\nTarget angle was {target_angle} deg, but rotation yieled angle with {_} deg."
                    s += f"\nCurrent angle was {current_dih[j]}. To reach target angle is a rotation of {angle} degrees was carried out."
                    s += f"\nRotation axis was vector from {traj.top.atom(bond[0])} to {traj.top.atom(bond[1])}"
                    s += f"\nOnly these atoms should have been affected by rotation: {far_side}"
                    s += "\nBut somehow this method still crashed. Maybe these prints will help."
                    raise Exception(s)

    # overwrite traj and return
    out_traj.xyz = new_xyz
    return out_traj


@overload
def _get_near_and_far_networkx(
    bondgraph: nx.Graph,
    edge_indices: np.ndarray,
    top: Optional[md.Topology] = None,
    parallel: bool = True,
) -> tuple[np.ndarray, None]: ...


@overload
def _get_near_and_far_networkx(
    bondgraph: nx.Graph,
    edge_indices: np.ndarray,
    top: Optional[md.Topology] = None,
    parallel: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]: ...


def _get_near_and_far_networkx(
    bondgraph: nx.Graph,
    edge_indices: np.ndarray,
    top: Optional[md.Topology] = None,
    parallel: bool = False,
) -> Union[tuple[list[np.ndarray], list[np.ndarray]], tuple[np.ndarray, None]]:
    """Returns near and far sides for a list of edges giving the indices of the
    two atoms at which the structure is broken.

    Args:
        bondgraph (networkx.classes.graph.Graph): The bondgraph describing the protein.
        edge_indices (np.ndarray): The edges where the graph will be broken at.
        top (Optional[md.Topology]): Used for printing helpful messages in exceptions.
        parallel (bool): Whether to return a dense array and None.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing the following:
            - near_sides (list[np.ndarray]): List of integer arrays giving the near
            - sides. len(near_sides) == len(edge_indices).
            - far_sides (list[np.ndarray]): Same as near sides, but this time the far sides.

    """
    # Third Party Imports
    import networkx as nx
    from networkx.algorithms.components.connected import connected_components

    assert edge_indices.shape[1] == 2, (
        f"Can only take `edge_indices` as a numpy array, with shape[1] = 2, but "
        f"the provided `edge_indices` has shape {edge_indices.shape[1]=}."
    )

    if parallel:
        out = np.zeros(shape=(len(edge_indices), len(bondgraph))).astype(bool)

    near_sides = []
    far_sides = []
    for i, edge in enumerate(edge_indices):
        G = nx.convert_node_labels_to_integers(bondgraph).copy()
        try:
            G.remove_edge(*edge)
        except nx.NetworkXError as e:
            if top:
                raise Exception(
                    f"Seems like the edge {[top.atom(a) for a in edge]=} "
                    f"{[top.atom(a).index for a in edge]=}is not "
                    f"part of the graph. This might originate from a bond, that has "
                    f"been deleted, but which atoms are still considered to be part of "
                    f"a dihedral. Maybe you want supplied this topology to the "
                    f"mdtraj_backmapping method with wrong featurization."
                ) from e
            else:
                raise Exception(
                    f"Please provide arg `top` to learn more about this Exception"
                ) from e
        except TypeError as e:
            raise Exception(
                f"Could not remove the edge {edge=}, {edge_indices.shape=}."
            ) from e
        components = [*connected_components(G)]
        if len(components) != 2:
            if top is None:
                raise Exception(
                    f"Splitting the topology of the trajectory at the edge "
                    f"{edge} does not work. Provide a topology to see, "
                    "which atoms are affected"
                )
            else:
                path = nx.shortest_path(G, *edge)
                assert isinstance(path, list)
                path_atoms = [top.atom(i) for i in path]
                a1 = top.atom(edge[0])
                a2 = top.atom(edge[1])
                raise Exception(
                    f"Splitting at edge {edge} does not work. Here are the "
                    f"atoms: {a1} and {a2}. Removing "
                    f"this edge resulted in {len(components)} disconnected components. "
                    f"The resSeqs of the residues are {a1.residue.resSeq} and "
                    f"{a2.residue.resSeq}. The indices are {a1.residue.index} and "
                    f"{a2.residue.index}. These two atoms are still connected via "
                    f"this path: {path_atoms}."
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
        far = np.asarray(subgraph.nodes)
        subgraph = G.subgraph(components[0]).copy()
        near = np.asarray(subgraph.nodes)
        if parallel:
            out[i][near] = True
        else:
            far_sides.append(far)
            near_sides.append(near)
    if not parallel:
        return near_sides, far_sides
    else:
        return out, None


def _angle(
    xyz: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Returns current angle between positions.

    Adapted from MDTraj.

    Args:
        xyz (np.ndarray). This function only takes a xyz array of a single frame and uses np.expand_dims()
            to make that fame work with the `_displacement` function from mdtraj.
        indices (Union[np.ndarray, list]): List of 3 ints describing the dihedral.

    Returns:
        np.ndarray: The angle.

    """
    indices = np.expand_dims(np.asarray(indices), 0)
    xyz = np.expand_dims(xyz, 0)
    ix01 = indices[:, [1, 0]]
    ix21 = indices[:, [1, 2]]

    u_prime = _displacement(xyz, ix01)
    v_prime = _displacement(xyz, ix21)
    u_norm = np.sqrt((u_prime**2).sum(-1))
    v_norm = np.sqrt((v_prime**2).sum(-1))

    u = u_prime / (u_norm[..., np.newaxis])
    v = v_prime / (v_norm[..., np.newaxis])

    return np.arccos((u * v).sum(-1))


def _dihedral(
    xyz: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Returns current dihedral angle between positions.

    Adapted from MDTraj.

    Args:
        xyz (np.ndarray). This function only takes a xyz array of a single frame and uses np.expand_dims()
            to make that fame work with the `_displacement` function from mdtraj.
        indices (Union[np.ndarray, list]): List of 4 ints describing the dihedral.

    Returns:
        np.ndarray: The dihedral.

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


def _displacement(xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Displacement vector between pairs of points in each frame

    Args:
        xyz (np.ndarray): The coordinates of the atoms.
        pairs (np.ndarray): An array with integers and shape (n_pairs, 2),
            defining the atom paris between which the displacement will
            be calculated.

    Returns:
        np.ndarray: An array with shape (n_pairs, ).

    """
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
