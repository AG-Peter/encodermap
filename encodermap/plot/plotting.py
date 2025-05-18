# -*- coding: utf-8 -*-
# encodermap/plot/plotting.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
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
"""Convenience functions for Plotting.

"""

##############################################################################
# Imports
##############################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Literal, Optional, Union, overload

# Third Party Imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets

# Encodermap imports
from encodermap.encodermap_tf1.misc import periodic_distance_np, sigmoid
from encodermap.misc.rotate import _dihedral
from encodermap.parameters.parameters import AnyParameters
from encodermap.trajinfo.info_all import TrajEnsemble
from encodermap.trajinfo.info_single import SingleTraj


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    import plotly.express as px
    import plotly.graph_objs as go


################################################################################
# Optional Imports
################################################################################


# Third Party Imports
from optional_imports import _optional_import


md = _optional_import("mdtraj")
nv = _optional_import("nglview")
mda = _optional_import("MDAnalysis")
pd = _optional_import("pandas")
go = _optional_import("plotly", "graph_objects")
px = _optional_import("plotly", "express")
make_subplots = _optional_import("plotly", "subplots.make_subplots")


################################################################################
# Globals
################################################################################


__all__: list[str] = [
    "distance_histogram",
    "distance_histogram_interactive",
    "plot_raw_data",
    "interactive_path_visualization",
    "plot_ramachandran",
    "plot_dssp",
    "plot_end2end",
    "plot_ball_and_stick",
    "plot_trajs_by_parameter",
    "plot_free_energy",
    "animate_lowd_trajectory",
]


GLOBAL_LAYOUT = {}


################################################################################
# Utilities
################################################################################


@overload
def get_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    weights: Optional[np.ndarray],
    avoid_zero_count: bool,
    transpose: bool,
    return_edges: Literal[False],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def get_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    weights: Optional[np.ndarray],
    avoid_zero_count: bool,
    transpose: bool,
    return_edges: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def get_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 100,
    weights: Optional[np.ndarray] = None,
    avoid_zero_count: bool = False,
    transpose: bool = False,
    return_edges: bool = False,
) -> Union[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Construct a 2D histogram.

    Args:
        x (np.ndarray): The x coordinates of the data.
        y (np.ndarray): The y coordinates of the data.
        bins (int): The number of bins passed to np.histogram2d.
        weights (np.ndarray): The weights passed to np.histogram2d.
        avoid_zero_count (bool): Avoid zero counts by lifting all
            histogram elements to the minimum value before computing the free
            energy. If False, zero histogram counts would yield infinity
            in the free energy.
        transpose (bool): Whether to transpose the output.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            xcenters, ycenters, and the histogram.

    Examples:
        >>> from encodermap.plot.plotting import get_histogram
        >>> x, y = np.random.uniform(size=(2, 500))
        >>> xcenters, ycenters, H = get_histogram(x, y)
        >>> xcenters.shape
        (100,)
        >>> H.shape
        (100, 100)
        >>> np.min(H)
        0.0
        >>> xcenters, ycenters, H = get_histogram(x, y, avoid_zero_count=True)
        >>> np.min(H)
        1.0

    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
    xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
    ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
    if avoid_zero_count:
        H = np.maximum(H, np.min(H[H.nonzero()]))
    if transpose:
        H = H.T
    if not return_edges:
        return xcenters, ycenters, H
    else:
        return xcenters, ycenters, xedges, yedges, H


def get_density(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 100,
    weights: Optional[np.ndarray] = None,
    avoid_zero_count: bool = False,
    transpose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a 2D histogram with density.

    Args:
        x (np.ndarray): The x coordinates of the data.
        y (np.ndarray): The y coordinates of the data.
        bins (int): The number of bins passed to np.histogram2d.
        weights (np.ndarray): The weights passed to np.histogram2d.
        avoid_zero_count (bool): Avoid zero counts by lifting all
            histogram elements to the minimum value before computing the free
            energy. If False, zero histogram counts would yield infinity
            in the free energy.
        transpose (bool): Whether to transpose the output.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            xcenters, ycenters, and the histogram.

    """
    xcenters, ycenters, H = get_histogram(
        x, y, bins, weights, avoid_zero_count, transpose
    )
    return xcenters, ycenters, to_density(H)


def to_density(H: np.ndarray) -> np.ndarray:
    """Normalize histogram counts.

    Args:
        H (np.ndarray): The histogram to normalize.

    Returns:
        np.ndarray: The normalized histogram.

    """
    return H / float(H.sum())


def to_free_energy(
    H: np.ndarray,
    kT: float = 1.0,
    minener_zero: bool = False,
):
    """Compute free energies from histogram counts.

    Args:
        H (np.ndarray): The density histogram to get the free energy from.
        kT (float): The value of kT in the desired energy unit. By default,
            energies are computed in kT (setting 1.0). If you want to
            measure the energy in kJ/mol at 298 K, use kT=2.479 and
            change the cbar_label accordingly. Defaults to 1.0.
        minener_zero (bool): Shifts the energy minimum to zero. Defaults to False.

    Returns:
        np.ndarray: The free energy values in units of kT.

    """
    F = np.inf * np.ones(shape=H.shape)
    nonzero = H.nonzero()
    F[nonzero] = -np.log(H[nonzero])
    if minener_zero:
        F[nonzero] -= np.min(F[nonzero])
    F = F * kT
    return F


def get_free_energy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 100,
    weights: Optional[np.ndarray] = None,
    kT: float = 1.0,
    avoid_zero_count: bool = False,
    minener_zero: bool = False,
    transpose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a 2D histogram with free energy.

    Args:
        x (np.ndarray): The x coordinates of the data.
        y (np.ndarray): The y coordinates of the data.
        bins (int): The number of bins passed to np.histogram2d.
        weights (np.ndarray): The weights passed to np.histogram2d.
        avoid_zero_count (bool): Avoid zero counts by lifting all
            histogram elements to the minimum value before computing the free
            energy. If False, zero histogram counts would yield infinity
            in the free energy.
        kT (float): The value of kT in the desired energy unit. By default,
            energies are computed in kT (setting 1.0). If you want to
            measure the energy in kJ/mol at 298 K, use kT=2.479 and
            change the cbar_label accordingly. Defaults to 1.0.
        minener_zero (bool): Shifts the energy minimum to zero. Defaults to False.
        transpose (bool): Whether to transpose the output.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            xcenters, ycenters, and the histogram.

    """
    xcenters, ycenters, H = get_density(
        x, y, bins, weights, avoid_zero_count, transpose
    )

    # to free energy
    H = to_free_energy(H, kT, minener_zero)

    return xcenters, ycenters, H


def hex_to_rgba(h, alpha=0.8):
    h = h.lstrip("#")
    r, g, b = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


################################################################################
# Private Functions
################################################################################


# @functools.cache
def _get_squiggly_arrow(n: int = 1, n_frames: int = 200) -> pd.DataFrame:
    if n == 1:
        x = np.linspace(0.2, 2.5, n_frames)
        y = np.sin(x * 2) / 0.5
        xy = np.stack([x, y]).T
        positions = np.full((n_frames, n_frames, 2), fill_value=np.nan)
        time = []
        for i, row in enumerate(xy):
            positions[i:, i] = row
            time.append(np.full(shape=(n_frames,), fill_value=i))
        time = np.concatenate(time)
        positions = positions.reshape(-1, 2)
        assert len(time) == len(positions)
        df = pd.DataFrame({"time": time, "x": positions[:, 0], "y": positions[:, 1]})
        return df
    else:
        df = pd.DataFrame({"time": [], "x": [], "y": [], "trajectory": []})
        for i in range(n):
            theta = np.random.random() * 2 * np.pi - np.pi
            rotmat = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )
            x = np.linspace(0.0, 2.5, n_frames)
            y = np.sin(x * 2) / 4
            x -= 1.25
            xy = rotmat @ np.stack([x, y])
            xy[0] += np.random.random((1,))[0]
            xy[1] += np.random.random((1,))[0]
            xy = xy.T
            positions = np.full((n_frames, n_frames, 2), fill_value=np.nan)
            time = []
            for j, row in enumerate(xy):
                positions[j:, j] = row
                time.append(np.full(shape=(n_frames,), fill_value=j))
            time = np.concatenate(time)
            positions = positions.reshape(-1, 2)
            assert len(time) == len(positions)
            sub_df = pd.DataFrame(
                {
                    "time": time,
                    "x": positions[:, 0],
                    "y": positions[:, 1],
                    "trajectory": np.full((len(time),), fill_value=str(i + 1)),
                }
            )
            df = pd.concat([df, sub_df])
        return df


def _project_onto_plane(x: np.ndarray, n: np.ndarray) -> np.ndarray:
    assert np.isclose(np.linalg.norm(n), 1)
    d = np.dot(x, n)
    p = d * n
    return x - p


def _angle_arc(
    points: np.ndarray,
    name: str,
    value: float,
    radius: float = 0.05,
    n_points: int = 100,
) -> go.Scatter3d:
    """Creates a `go.Scatetr3d` plot as an arc to represent he dihedral defined by `points`.

    Args:
        points (np.ndarray): The points as a (4, )-shaped numpy array.
        name (str): The name of the angle arc when the mouse is hovered.
        value (float): The value of the dihedral in radians.
        radius (float): The radius of the arc. Defaults to 0.05 nm.
        n_points (int): The number of points used to plot this arc. More
            points might slow donw the system. Defaults to 100.

    Returns:
        go.Scatter3d: The plotly trace.

    """
    center = points[1]
    u = a = points[1] - points[0]
    v = points[2] - points[0]
    face_normal = np.cross(u, v)
    face_normal_unit = face_normal / np.linalg.norm(face_normal)

    u = np.cross(face_normal, a)
    u_unit = u / np.linalg.norm(u)
    a = a / np.linalg.norm(a)
    b = u_unit

    rho = np.linspace(value - np.pi / 4, np.pi, num=n_points)
    hovertemplate = "%{meta[0]:.2f} deg"
    meta = [np.rad2deg(value)]
    out = (
        center
        + radius * a * np.cos(rho)[:, np.newaxis]
        + radius * b * np.sin(rho)[:, np.newaxis]
    )
    return go.Scatter3d(
        x=out[:, 0],
        y=out[:, 1],
        z=out[:, 2],
        name="",
        line={
            "color": "black",
            "width": 5,
            "dash": "dash",
        },
        mode="lines",
        hovertemplate=hovertemplate,
        meta=meta,
    )


def _dihedral_arc(
    points: np.ndarray,
    name: str,
    radius: float = 0.05,
    n_points: int = 100,
    initial_points: Literal["random", "select"] = "select",
    true_to_value: bool = True,
) -> go.Scatter3d:
    # get the center
    center = np.mean(points[1:3], axis=0)
    face_normal = points[2] - points[1]
    face_normal_unit = face_normal / np.linalg.norm(face_normal)
    sorted = np.argsort(face_normal)

    if initial_points == "random":
        # first, get a random vector on the plane with normal `face_normal`
        vertical_to_face_normal = np.zeros((3,))
        ind_largest = sorted[-1]
        ind_2nd_largest = sorted[-2]
        vertical_to_face_normal[ind_2nd_largest] = -face_normal[ind_largest]
        vertical_to_face_normal[ind_largest] = face_normal[ind_2nd_largest]
        vertical_to_face_normal_unit = vertical_to_face_normal / np.linalg.norm(
            vertical_to_face_normal
        )
        a = vertical_to_face_normal_unit
        dot = np.dot(face_normal, vertical_to_face_normal)
        assert np.isclose(dot, 0, atol=1e-3)

        # then get the crossproduct
        u = np.cross(face_normal, vertical_to_face_normal)
        u_unit = u / np.linalg.norm(u)
        b = u_unit
        hovertemplate = "%{meta[0]}"
        meta = [name]
    elif initial_points == "select":
        a = points[0] - points[1]
        c = points[3] - points[2]
        a = _project_onto_plane(a, face_normal_unit)

        u = np.cross(face_normal, a)
        u_unit = u / np.linalg.norm(u)
        a = a / np.linalg.norm(a)
        b = u_unit

        dihedral_value = _dihedral(points, [0, 1, 2, 3])[0, 0]
        if true_to_value:
            if dihedral_value >= 0:
                rho = np.linspace(
                    0,
                    dihedral_value,
                    num=n_points,
                )
            else:
                rho = np.linspace(
                    dihedral_value,
                    0,
                    num=n_points,
                )
        else:
            rho = np.linspace(
                0,
                np.pi,
                num=n_points,
            )
        hovertemplate = "%{meta[0]} %{meta[1]:.2f} deg"
        meta = [name.split()[1], np.rad2deg(dihedral_value)]
    else:
        raise ValueError(
            f"Argument `initial_points` must be 'random' or 'select', not {initial_points}."
        )

    out = (
        center
        + radius * a * np.cos(rho)[:, np.newaxis]
        + radius * b * np.sin(rho)[:, np.newaxis]
    )
    return go.Scatter3d(
        x=out[:, 0],
        y=out[:, 1],
        z=out[:, 2],
        name="",
        line={
            "color": "black",
            "width": 5,
            "dash": "dash",
        },
        mode="lines",
        hovertemplate=hovertemplate,
        meta=meta,
    )


def _flatten_coords(traj: "SingleTraj") -> np.ndarray:
    """Flattens coordinates, so it is easier to render them as images."""
    # Third Party Imports
    import networkx as nx
    from mdtraj.geometry.angle import _angle
    from networkx import connected_components
    from transformations import affine_matrix_from_points, rotation_matrix

    # Local Folder Imports
    from ..loading.features import CentralAngles, CentralDihedrals, SideChainDihedrals
    from ..misc.rotate import _get_near_and_far_networkx, mdtraj_rotate

    indices = []
    indices.append(CentralDihedrals(traj).indexes)
    indices.append(SideChainDihedrals(traj).indexes)
    indices = np.vstack(indices)
    angles = np.full((1, indices.shape[0]), 0)
    angles[::2] = 180
    xyz = (
        mdtraj_rotate(
            traj.traj,
            angles=angles,
            indices=indices,
            deg=True,
        )
        .xyz[0]
        .copy()
    )

    # get best surface using 3d least squares
    centroid = xyz.mean(axis=0)
    xyzT = np.transpose(xyz)
    xyzR = xyz - centroid
    xyzRT = np.transpose(xyzR)
    u, sigma, v = np.linalg.svd(xyzRT)
    normal = u[2]
    normal = normal / np.linalg.norm(normal)

    # project points
    a, b, c = normal
    d = -a * centroid[0] - b * centroid[1] - c * centroid[2]
    projected_points = []
    for p in xyz:
        projected_points.append(p - (p.dot(normal) + d / normal.dot(normal)) * normal)
    xyz = np.array(projected_points)

    # fix distances
    edges = []
    edge_lengths = []
    atoms_in_bonds_is = set()
    atoms_in_bonds_should_be = xyz.shape[0]
    for a, b in traj.top.bonds:
        atoms_in_bonds_is.add(a.index)
        atoms_in_bonds_is.add(b.index)
        edges.append([a.index, b.index])
        edge_lengths.append(np.linalg.norm(traj.xyz[0, b.index] - traj.xyz[0, a.index]))
    assert (
        len(atoms_in_bonds_is) == atoms_in_bonds_should_be
    ), f"Can't flatten topology: {traj.top}. There are atoms which are not part of bonds."
    bondgraph = traj.top.to_bondgraph()
    edges = np.asarray(edges)
    edge_lengths = np.asarray(edge_lengths)
    near_and_far_networkx = _get_near_and_far_networkx(
        bondgraph,
        edges,
        traj.top,
        parallel=True,
    )[0]
    for edge, indices, length_should_be in zip(
        edges, near_and_far_networkx, edge_lengths
    ):
        vec = xyz[edge[1]] - xyz[edge[0]]
        length_is = np.linalg.norm(vec)
        vec /= np.linalg.norm(vec)
        trans = vec * length_should_be
        xyz[~indices] += trans

    # fix angles
    angle_should_be = 2 * np.pi / 3
    angle_indices = CentralAngles(traj).angle_indexes
    for i, (a, b, c) in enumerate(angle_indices):
        angle_center = xyz[b]
        ba = a - b
        bc = c - b
        angle_value = np.arccos(
            np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        )
        diff = angle_should_be - angle_value
        G = nx.convert_node_labels_to_integers(bondgraph).copy()
        G.remove_edge(b, c)
        components = [*connected_components(G)]
        if c in components[0] and b in components[1]:
            components = components[::-1]
        subgraph = G.subgraph(components[1]).copy()
        far = np.asarray(subgraph.nodes)
        M = rotation_matrix(
            diff,
            normal,
            angle_center,
        )
        padded = np.pad(
            xyz[far].copy(), ((0, 0), (0, 1)), mode="constant", constant_values=1
        )
        xyz[far] = M.dot(padded.T).T[:, :3]

    data = [
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
        )
    ]
    fig = go.Figure(
        data=data,
        layout={
            "width": 800,
            "height": 800,
        },
    )
    fig.show()

    raise Exception(f"{xyz.shape=}")


def _plot_ball_and_stick(
    traj: Union["SingleTraj", md.Trajectory],
    frame_subsample: Union[int, slice] = slice(None, None, 100),
    highlight: Literal[
        "atoms", "bonds", "angles", "dihedrals", "side_dihedrals", "central_dihedrals"
    ] = "atoms",
    atom_indices: Optional[Sequence[int]] = None,
    custom_colors: Optional[dict[int, str]] = None,
    add_angle_arcs: bool = True,
    angle_arcs_true_to_value: bool = True,
    animation: bool = False,
    persistent_hover: bool = False,
    flatten: bool = False,
) -> go.Figure:  # pragma: no cover
    if hasattr(traj, "copy"):
        traj = traj.copy()
    else:
        traj = SingleTraj(traj)

    if atom_indices is None:
        atom_indices = np.arange(traj.n_atoms)
    if animation:
        raise NotImplementedError(
            f"Animation of ball and stick plot not yet implemented."
        )
    # data for plotting and annotation
    if atom_indices is not None:
        try:
            traj.atom_slice(atom_indices)
        except Exception as e:
            raise Exception(
                f"Can't index {traj=} with {np.asarray(atom_indices)=}"
            ) from e
    if not animation:
        traj = traj[0]
    xyz = traj.xyz[frame_subsample]
    if flatten:
        assert not any([a.element.symbol == "H" for a in traj.top.atoms]), (
            f"Can only create a flattened representation for trajs without hydrogen. "
            f"Use the `atom_indices` argument to remove the hydrogen."
        )
        xyz = _flatten_coords(traj)
    times = traj.time[frame_subsample]
    atom_names = np.array([str(a) for a in traj.top.atoms])
    bonds = [(a.index, b.index) for a, b in traj.top.bonds]
    sizes = np.array(
        [24 if a.element.symbol != "H" else 10 for a in traj.top.atoms]
    ).astype(float)
    if highlight in [
        "bonds",
        "angles",
        "dihedrals",
        "central_dihedrals",
        "side_dihedrals",
    ]:
        sizes /= 1.3
    elements = np.array([a.element.number for a in traj.top.atoms])
    coords = [f"x: {i:.3f}<br>y: {j:.3f}<br>z: {k:.3f}" for i, j, k in xyz[0]]
    assert len(coords) == len(atom_names), f"{len(coords)=} {len(atom_names)=}"
    colormap = {
        1: "rgb(200, 200, 200)",  # hydrogen
        6: "rgb(80, 80, 80)",  # carbon
        7: "rgb(0, 0, 255)",  # nitrogen
        8: "rgb(255, 0, 0)",  # oxygen
        15: "rgb(160, 32, 240)",  # phosphorus
        16: "rgb(255, 255, 0)",  # sulfur
        34: "rgb(170, 74, 68)",  # selenium
    }
    if custom_colors is None:
        color = []
        for i in elements:
            if i in colormap:
                color.append(colormap[i])
            else:
                color.append("rgb(255, 0, 124)")
    else:
        color = np.full(shape=(len(elements),), fill_value="rgb(126, 126, 126)")
        for atom, c in custom_colors.items():
            color[atom] = c

    # for circle arcs
    circles = []

    # set customdata and hovertemplate
    if highlight == "atoms":
        customdata = np.stack(
            (
                atom_names,
                coords,
            ),
            axis=-1,
        )
        hovertemplate = "%{customdata[0]}:<br>%{customdata[1]}"
        hoverinfo = None
    elif highlight == "angles":
        # Local Folder Imports
        from ..loading.features import CentralAngles, SideChainAngles

        x_centers = []
        y_centers = []
        z_centers = []
        angle_names = []
        annotations_text = None

        # Central Angles
        f = CentralAngles(traj=traj)
        for p, name, value in zip(f.indexes, f.describe(), f.transform()[0]):
            x_centers.append(xyz[0, p[1], 0]),
            y_centers.append(xyz[0, p[1], 1]),
            z_centers.append(xyz[0, p[1], 2])
            angle_names.append(
                f"Angle: {traj.top.atom(p[0])} - {traj.top.atom(p[1])} - {traj.top.atom(p[2])}"
            )
            if add_angle_arcs:
                circles.append(
                    _angle_arc(
                        xyz[0, p],
                        name=name,
                        value=value,
                    )
                )

        # Sidechain Angles
        f = SideChainAngles(traj=traj)
        for p, name, value in zip(f.indexes, f.describe(), f.transform()[0]):
            x_centers.append(xyz[0, p[1], 0]),
            y_centers.append(xyz[0, p[1], 1]),
            z_centers.append(xyz[0, p[1], 2])
            angle_names.append(
                f"Angle: {traj.top.atom(p[0])} - {traj.top.atom(p[1])} - {traj.top.atom(p[2])}"
            )
            if add_angle_arcs:
                circles.append(
                    _angle_arc(
                        xyz[0, p],
                        name=name,
                        value=value,
                    )
                )

        customdata = None
        hovertemplate = None
        hoverinfo = "skip"
        center_customdata = angle_names
        center_hovertemplate = "%{customdata}"

    elif highlight == "bonds":
        # Local Folder Imports
        from ..loading.features import AllBondDistances

        f = AllBondDistances(traj=traj)

        x_centers = []
        y_centers = []
        z_centers = []
        bond_names = []
        annotations_text = []

        for p, name in zip(f.indexes, f.describe()):
            x_centers.append(np.mean(xyz[0, p, 0])),
            y_centers.append(np.mean(xyz[0, p, 1])),
            z_centers.append(np.mean(xyz[0, p, 2]))
            bond_names.append(
                f"Bond between {traj.top.atom(p[0])} and {traj.top.atom(p[1])}"
            )
            annotations_text.append(f"{traj.top.atom(p[0])} - {traj.top.atom(p[1])}")

        customdata = None
        hovertemplate = None
        hoverinfo = "skip"
        center_customdata = bond_names
        center_hovertemplate = "%{customdata}"
    elif highlight in ["dihedrals", "side_dihedrals", "central_dihedrals"]:
        # Local Folder Imports
        from ..loading.features import CentralDihedrals, SideChainDihedrals

        x_centers = []
        y_centers = []
        z_centers = []
        dihedral_names = []
        annotations_text = []

        # Central Dihedrals
        if highlight in ["dihedrals", "central_dihedrals"]:
            f = CentralDihedrals(traj=traj)
            for p, name in zip(f.indexes, f.describe()):
                x_centers.append(np.mean(xyz[0, p[1:3], 0])),
                y_centers.append(np.mean(xyz[0, p[1:3], 1])),
                z_centers.append(np.mean(xyz[0, p[1:3], 2]))
                dihedral_names.append(name)
                annotations_text.append(name.split()[1])
                if add_angle_arcs:
                    circles.append(
                        _dihedral_arc(
                            xyz[0, p], name=name, true_to_value=angle_arcs_true_to_value
                        )
                    )

        # Sidechain Dihedrals
        if highlight in ["dihedrals", "side_dihedrals"]:
            f = SideChainDihedrals(traj=traj)
            for p, name in zip(f.indexes, f.describe()):
                x_centers.append(np.mean(xyz[0, p[1:3], 0])),
                y_centers.append(np.mean(xyz[0, p[1:3], 1])),
                z_centers.append(np.mean(xyz[0, p[1:3], 2]))
                dihedral_names.append(name)
                annotations_text.append(name.split()[1])
                if add_angle_arcs:
                    circles.append(
                        _dihedral_arc(
                            xyz[0, p], name=name, true_to_value=angle_arcs_true_to_value
                        )
                    )

        customdata = None
        hovertemplate = None
        hoverinfo = "skip"
        center_customdata = dihedral_names
        center_hovertemplate = "%{customdata}"
    else:
        raise TypeError(
            f"The argument `highlight` must be one of the following: "
            f"'atoms', 'bonds', 'angles', 'dihedrals'. You supplied {highlight}"
        )

    # create scatter trace
    scatter = go.Scatter3d(
        x=xyz[0, :, 0],
        y=xyz[0, :, 1],
        z=xyz[0, :, 2],
        customdata=customdata,
        mode="markers",
        hovertemplate=hovertemplate,
        name="Atoms",
        marker=dict(
            size=sizes,
            color=color,
            opacity=1.0,
        ),
        hoverinfo=hoverinfo,
    )

    # create line trace
    x_lines = []
    y_lines = []
    z_lines = []
    for p in bonds:
        for i in range(2):
            x_lines.append(xyz[0, p[i], 0])
            y_lines.append(xyz[0, p[i], 1])
            z_lines.append(xyz[0, p[i], 2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    lines = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        name="",
        line=dict(
            color="black",
            width=(
                6
                if highlight
                in ["bonds", "dihedrals", "central_dihedrals", "side_dihedrals"]
                else 1
            ),
        ),
        hoverinfo="skip",
    )

    # create figure
    if highlight == "atoms":
        data = [scatter, lines]
    else:
        centers = go.Scatter3d(
            x=x_centers,
            y=y_centers,
            z=z_centers,
            mode="markers",
            marker=dict(
                size=30,
                color="rgba(0, 0, 0, 0)",
                opacity=0.0,
            ),
            name=f"{highlight}".capitalize(),
            customdata=center_customdata,
            hovertemplate=center_hovertemplate,
        )
        data = [centers, scatter, lines]

    if highlight in ["dihedrals", "angles", "central_dihderals", "side_dihedrals"]:
        data.extend(circles)

    fig = go.Figure(
        data=data,
    )

    if persistent_hover:
        annotations = []
        if highlight == "atoms":
            zipped = zip(xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2], traj.top.atoms)
            for x, y, z, a in zipped:
                if a.element.symbol == "H":
                    continue
                annotations.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "text": str(a),
                    }
                )

        else:
            zipped = zip(x_centers, y_centers, z_centers, annotations_text)
            for x, y, z, text in zipped:
                annotations.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "text": text,
                    }
                )
    else:
        annotations = []

    scene = {
        "xaxis_gridcolor": "rgb(102, 102, 102)",
        "yaxis_gridcolor": "rgb(102, 102, 102)",
        "zaxis_gridcolor": "rgb(102, 102, 102)",
        "annotations": annotations,
    }
    if "scene" in GLOBAL_LAYOUT:
        scene |= GLOBAL_LAYOUT["scene"]
        global_layout = {k: v for k, v in GLOBAL_LAYOUT.items() if k != "scene"}
    else:
        global_layout = GLOBAL_LAYOUT.copy()

    fig.update_layout(
        height=900,
        width=900,
        showlegend=False,
        **global_layout,
        scene=scene,
    )

    # create frames
    if animation:
        frames = [go.Frame(data=[scatter, lines])]
        for points in xyz:
            x_lines = []
            y_lines = []
            z_lines = []
            for p in bonds:
                for i in range(2):
                    x_lines.append(points[p[i], 0])
                    y_lines.append(points[p[i], 1])
                    z_lines.append(points[p[i], 2])
                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        customdata=atom_names,
                        mode="markers",
                        hovertemplate="%{customdata}",
                        name="",
                        marker=dict(
                            size=sizes,
                            color=color,
                            opacity=1.0,
                        ),
                    ),
                    go.Scatter3d(
                        x=x_lines,
                        y=y_lines,
                        z=z_lines,
                        mode="lines",
                        name="",
                        line=dict(
                            color="black",
                        ),
                    ),
                ],
            )
            frames.append(frame)
        fig.update(frames=frames)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        redraw=True, fromcurrent=True, mode="immediate"
                                    )
                                ),
                            ],
                        )
                    ],
                )
            ],
            # sliders=(
            #     [
            #         {
            #             "steps": [
            #                 {
            #                     "args": [
            #                         [f.name],
            #                         {
            #                             "frame": {"duration": 0, "redraw": True},
            #                             "mode": "immediate",
            #                         },
            #                     ],
            #                     "label": f.name,
            #                     "method": "animate",
            #                 }
            #                 for f in frames
            #             ],
            #         }
            #     ],
            # ),
            scene=dict(
                xaxis=dict(range=[np.min(xyz[..., 0]), np.max(xyz[..., 0])]),
                yaxis=dict(range=[np.min(xyz[..., 1]), np.max(xyz[..., 1])]),
                zaxis=dict(range=[np.min(xyz[..., 2]), np.max(xyz[..., 2])]),
            ),
        )
    return fig


################################################################################
# Plotting Functions
################################################################################


def animate_lowd_trajectory(
    n: int = 1,
    potential: bool = False,
    n_frames: int = 200,
) -> None:
    if not potential:
        p_init = np.random.random((1, 2)) * 10
        p = p_init.copy()
        v_init = np.random.random((1, 2)) * 0.05 - 0.025
        positions = np.full((n_frames, n_frames, 2), np.nan)
        time = []
        for i in range(n_frames):
            positions[i:, i] = p.copy()
            p += v_init
            time.append(np.full(shape=(n_frames,), fill_value=i))
        time = np.concatenate(time)
        x_min = np.nanmin(positions[..., 0]) - 1
        x_max = np.nanmax(positions[..., 0]) + 1
        y_min = np.nanmin(positions[..., 1]) - 1
        y_max = np.nanmax(positions[..., 1]) + 1
        positions = positions.reshape(-1, 2)
        assert len(time) == len(positions)
        df = pd.DataFrame({"time": time, "x": positions[:, 0], "y": positions[:, 1]})
        fig = px.line(
            data_frame=df,
            x="x",
            y="y",
            animation_frame="time",
            range_x=(x_min, x_max),
            range_y=(y_min, y_max),
            height=500,
            width=800,
        )
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
        fig.show()
    else:
        if potential is True:
            if n == 1:
                df = _get_squiggly_arrow(n_frames=n_frames)
                x_min = np.nanmin(df["x"].values) - 1
                x_max = np.nanmax(df["x"].values) + 1
                y_min = np.nanmin(df["y"].values) - 1
                y_max = np.nanmax(df["y"].values) + 1
                fig = px.line(
                    data_frame=df,
                    x="x",
                    y="y",
                    animation_frame="time",
                    range_x=(x_min, x_max),
                    range_y=(y_min, y_max),
                    height=500,
                    width=800,
                )
                fig.layout.updatemenus[0].buttons[0].args[1]["transition"][
                    "duration"
                ] = 0
                fig.show()
            else:
                df = _get_squiggly_arrow(n=n, n_frames=n_frames)
                x_min = np.nanmin(df["x"].values) - 1
                x_max = np.nanmax(df["x"].values) + 1
                y_min = np.nanmin(df["y"].values) - 1
                y_max = np.nanmax(df["y"].values) + 1
                fig = px.line(
                    data_frame=df,
                    x="x",
                    y="y",
                    animation_frame="time",
                    color="trajectory",
                    range_x=(x_min, x_max),
                    range_y=(y_min, y_max),
                    height=500,
                    width=800,
                )
                fig.layout.updatemenus[0].buttons[0].args[1]["transition"][
                    "duration"
                ] = 0
                fig.show()
        else:
            print(f"{potential=}")


def plot_trajs_by_parameter(
    trajs: Union[SingleTraj, TrajEnsemble],
    parameter: Union[
        Literal[
            "common_str",
            "frame",
            "encoded_frame",
            "traj_num",
            "topology",
            "free_energy",
        ],
        str,
    ] = "common_str",
    type: Literal["scatter", "heatmap"] = "scatter",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    contourtype: Literal["contour", "contourf"] = "countour",
    col: str = "lowd",
    nbins: int = 100,
    alpha: float = 0.8,
    z_name_overwrite: str = "",
    show: bool = True,
    cbar: bool = True,
) -> go.Figure:
    if x is None:
        assert y is None, "Must provide either x and y or both None."
        assert col in trajs._CVs, (
            f"The CV `col`={col} cannot be found in the `trajs` with CVs: "
            f"{list(trajs.CVs.keys())}. Please use `load_CVs` to load the "
            f"low-dimensional coordinates for the `trajs`."
        )
        x, y = trajs.CVs[col].T

    if (type == "scatter" and x.size > 25_000) and not os.getenv(
        "ENCODERMAP_SKIP_SCATTER_SIZE_CHECK", "False"
    ) == "True":
        print(
            f"The number of points is very large ({x.size}). Using scatter "
            f"with this number of points might crash your browser and maybe "
            f"even your system. Set the environment variable "
            f"'ENCODERMAP_SKIP_SCATTER_SIZE_CHECK' to 'True' to skip this check"
        )
        return

    data = None
    if parameter == "common_str":
        data = []
        for traj in trajs:
            data.extend([traj.common_str for i in range(traj.n_frames)])
    elif parameter == "free_energy":
        fig = go.Figure(
            data=[_plot_free_energy(*trajs.lowd.T, cbar=cbar)],
            layout={
                "autosize": True,
                "height": 800,
                "width": 800,
                "title": "Free Energy",
                "xaxis_title": "x in a.u.",
                "yaxis_title": "y in a.u.",
            },
        )
        if show:
            fig.show()
        return fig
    elif parameter == "encoded_frame":
        # Encodermap imports
        from encodermap.loading.features import pair

        type = "scatter"
        data = []
        for traj in trajs:
            data.extend([pair(traj.traj_num, i) for i in range(traj.n_frames)])
    elif parameter == "traj_num":
        data = []
        for traj in trajs:
            data.extend([traj.traj_num for i in range(traj.n_frames)])
    else:
        if parameter in trajs.CVs:
            if (_ := trajs.CVs[parameter]).ndim == 1:
                data = _

    if data is None:
        raise Exception(
            f"Argument `parameter` must be one of 'common_str', 'frame', "
            f"'encoded_frame', 'traj_num', 'topology', 'free_energy', or any "
            f"of the `TrajEnsemble` 1-dimensional CVs."
            f"You provided {parameter}."
        )

    # this is the same no matter what datasource we use
    df = pd.DataFrame({"x": x, "y": y, "data": data})
    if z_name_overwrite:
        parameter = z_name_overwrite
    title = parameter.replace("_", " ").title()
    title = (
        f"{title} for Trajectories with {trajs.n_frames} frames, "
        f"{trajs.n_trajs} trajs and {len(trajs.top)} uniques topologies."
    )
    if type == "scatter":
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="data",
            color_continuous_scale="Viridis",
            render_mode="webgl",
            labels={
                "x": "x in a.u.",
                "y": "y in a.u.",
                "data": parameter,
            },
            opacity=alpha,
        )
        if not cbar:
            fig.update_coloraxes(showscale=False)
        fig.update_layout(
            {
                "autosize": True,
                "height": 800,
                "width": 800,
                "title": title,
            }
            | GLOBAL_LAYOUT
        )
    elif type == "heatmap":
        if len(np.unique(df["data"])) > 10:
            colors = px.colors.qualitative.Alphabet
        else:
            colors = px.colors.qualitative.Plotly

        traces = []
        bins = [
            np.linspace(np.min(df["x"]), np.max(df["x"]), nbins + 1, endpoint=True),
            np.linspace(np.min(df["y"]), np.max(df["y"]), nbins + 1, endpoint=True),
        ]
        xcenters = np.mean(np.vstack([bins[0][:-1], bins[0][1:]]), axis=0)
        ycenters = np.mean(np.vstack([bins[1][:-1], bins[1][1:]]), axis=0)
        for i, (datapoint, sub_df) in enumerate(df.groupby(data)):
            color = colors[i]
            H, _, __ = np.histogram2d(*sub_df[["x", "y"]].values.T, bins=bins)
            traces.append(
                go.Contour(
                    x=xcenters,
                    y=ycenters,
                    z=H.T,
                    contours_type="constraint",
                    contours_operation="<",
                    contours_value=0,
                    contours_coloring="none",
                    fillcolor=hex_to_rgba(color, alpha=alpha),
                    line_color=color,
                    name=datapoint,
                    visible=True,
                ),
            )
            # if contourtype == "contourf":
            #     H = H.astype(bool).astype(float)
            #     H[H == 0] = np.nan
            #     traces.append(
            #         go.Contour(
            #             x=xcenters,
            #             y=ycenters,
            #             z=H.T,
            #             colorscale=[[0, hex_to_rgba(color, alpha=alpha)], [1, "rgba(0, 0, 0, 0)"]],
            #             showscale=False,
            #         ),
            #     )

        fig = go.Figure(
            data=traces,
            layout={
                "autosize": True,
                "width": 800,
                "height": 800,
                "title": title,
            }
            | GLOBAL_LAYOUT,
        )
    else:
        raise Exception(
            f"Argument `type` must be either 'scatter' or 'heatmap'. You provided {type}."
        )
    if show:
        fig.show()
    return fig


def _plot_free_energy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 100,
    weights: Optional[np.ndarray] = None,
    kT: float = 1.0,
    avoid_zero_count: bool = False,
    minener_zero: bool = True,
    transpose: bool = True,
    cbar: bool = False,
    cbar_label: str = "free energy / kT",
    colorbar_x: Optional[float] = None,
) -> go.Contour:
    """Plots free energy using plotly.

    Args:
        x (np.ndarray): The x coordinates of the data.
        y (np.ndarray): The y coordinates of the data.
        bins (int): The number of bins passed to np.histogram2d.
        weights (np.ndarray): The weights passed to np.histogram2d.
        avoid_zero_count (bool): Avoid zero counts by lifting all
            histogram elements to the minimum value before computing the free
            energy. If False, zero histogram counts would yield infinity
            in the free energy.
        kT (float): The value of kT in the desired energy unit. By default,
            energies are computed in kT (setting 1.0). If you want to
            measure the energy in kJ/mol at 298 K, use kT=2.479 and
            change the cbar_label accordingly. Defaults to 1.0.
        minener_zero (bool): Shifts the energy minimum to zero. Defaults to False.
        transpose (bool): Whether to transpose the output.
        cbar (bool): Whether to display a colorbar. Dewfaults to False.
        cbar_label (str): The label of the colorbar. Defaults to 'free energy / kT'.
        colorbar_x (Optional[float]): Sets the x position with respect to xref
            of the color bar (in plot fraction). When xref is “paper”, None becomes
            1.02 when orientation is “v” and 0.5 when orientation is “h”. When
            xref is “container”, None becaomses 1 when orientation is “v” and
            0.5 when orientation is “h”. Must be between
            0 and 1 if xref is “container” and between “-2” and 3 if xref is
            “paper”.

    Returns:
        go.Contour: The contour plot.

    Examples:
        >>> import plotly.graph_objects as go
        >>> from encodermap.plot.plotting import _plot_free_energy
        ...
        >>> x, y = np.random.normal(size=(2, 1000))
        >>> trace = _plot_free_energy(x, y, bins=10)
        >>> fig = go.Figure(data=[trace])
        >>> np.any(fig.data[0].z == float("inf"))
        True

    """
    X, Y, Z = get_free_energy(
        x=x,
        y=y,
        weights=weights,
        bins=bins,
        kT=kT,
        avoid_zero_count=avoid_zero_count,
        minener_zero=minener_zero,
        transpose=transpose,
    )
    trace = go.Contour(
        x=X,
        y=Y,
        z=Z,
        name="Lowd projection",
        showscale=cbar,
        hoverinfo="none",
        colorscale="Viridis",
        colorbar_title=cbar_label,
        # histfunc="count",
        colorbar_x=colorbar_x,
    )
    return trace


def plot_free_energy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 100,
    weights: Optional[np.ndarray] = None,
    kT: float = 1.0,
    avoid_zero_count: bool = False,
    minener_zero: bool = True,
    transpose: bool = True,
    cbar: bool = False,
    cbar_label: str = "free energy / kT",
    colorbar_x: Optional[float] = None,
) -> None:
    """Plots free energy using plotly.

    Args:
        x (np.ndarray): The x coordinates of the data.
        y (np.ndarray): The y coordinates of the data.
        bins (int): The number of bins passed to np.histogram2d.
        weights (np.ndarray): The weights passed to np.histogram2d.
        avoid_zero_count (bool): Avoid zero counts by lifting all
            histogram elements to the minimum value before computing the free
            energy. If False, zero histogram counts would yield infinity
            in the free energy.
        kT (float): The value of kT in the desired energy unit. By default,
            energies are computed in kT (setting 1.0). If you want to
            measure the energy in kJ/mol at 298 K, use kT=2.479 and
            change the cbar_label accordingly. Defaults to 1.0.
        minener_zero (bool): Shifts the energy minimum to zero. Defaults to False.
        transpose (bool): Whether to transpose the output.
        cbar (bool): Whether to display a colorbar. Dewfaults to False.
        cbar_label (str): The label of the colorbar. Defaults to 'free energy / kT'.
        colorbar_x (Optional[float]): Sets the x position with respect to xref
            of the color bar (in plot fraction). When xref is “paper”, None becomes
            1.02 when orientation is “v” and 0.5 when orientation is “h”. When
            xref is “container”, None becaomses 1 when orientation is “v” and
            0.5 when orientation is “h”. Must be between
            0 and 1 if xref is “container” and between “-2” and 3 if xref is
            “paper”.

    """
    fig = go.Figure(
        data=[
            _plot_free_energy(
                x=x,
                y=y,
                bins=bins,
                weights=weights,
                kT=kT,
                avoid_zero_count=avoid_zero_count,
                minener_zero=minener_zero,
                transpose=transpose,
                cbar=cbar,
                cbar_label=cbar_label,
                colorbar_x=colorbar_x,
            ),
        ],
        layout={
            "width": 500,
            "height": 500,
        }
        | GLOBAL_LAYOUT,
    )
    fig.show()


def interactive_path_visualization(
    traj: SingleTraj,
    lowd: Union[np.ndarray, pd.DataFrame],
    path: np.ndarray,
) -> widgets.GridBox:
    assert len(traj) == len(
        path
    ), f"Path has {len(path)} points, Trajectory has {len(traj)} frames."

    # define the traces
    if isinstance(lowd, pd.DataFrame):
        lowd = lowd[["x", "y"]].values
    trace1 = _plot_free_energy(*lowd.T, transpose=True)
    trace2 = go.Scatter(
        mode="lines",
        x=path[:, 0],
        y=path[:, 1],
        name="Path",
    )
    trace3 = go.Scatter(
        mode="markers",
        marker={"size": 10},
        x=[path[0, 0]],
        y=[path[0, 1]],
        name="Current path pos.",
    )

    # create a figure widget
    g = go.FigureWidget(
        data=[trace1, trace2, trace3],
        layout=go.Layout(
            {
                "height": 500,
                "width": 500,
                "showlegend": False,
                "margin": {
                    "t": 0,
                    "b": 0,
                    "l": 0,
                    "r": 0,
                },
            }
        ),
    )

    # create the nglview widget
    nglview = nv.show_mdtraj(traj.traj)

    # create the media slider
    media_widget = widgets.Play(
        value=0,
        min=0,
        max=len(path),
        step=1,
        disabled=False,
    )
    media_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(path),
    )
    widgets.jslink((media_widget, "value"), (media_slider, "value"))

    box1 = widgets.Box(
        children=[g],
        layout=widgets.Layout(
            width="auto",
            height="auto",
            grid_area="main",
        ),
        style=widgets.Style(
            margin="0 0 0 0",
            pad="0 0 0 0",
        ),
    )

    box2 = widgets.Box(
        children=[nglview],
        layout=widgets.Layout(
            width="auto",
            height="auto",
            grid_area="sidebar",
        ),
        style=widgets.Style(
            margin="0 0 0 0",
            pad="0 0 0 0",
        ),
    )

    box3 = widgets.HBox(
        children=[media_widget, media_slider],
        layout=widgets.Layout(
            width="auto",
            height="auto",
            grid_area="footer",
            align_content="center",
        ),
        style=widgets.Style(
            margin="0 0 0 0",
            pad="0 0 0 0",
        ),
    )

    container = widgets.GridBox(
        children=[
            box1,
            box2,
            box3,
        ],
        layout=widgets.Layout(
            width="100%",
            grid_template_columns="auto auto",
            grid_template_rows="1000 px",
            grid_gap="5px",
            grid_template_areas="""
            "main sidebar sidebar sidebar"
            "footer footer footer footer"
            """,
        ),
    )

    def advance_path(n: int) -> None:
        n = n["new"]
        print(n)
        nglview.frame = n
        g.data[2].x = [path[n, 0]]
        g.data[2].y = [path[n, 1]]

    media_slider.observe(advance_path, names="value")

    return container


def distance_histogram_interactive(
    data: Union[np.ndarray, pd.DataFrame],
    periodicity: float,
    low_d_max: float = 5.0,
    n_values: int = 1000,
    bins: Union[Literal["auto"], int] = "auto",
    initial_guess: Optional[tuple[float, ...]] = None,
    renderer: Optional[Literal["colab", "plotly_mimetype+notebook"]] = None,
    parameters: Optional["AnyParameters"] = None,
) -> None:  # pragma: no cover
    """Interactive version of `distance_histogram`.

    Note:

    Args:
        data (np.ndarray): 2-dimensional numpy array. Columns should iterate
            over the dimensions of the datapoints, i.e. the dimensionality
            of the data. The rows should iterate over datapoints.
        periodicity (float): Periodicity of the data. Use `float("inf")`
            for non-periodic data.
        low_d_max (float): Upper limit for plotting the low_d sigmoid.
            Defaults to 5.0.
        n_values (int): The number of x-values to use for the  plotting
            of the sigmoid functions. Used in `np.linspace(min, max, n_values)`.
            Defaults to 1000.
        bins (Union[Literal["auto"], int]): Number of bins for histogram.
            Use 'auto' to let numpy decide how many bins to use. Defaults to 'auto'.
        initial_guess (Optional[tuple[float, ...]]): Tuple of sketchmap
            sigmoid parameters n shape (highd_sigma, highd_a, highd_b,
            lowd_sigma, lowd_a, lowd_b). If None is provided, the default
            values: (4.5, 12, 6, 1, 2, 6) are chosen. Defaults to None.
        parameters (AnyParameters): An instance of `encodermap.Parameters`, or
            `encodermap.ADCParameters`, to which the sigmoid parameters will be
            set.
        skip_data_size_check (bool): Whether to skip a check, that prevents the
            kernel to be killed when large datasets are passed.

    """
    # decide the renderer
    if renderer is None:
        try:
            # Third Party Imports
            from google.colab import data_table

            renderer = "colab"
        except (ModuleNotFoundError, NameError):
            renderer = "plotly_mimetype+notebook"

    assert not np.any(np.isnan(data)), "You provided some nans."

    # some helper functions
    def my_ceil(a, precision=0):
        return np.round(a + 0.5 * 10 ** (-precision), precision)

    def sigmoid(r, sigma=1, a=1, b=1):
        return 1 - (1 + (2 ** (a / b) - 1) * (r / sigma) ** a) ** (-b / a)

    def get_connection_traces(highd, lowd, lowd_max, highd_max):
        for i, (h, l) in enumerate(zip(highd, lowd)):
            l_plot = l / lowd_max
            h_plot = h / highd_max
            yield go.Scatter(
                x=[l_plot, h_plot],
                y=[0, 1],
                mode="lines",
                name=f"connection_{i}",
                showlegend=False,
                line_width=0.8,
                line_color="black",
                hovertemplate=f"{h:.2f} in highd maps to {l:.2f} in lowd {lowd_max=}",
            )

    # get the distances while accounting for periodicity
    vecs = periodic_distance_np(
        np.expand_dims(data, axis=1), np.expand_dims(data, axis=0), periodicity
    )
    dists = np.linalg.norm(vecs, axis=2)
    while True:
        try:
            dists = np.linalg.norm(dists, axis=2)
        except np.exceptions.AxisError:
            break
    dists = dists.reshape(-1)
    high_d_max = np.max(dists)

    # use the initial guess or default values
    if initial_guess is None:
        initial_guess = (4.5, 12, 6, 1, 2, 6)

    # instantiate the sliders
    lowd_sigma_slider = widgets.FloatSlider(
        value=initial_guess[3],
        min=0.1,
        max=my_ceil(low_d_max, 1),
        step=0.1,
        description="lowd sigma",
        continuous_udpate=True,
    )
    lowd_a_slider = widgets.FloatSlider(
        value=initial_guess[4],
        min=0.1,
        max=12.0,
        step=0.1,
        description="lowd a",
        continuous_udpate=True,
    )
    lowd_b_slider = widgets.FloatSlider(
        value=initial_guess[5],
        min=0.1,
        max=12.0,
        step=0.1,
        description="lowd b",
        continuous_udpate=True,
    )
    highd_sigma_slider = widgets.FloatSlider(
        value=initial_guess[0],
        min=0.1,
        max=my_ceil(np.max(dists), 1),
        step=0.1,
        description="highd sigma",
        continuous_udpate=True,
    )
    highd_a_slider = widgets.FloatSlider(
        value=initial_guess[1],
        min=0.1,
        max=12.0,
        step=0.1,
        description="highd a",
        continuous_udpate=True,
    )
    highd_b_slider = widgets.FloatSlider(
        value=initial_guess[2],
        min=0.1,
        max=12.0,
        step=0.1,
        description="highd b",
        continuous_udpate=True,
    )

    # histogram
    H, edges = np.histogram(dists, bins=bins, density=True)
    H *= 1 / max(H)
    centers = np.mean(np.vstack([edges[0:-1], edges[1:]]), axis=0)

    # highd sigmoid
    x_h = np.linspace(0, max(dists), n_values)
    highd_data = {
        "sigma": highd_sigma_slider.value,
        "a": highd_a_slider.value,
        "b": highd_b_slider.value,
    }
    y_h = sigmoid(x_h, **highd_data)

    # diff and norm
    dy = np.diff(y_h)
    dy_norm = dy / max(dy)

    # lowd sigmoid
    x_l = np.linspace(0, low_d_max, n_values)
    lowd_data = {
        "sigma": lowd_sigma_slider.value,
        "a": lowd_a_slider.value,
        "b": lowd_b_slider.value,
    }
    y_l = sigmoid(x_l, **lowd_data)
    edges_sig = sigmoid(edges, **highd_data)
    idx = np.argmin(
        np.abs(np.expand_dims(edges_sig, axis=1) - np.expand_dims(y_l, axis=0)), axis=1
    )
    edges_l = x_l[idx]

    # initial subplot with two traces
    fig = make_subplots(rows=3, cols=1, subplot_titles=["highd", "scaling", "lowd"])

    # add the bar
    fig.add_trace(
        go.Bar(
            x=centers,
            y=H,
            name="highd dists",
            xaxis="x1",
            yaxis="y1",
            marker_color="blue",
            marker_opacity=0.5,
        )
    )

    # add the lowd sigmoid
    fig.add_trace(
        go.Scatter(
            x=x_l,
            y=y_l,
            mode="lines",
            name="lowd sigmoid",
            line_color="orange",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # add the title
    fig.update_layout(
        height=800,
        width=600,
        title={
            "text": "Interact with the plot to select sigmoid parameters",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "middle",
        },
        # hovermode="x",
    )

    # add the highd sigmoids to a second axis
    fig.add_trace(
        go.Scatter(
            x=x_h,
            y=y_h,
            name="sigmoid",
            line_color="orange",
            mode="lines",
            xaxis="x4",
            yaxis="y4",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_h,
            y=dy_norm,
            name="diff sigmoid",
            line_color="green",
            mode="lines",
            xaxis="x4",
            yaxis="y4",
        ),
        row=1,
        col=1,
    )

    # some adjustmentns for xaxis3
    fig.update_layout(
        xaxis1=dict(
            title="highd distance",
            showgrid=True,
        ),
        # xaxis2=dict(
        #     showticklabels=False,
        # ),
        xaxis3=dict(
            title="lowd distance",
        ),
        yaxis2=dict(
            showticklabels=False,
        ),
        bargap=0,
    )

    # make the figure responsive
    # add connections lines
    trace_names = []
    _lowd_max = np.max(x_l).copy()
    _highd_max = np.max(x_h)
    for trace in get_connection_traces(edges, edges_l, _lowd_max, _highd_max):
        fig.add_trace(trace, row=2, col=1)
        trace_names.append(trace.name)

    # create a figure widget
    g = go.FigureWidget(fig)
    lowd_sigmoid_trace_index = [
        trace["name"] == "lowd sigmoid" for trace in g["data"]
    ].index(True)
    highd_sigmoid_trace_index = [
        trace["name"] == "sigmoid" for trace in g["data"]
    ].index(True)
    diff_sigmoid_trace_index = [
        trace["name"] == "diff sigmoid" for trace in g["data"]
    ].index(True)
    trace_names = np.where(
        np.in1d(np.asarray([t.name for t in g["data"]]), np.asarray(trace_names))
    )[0]
    object_mapping = {
        "lowd sigma": {"update_data": [lowd_sigmoid_trace_index], "keyword": "sigma"},
        "lowd a": {"update_data": [lowd_sigmoid_trace_index], "keyword": "a"},
        "lowd b": {"update_data": [lowd_sigmoid_trace_index], "keyword": "b"},
        "highd sigma": {
            "update_data": [highd_sigmoid_trace_index, diff_sigmoid_trace_index],
            "keyword": "sigma",
        },
        "highd a": {
            "update_data": [highd_sigmoid_trace_index, diff_sigmoid_trace_index],
            "keyword": "a",
        },
        "highd b": {
            "update_data": [highd_sigmoid_trace_index, diff_sigmoid_trace_index],
            "keyword": "b",
        },
    }

    # define the response function
    def response(change):
        nonlocal highd_data
        nonlocal lowd_data
        nonlocal edges
        key = change["owner"].description
        indices = object_mapping[key]["update_data"]
        kwarg = object_mapping[key]["keyword"]
        if "highd" in key:
            highd_data |= {kwarg: change["new"]}
        else:
            lowd_data |= {kwarg: change["new"]}
        y_h = sigmoid(x_h, **highd_data)
        y_l = sigmoid(x_l, **lowd_data)
        dy = np.diff(y_h)
        dy_norm = dy / max(dy)
        edges_sig = sigmoid(edges, **highd_data)
        idx = np.argmin(
            np.abs(np.expand_dims(edges_sig, axis=1) - np.expand_dims(y_l, axis=0)),
            axis=1,
        )
        new_edges_l = x_l[idx]

        # update the parameters
        if parameters is not None:
            if hasattr(parameters, "cartesian_dist_sig_parameters"):
                attr_name = "cartesian_dist_sig_parameters"
            else:
                attr_name = "dist_sig_parameters"
            payload = (
                highd_data["sigma"],
                highd_data["a"],
                highd_data["b"],
                lowd_data["sigma"],
                lowd_data["a"],
                lowd_data["b"],
            )
            setattr(parameters, attr_name, payload)

        # update the fig
        with g.batch_update():
            g.data[highd_sigmoid_trace_index].y = y_h
            g.data[diff_sigmoid_trace_index].y = dy_norm
            g.data[lowd_sigmoid_trace_index].y = y_l
            for i, (j, l, h) in enumerate(zip(trace_names, new_edges_l, edges)):
                # if i % 10 == 0:
                l_plot = l / _lowd_max
                h_plot = h / _highd_max
                g.data[j].x = [l_plot, h_plot]
                g.data[j].hovertemplate = f"{h:.2f} in highd maps to {l:.2f} in lowd"

    # observe the widgets
    lowd_sigma_slider.observe(response, names="value")
    lowd_a_slider.observe(response, names="value")
    lowd_b_slider.observe(response, names="value")
    highd_sigma_slider.observe(response, names="value")
    highd_a_slider.observe(response, names="value")
    highd_b_slider.observe(response, names="value")

    # create containers
    lowd_container = widgets.HBox(
        children=[lowd_sigma_slider, lowd_a_slider, lowd_b_slider]
    )
    highd_container = widgets.HBox(
        children=[highd_sigma_slider, highd_a_slider, highd_b_slider]
    )

    # start the app
    return widgets.VBox([lowd_container, highd_container, g])


def distance_histogram(
    data: np.ndarray,
    periodicity: float,
    sigmoid_parameters: tuple[float, float, float, float, float, float],
    axes: Optional[plt.Axes] = None,
    low_d_max: int = 5,
    bins: Union[Literal["auto"], int] = "auto",
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:  # pragma: no cover
    """Plots the histogram of all pairwise distances in the data.

    It also shows the sigmoid function and its normalized derivative.

    Args:
        data (np.ndarray): 2-dimensional numpy array. Columns should iterate
            over the dimensions of the datapoints, i.e. the dimensionality
            of the data. The rows should iterate over datapoints.
        periodicity (float): Periodicity of the data. Use float("inf")
            for non-periodic data.
        sigmoid_parameters (tuple): Tuple of sketchmap sigmoid parameters
            in shape (Sigma, A, B, sigma, a, b).
        axes (Union[np.ndarray, None], optional): A numpy array of two
            matplotlib.axes objects or None. If None is provided, the axes will
            be created. Defaults to None.
        low_d_max (int, optional): Upper limit for plotting the low_d sigmoid.
            Defaults to 5.
        bins (Union[str, int], optional): Number of bins for histogram.
            Use 'auto' to let matplotlib decide how many bins to use. Defaults to 'auto'.

    Returns:
        tuple: A tuple containing the following:
            - plt.axes: A matplotlib.pyplot axis used to plot the high-d distance
                sigmoid.
            - plt.axes: A matplotlib.pyplot axis used to plot the high-d distance
                histogram (a twinx of the first axis).
            - plt.axes: A matplotlib.pyplot axis used to plot the lowd sigmoid.

    """

    vecs = periodic_distance_np(
        np.expand_dims(data, axis=1), np.expand_dims(data, axis=0), periodicity
    )
    dists = np.linalg.norm(vecs, axis=2)
    while True:
        try:
            dists = np.linalg.norm(dists, axis=2)
        except np.AxisError:
            break
    dists = dists.reshape(-1)

    if axes is None:
        fig, axes = plt.subplots(2)
    axe2 = axes[0].twinx()
    counts, edges, patches = axe2.hist(
        dists, bins=bins, density=True, edgecolor="black"
    )
    x = np.linspace(0, max(dists), 1000)
    y = sigmoid(x, *sigmoid_parameters[:3])
    edges_sig = sigmoid(edges, *sigmoid_parameters[:3])
    dy = np.diff(y)
    dy_norm = dy / max(dy)
    axes[0].plot(x, y, color="C1", label="sigmoid")
    axes[0].plot(x[:-1], dy_norm, color="C2", label="diff sigmoid")

    axes[0].legend()
    axes[0].set_xlabel("distance")
    axes[0].set_ylim((0, 1))
    axes[0].set_zorder(axe2.get_zorder() + 1)
    axes[0].patch.set_visible(False)
    axes[0].set_title("high-d")

    x = np.linspace(0, low_d_max, 1000)
    y = sigmoid(x, *sigmoid_parameters[3:])
    dy = np.diff(y)
    dy_norm = dy / max(dy)
    idx = np.argmin(
        np.abs(np.expand_dims(edges_sig, axis=1) - np.expand_dims(y, axis=0)), axis=1
    )
    edges_x = x[idx]

    axes[1].plot(x, y, color="C1", label="sigmoid")

    axes[1].legend()
    axes[1].set_xlabel("distance")
    axes[1].set_ylim((0, 1))
    axes[1].set_title("low-d")
    for i in range(len(edges)):
        if edges_x[i] != edges_x[-1]:
            axes[1].annotate(
                "",
                xy=(edges[i], 0),
                xytext=(edges_x[i], 0),
                xycoords=axes[0].transData,
                textcoords=axes[1].transData,
                arrowprops=dict(facecolor="black", arrowstyle="-", clip_on=False),
            )
    axes[0].figure.tight_layout()
    return axes[0], axe2, axes[1]


def plot_raw_data(
    xyz: Union[np.ndarray, "SingleTraj"],
    frame_slice: slice = slice(0, 5),
    atom_slice: slice = slice(0, 50, 5),
) -> go.Figure:  # pragma: no cover
    """Plots the raw data of a trajectory as xyz slices in a 3D plot.

    Conventions:
        * x: The cartesian coordinates.
        * y: The atom.
        * z: The frame.

    Args:
        xyz (Union[np.ndarray], "SingleTraj"]): Can be either a numpy array with
            shape (n_frames, n_atoms, 3) or a SingleTraj object.
        frame_slice (slice): A slice to select the frames you want.
        atom_slice (slice): A slice to select the atoms you want.

    """
    if not isinstance(xyz, np.ndarray):
        data = xyz.xyz
        frame_extend, atom_extend = data.shape[:2]
        atoms_ind = []
        for i in np.array(xyz.top.select("all"))[atom_slice]:
            atoms_ind.append(str(xyz.top.atom(i)))
    else:
        data = xyz
        frame_extend, atom_extend = data.shape[:2]
        atoms_ind = np.arange(atom_extend)[atom_slice]
    frames_ind = np.arange(frame_extend)[frame_slice]

    # create the surfaces
    surfaces = []
    cmin = float("inf")
    cmax = -float("inf")
    for frame, xyz_slice in zip(frames_ind, data[frame_slice, atom_slice]):
        x = np.arange(3)
        y = np.arange(len(atoms_ind))
        x, y = np.meshgrid(x, y)
        z = np.full(x.shape, frame)
        cmin = min([cmin, xyz_slice.min()])
        cmax = min([cmax, xyz_slice.max()])
        customdata = np.stack(
            (
                np.full(xyz_slice.T.shape, fill_value=frame),
                np.tile(atoms_ind, [3, 1]),
                np.tile(["x", "y", "z"], [len(atoms_ind), 1]).T,
                xyz_slice.T,
            ),
            axis=-1,
        )
        text = (
            "Cartesian coordinate %{customdata[2]}<br>of atom %{customdata[1]}<br>at "
            "frame %{customdata[0]:.d}: %{customdata[3]:.3f}"
        )
        surfaces.append(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=xyz_slice,
                customdata=customdata,
                coloraxis="coloraxis",
                hovertemplate=text,
                name="",
            )
        )

    # create the figure
    fig = go.Figure(data=surfaces)
    fig.update_layout(
        title_text="Raw data plot",
        title_x=0.5,
        scene=dict(
            xaxis_title="xyz",
            yaxis_title="Atom",
            zaxis_title="Frame no.",
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["x", "y", "z"],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=np.arange(len(atoms_ind)),
                ticktext=atoms_ind,
            ),
            zaxis=dict(
                tickmode="array",
                tickvals=frames_ind,
                ticktext=frames_ind,
            ),
        ),
        legend_title="Cartesian coordinate value",
        width=700,
        height=700,
        coloraxis=dict(
            colorscale="viridis",
            colorbar_thickness=25,
            colorbar_len=0.75,
            cmin=cmin,
            cmax=cmax,
        ),
        coloraxis_colorbar=dict(
            title="value of coordinate",
        ),
    )
    return fig


def plot_ball_and_stick(
    traj: "SingleTraj",
    frame_subsample: Union[int, slice] = slice(None, None, 100),
    highlight: Literal["atoms", "bonds", "angles", "dihedrals"] = "atoms",
    atom_indices: Optional[Sequence[int]] = None,
    custom_colors: Optional[dict[int, str]] = None,
    add_angle_arcs: bool = True,
    animation: bool = False,
    persistent_hover: bool = False,
    flatten: bool = False,
) -> None:  # pragma: no cover
    fig = _plot_ball_and_stick(
        traj=traj,
        frame_subsample=frame_subsample,
        highlight=highlight,
        atom_indices=atom_indices,
        custom_colors=custom_colors,
        add_angle_arcs=add_angle_arcs,
        animation=animation,
        persistent_hover=persistent_hover,
        flatten=flatten,
    )
    fig.show()


def plot_ramachandran(
    angles: Union[tuple[np.ndarray, np.ndarray], np.ndarray, "SingleTraj"],
    subsample: Optional[Union[int, slice, np.ndarray]] = None,
) -> None:  # pragma: no cover
    """Plots a Ramachandran plot using plotly.

    Args:
        angles (Union[tuple[np.ndarray, np.ndarray], np.ndarray, "SingleTraj"]):
            Either a tuple of np.ndarray in which case it is assumed that the
            arrays are ordered like (psi, phi). Or an array of shape
            (2, n_frames, n_angles), in which case it is unpacked into psi and
            phi angles.
        subsample (Optional[Union[int, slice, np.ndarray]]): Any way to subsample
            the data along the time-axis. Can be int (one frame), slice (more frames,
            defined by start, stop, step) or np.ndarray (more frames defined by
            their integer index).

    """
    if isinstance(angles, tuple):
        psi, phi = angles
    elif isinstance(angles, np.ndarray):
        if angles.ndim == 3:
            psi, phi = angles
        else:
            psi, phi = angles[::2], angles[1::2]
    elif angles.__class__.__name__ == "SingleTraj":
        if not "central_dihedrals" in angles._CVs:
            angles.load_CV("central_dihedrals")
        _angles = angles._CVs.central_dihedrals
        psi = _angles[
            0, ..., _angles.CENTRAL_DIHEDRALS.str.lower().str.contains("psi")
        ].values
        phi = _angles[
            0, ..., _angles.CENTRAL_DIHEDRALS.str.lower().str.contains("phi")
        ].values
    else:
        raise ValueError("Wrong type for arg `angles`.")

    psi = psi.flatten()
    phi = phi.flatten()
    if np.all(psi < 4):
        mode = "rad"
        tickrange = np.linspace(-np.pi, np.pi, 5)
        ranges = [-np.pi, np.pi]
    else:
        mode = "deg"
        ranges = [-180, 180]
    tickrange = np.linspace(ranges[0], ranges[1], 5)
    ticklabels = [-180, -90, 0, 90, 180]
    if subsample is not None:
        psi = psi[::subsample]
        phi = phi[::subsample]

    fig = px.density_contour(
        x=phi,
        y=psi,
        marginal_x="violin",
        marginal_y="violin",
        labels={"x": "phi", "y": "psi"},
        range_x=ranges,
        range_y=ranges,
    )

    fig.data[0]["contours"].coloring = "fill"

    fig.update_layout(
        width=700,
        height=700,
        title_text="Ramachandran plot",
        xaxis=dict(
            tickmode="array",
            tickvals=tickrange,
            ticktext=ticklabels,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tickrange,
            ticktext=ticklabels,
        ),
    )

    fig.show()


def plot_dssp(
    traj: SingleTraj,
    simplified: bool = True,
    subsample: Optional[Union[int, slice, np.ndarray]] = None,
    residue_subsample: int = 25,
) -> go.Figure:  # pragma: no cover
    # get the dssp and color values
    # Third Party Imports
    import mdtraj as md

    dssp = md.compute_dssp(traj, simplified=simplified)

    # the yticks and yticklabels are created here
    residues = np.arange(traj.top.n_residues)
    residue_names = np.array([str(r) for r in traj.top.residues])

    # subsample the dssp array
    if subsample is not None:
        if isinstance(subsample, int):
            subsample = slice(None, None, subsample)
        dssp = dssp[subsample]

    # vectorize the dssp-str -> rgb-value function and apply
    func = np.vectorize(partial(dssp_to_rgb, simplified=simplified))
    dssp_color = np.swapaxes(np.dstack(func(dssp)), 0, 1)
    func = np.vectorize(partial(dssp_to_text, simplified=simplified))
    dssp_text = func(dssp)

    # create fig
    fig = px.imshow(dssp_color)
    customdata_res_names = np.tile(residue_names, (dssp.shape[0], 1))
    customdata = np.stack((customdata_res_names.T, dssp_text.T), axis=-1)
    fig.update(
        data=[
            {
                "customdata": customdata,
                "hovertemplate": "Time: %{x}<br>Residue: %{customdata[0]}<br>DSSP: %{customdata[1]}",
                "name": "",
            },
        ],
    )

    # subsample the residues for label purposes
    if residue_subsample > -1:
        residues = residues[::residue_subsample]
        residue_names = residue_names[::residue_subsample]

    # combine and update layout
    # fig = go.Figure(data=fig1.data + fig2.data)
    fig.update_layout(
        width=1000,
        height=700,
        title="DSSP plot",
        xaxis=dict(title="time"),
        yaxis=dict(
            title="residue",
            tickmode="array",
            tickvals=residues,
            ticktext=residue_names,
        ),
        legend=dict(
            title="DSSP",
            orientation="h",
        ),
    )

    # add the legend
    simplified_legend = {
        "Coil": "rgb(1.0, 1.0, 1.0)",
        "Extended": "rgb(1.0, 0.0, 0.0)",
        "Helical": "rgb(0.0, 0.0, 1.0)",
    }
    legend = {
        "Coil": "rgb(1.0, 1.0, 1.0)",
        "Isolated beta-bridge": "rgb(0.0, 0.0, 0.0)",
        "Extended beta-ladder": "rgb(1.0, 0.0, 0.0)",
        "3/10-helix": "rgb(0.5, 0.5, 0.5)",
        "Alpha-helix": "rgb(0.0, 0.0, 1.0)",
        "Pi-helix": "rgb(0.0, 1.0, 1.0)",
        "Bend": "rgb(0.0, 1.0, 0.0)",
        "Hydrogen bonded Turn": "rgb(1.0, 1.0, 0.0)",
    }
    iterator = simplified_legend if simplified else legend
    for key, val in iterator.items():
        trace = go.Bar(
            x=[0],
            y=[0],
            name=key,
            legend="legend1",
            # visible="legendonly",
            showlegend=True,
        )
        trace.update(
            marker_color=val,
        )
        fig.add_trace(trace)
    # show
    return fig


def dssp_to_text(
    val: str,
    simplified: bool = False,
) -> str:  # pragma: no cover
    simplified_dssp = {"C": "Coil", "E": "Extended", "H": "Helical"}
    dssp = {
        " ": "Coil",
        "B": "Isolated beta-bridge",
        "E": "Extended beta-ladder",
        "G": "3/10-helix",
        "H": "Alpha-helix",
        "I": "Pi-helix",
        "S": "Bend",
        "T": "Hydrogen bonded Turn",
    }
    if simplified:
        return simplified_dssp[val]
    return dssp[val]


def dssp_to_rgb(
    val: str,
    simplified: bool = False,
) -> tuple[int, int, int]:  # pragma: no cover
    """Here are the values returned for simplified:

        * "C": coil, white, rgb(1, 1, 1)
        * "E": extended, red, rgb(1, 0, 0)
        * "H": helix, blue, rgb(0, 0, 1)

    And here for the full DSSP assignment:

        * " ": coil, white, rgb(1, 1, 1)
        * "B": b-bridge, black, rgb(0, 0, 0)
        * "E": b-sheet, red, rgb(1, 0, 0)
        * "G": 3_10 helix, grey, rgb(0.5, 0.5, 0.5)
        * "H": A-helix, blue, rgb(0, 0, 1)
        * "I": pi-helix, purple, rgb(0, 1, 1)
        * "S": bend, green, rgb(0, 1, 0)
        * "T": turn, yellow(1, 1, 0)

    Args:
        val (str): The dssp value.
        simplified (bool): Whether to use the simplified scheme.

    """
    simplified_dssp = {"C": (1.0, 1.0, 1.0), "E": (1.0, 0.0, 0.0), "H": (0.0, 0.0, 1.0)}
    dssp = {
        " ": (1.0, 1.0, 1.0),
        "B": (0.0, 0.0, 0.0),
        "E": (1.0, 0.0, 0.0),
        "G": (0.5, 0.5, 0.5),
        "H": (0.0, 0.0, 1.0),
        "I": (0.0, 1.0, 1.0),
        "S": (0.0, 1.0, 0.0),
        "T": (1.0, 1.0, 0.0),
    }
    if simplified:
        return simplified_dssp[val]
    return dssp[val]


def plot_end2end(
    traj: SingleTraj,
    selstr: str = "name CA",
    subsample: Optional[Union[int, slice, np.ndarray]] = None,
    rolling_avg_window: int = 5,
) -> go.Figure:  # pragma: no cover
    atoms = traj.top.select(selstr)[[0, -1]]
    dists = md.compute_distances(traj, [atoms])[:, 0]
    time = traj.time
    if subsample is not None:
        if isinstance(subsample, int):
            subsample = slice(None, None, subsample)
        dists = dists[subsample]
        time = time[subsample]

    fig = px.scatter(
        x=time,
        y=dists,
        labels=dict(
            x="time in ps",
            y="dist in nm",
        ),
        opacity=0.2,
        trendline="rolling",
        trendline_options=dict(
            window=rolling_avg_window,
        ),
        title="end to end distance",
        marginal_y="violin",
    )
    return fig


def _zoomingBoxManual(ax1, ax2, color="red", linewidth=2, roiKwargs={}, arrowKwargs={}):
    """Fakes a zoom effect between two mpl.axes.Axes.

    Uses mpl.patches.ConnectionPatch and mpl.patches.Rectangle
    to make it seem like ax2 is a zoomed in version of ax1.
    Instead of defining the coordinates of the zooming rectangle
    The axes limits of ax2 are used.

    Args:
        ax1 (plt.axes): The axes with the zoomed-out data.
        ax2 (plt.axes): The second axes with the zoomed-in data.
        color (str): The color of the zoom effect. Is passed into mpl,
            thus can be str, or tuple, ... Defaults to 'red'
        linewidth (int): The linewidth. Defaults to 2.
        roiKwargs (dict): Keyworded arguments for the rectangle.
            Defaults to {}.
        arrowKwargs (dict): Keyworded arguments for the arrow.
            Defaults to {}.

    """
    limits = np.array([*ax2.get_xlim(), *ax2.get_ylim()])
    roi = limits
    roiKwargs = dict(
        dict(
            [
                ("fill", False),
                ("linestyle", "dashed"),
                ("color", color),
                ("linewidth", linewidth),
            ]
        ),
        **roiKwargs,
    )
    ax1.add_patch(
        mpl.patches.Rectangle(
            [roi[0], roi[2]], roi[1] - roi[0], roi[3] - roi[2], **roiKwargs
        )
    )
    arrowKwargs = dict(
        dict([("arrowstyle", "-"), ("color", color), ("linewidth", linewidth)]),
        **arrowKwargs,
    )
    corners = np.vstack([limits[[0, 1, 1, 0]], limits[[2, 2, 3, 3]]]).T
    con1 = mpl.patches.ConnectionPatch(
        xyA=corners[0],
        xyB=corners[1],
        coordsA="data",
        coordsB="data",
        axesA=ax2,
        axesB=ax1,
    )
    con1.set_color([0, 0, 0])
    ax2.add_artist(con1)
    con1.set_linewidth(2)
    con2 = mpl.patches.ConnectionPatch(
        xyA=corners[3],
        xyB=corners[2],
        coordsA="data",
        coordsB="data",
        axesA=ax2,
        axesB=ax1,
    )
    con2.set_color([0, 0, 0])
    ax2.add_artist(con2)
    con2.set_linewidth(2)


def render_vmd(
    filepath,
    rotation=[0, 0, 0],
    scale=1,
    script_location="auto",
    image_location="auto",
    debug=False,
    image_name="",
    drawframes=False,
    ssupdate=True,
    renderer="tachyon",
    additional_spheres=[],
    additional_lines=[],
    surf=None,
    custom_script=None,
):
    """Render pdb file with a combination of vmd, tachyon and image magick.

    This function creates a standardised vmd tcl/tk script and writes it
    to disk. Then vmd is called with the subprocess package and used to
    create a tachyon input file. Tachyon is then called to render the image
    with ambient occlusion and soft lighting. The output is a high quality
    targa (.tga) image, which is converted to png using image magick.

    Args:
        filepath (str): Location of the pdb file which should be rendered.
        rotation ([x_rot, y_rot, z_rot], optional): List of rotation values. Defaults to [0, 0, 0].
        scale (float, optional): By how much the structure should be scaled. Defaults to 1.
        script_location (str, optional): Where to save the script. Script will be removed
            after finish nonehteless. Defaults to 'auto' and writes into cwd.
        image_location (str, optional): Where to render the images file to. Will be
            deleted nonetheless. Don't give an extension for this. Defaults to 'auto' and
            writes into cwd.
        debug (bool, optional): Print debug info. Defaults to False.
        image_name (str, optional): This string will be used to save the image to after it has
            been rendered and converted to png. This will not be deleted. Defaults to ''.
        drawframes (bool, optional): If a trajectory is loaded, this will render all frames in it.
            Defaults to False.
        ssupdate (bool, optional): Updates the secondary structure for every frame. Normally
            vmd uses the secondary structure of the first frame. Setting this to True calcs
            the sec struct for every frame. Defaults to True.
        renderer (str, optional): Which renderer to use.
            * 'tachyon' uses the external Tachyon rendered. So vmd -> .dat -> .tga -> .png.
            * 'snapshot' uses the vmd internal snapshot renderer.
            Defaults to 'tachyon'.
        additional_spheres (list, optional): Draw spheres around two subunits to make
            them visually more distinct. Takes a list of lists. Each list in the main
            list should contain 4 values [x, y, z, r] (r for radius). Defaults to [].
        additional_lines (list, optional): A list of additional lines that should be added to the
            script. Please refert to the vmd manual for further info. Defaults to [].
        surf (Union[str, None], optional): A string defining the surface renderer. Can either be
            'quicksurf' or 'surf'. If None is provided, the surface won't be rendered (falls back
            to cartoon representation). Defaults to None.
        custom_script (Union[str, None], optional): Provide a completely custom script as this option.
            The render commands will still be appended to this script. If None is provided, the
            default script will be used.

    See also:
        See this nice webpage about rendering publication worthy images with vmd.
        https://www.ks.uiuc.edu/Research/vmd/minitutorials/tachyonao/

    Returns:
        image (np.ndarray): This array contains the raw pixel data.
            Can be used with matplotlib to have a quick view of the image.

    """
    if "." in image_location:
        raise Exception(
            "The argument image_location does not take a file extension, because the name is used for a .dat, .tga and .png file."
        )

    # add a shebang to the script
    # script = '#!/home/soft/bin/vmd\n\n'

    # print debug hello world
    script = 'puts "Hello World"\n'

    # if a list of files is provided we iterate over them
    if isinstance(filepath, list):
        for i, file in enumerate(filepath):
            # load molecule and change representation
            script += f"mol new {file}\n"
            if surf is None:
                script += f"mol modstyle 0 {i} newcartoon 0.3 50\n"
                script += f"mol modcolor 0 {i} structure\n"
            elif surf == "quicksurf":
                script += f"mol modstyle 0 {i} quicksurf 0.6 0.7 0.7 Medium\n"
            else:
                script += f"mol modstyle 0 {i} {surf}\n"
            script += f"mol modmaterial 0 {i} AOChalky\n"
            if drawframes and md.load(file).n_frames > 1:
                if renderer == "STL":
                    # Standard Library Imports
                    import warnings

                    warnings.warn(
                        "Rendering multiple frames with STL may lead to "
                        "undesired results. Instead of yielding the union of "
                        "all single-frame surfaces, you will get a mishmash of "
                        "all surfaces with intersection faces etc."
                    )
                script += f"mol drawframes 0 {i} 0:1:999\n"
    else:
        # load molecule and change representation
        script += f"mol new {filepath}\n"
        if surf is None:
            script += "mol modstyle 0 0 newcartoon 0.3 50\n"
            script += "mol modcolor 0 0 structure\n"
        elif surf == "quicksurf":
            script += "mol modstyle 0 0 quicksurf 0.6 0.7 0.7 Medium\n"
        else:
            script += f"mol modstyle 0 0 {surf}\n"
        script += "mol modmaterial 0 0 AOChalky\n"
        if drawframes:
            script += "mol drawframes 0 0 0:1:999\n"

    if ssupdate:
        print(
            "\033[93m"
            + "For the ssupdate function to work encodermap/vmd/sscache.tcl will be sourced within vmd. If no Error is thrown the file is present."
            + "\033[0m"
        )
        sscache_location = (
            os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
            + "/vmd/sscache.tcl"
        )
        if not os.path.isfile(sscache_location):
            raise FileNotFoundError(
                f"The sscache.tcl script is not here: {sscache_location}. Please put it there."
            )
        script += f"source {sscache_location}\n"
        script += "start_sscache 0\n"
    #         script += "proc update_secondary_structure_assigment { args } {"
    #         script += "  foreach molid [molinfo list] {"
    #         script += "    mol ssrecalc $molid"
    #         script += "  }"
    #         script += "}"
    #         script += "trace variable vmd_frame(0) w update_secondary_structure_assigment"

    # change some parameters to make a nice image
    script += "color Display Background white\n"
    script += "color Axes Labels black\n"
    script += "display depthcue off\n"
    script += "display ambientocclusion on\n"
    script += "display aoambient 1.0\n"
    script += "display aodirect 0.3\n"
    script += "display antialias on\n"
    # script += 'display resize 2000 2000\n'
    script += "axes location off\n"

    # scale and rotate
    script += f"rotate x by {rotation[0]}\n"
    script += f"rotate y by {rotation[1]}\n"
    script += f"rotate z by {rotation[2]}\n"
    script += f"scale by {scale}\n"

    # define image location
    if image_location == "auto":
        image_location = os.getcwd() + "/vmdscene"

    # add spheres
    if np.any(additional_spheres):
        for _, color in zip(additional_spheres, ["grey", "iceblue"]):
            x, y, z, r = np.round(_, 2)
            script += f"draw color {color}\n"
            script += f"draw sphere {{ {x} {y} {z} }} radius {r} resolution 25\n"
            script += "draw material Transparent\n"

    # add additional lines
    if additional_lines:
        for line in additional_lines:
            script += line + "\n"

    if custom_script is not None:
        script = custom_script

    # render command. Alternatively, I can use external Tachyon, which makes better images
    if renderer == "tachyon":
        script += f"render Tachyon {image_location}.dat\n"
    elif renderer == "snapshot":
        script += "render aasamples TachyonInternal 6\n"
        script += f"render TachyonInternal {image_location}.tga\n"
    elif renderer == "STL":
        script += "axes location off\n"
        script += f"render STL {image_location}.stl\n"
    elif renderer == "Wavefront":
        script += "axes location off\n"
        script += f"render Wavefront {image_location}.obj\n"
    else:
        raise NotImplementedError(
            "Other renderers than tachyon and snaphsot currently not supported."
        )

    # list molecules and quit
    script += "mol list\n"
    script += "quit"

    if debug:
        print(script)

    # write the script
    if script_location == "auto":
        script_location = os.getcwd() + "/vmd_script.tcl"
    with open(script_location, "w") as f:
        f.write(script)

    # call vmd -e script
    cmd = f"vmd -e {script_location} -dispdev none"
    if debug:
        print(cmd)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    (stdout, stderr) = proc.communicate()
    if debug:
        print(stdout.decode("utf-8"))
        print("\n")
        print(stderr.decode("utf-8"))

    # check if image has been written
    if renderer == "tachyon":
        assert os.path.isfile(
            f"{image_location}.dat"
        ), "Tachyon datafile not generated by renderer"
    else:
        assert os.path.isfile(
            f"{image_location}.tga"
        ), f"Snapshot image not created. {stderr.decode()} {stdout.decode()}"

    time.sleep(2)
    assert os.path.isfile(
        f"{image_location}.tga"
    ), f"Tachyon datafile not generated by renderer. Here's the script:\n\n{script}\n\n"

    if renderer == "tachyon":
        # call Tachyon and render
        cmd = f"/usr/bin/tachyon -aasamples 12 {image_location}.dat -res 2000 2000 -fullshade -format TARGA -o {image_location}.tga"
        if debug:
            print(cmd)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        (stdout, stderr) = proc.communicate()
        if debug:
            print(stdout.decode("utf-8"))
            print("\n")
            print(stderr.decode("utf-8"))

        # check if image has been written
        assert os.path.isfile(
            f"{image_location}.tga"
        ), "Tachyon renderer did not render image"

    if renderer == "STL":
        if image_name:
            shutil.copyfile(f"{image_location}.stl", image_name)
        # Third Party Imports
        import trimesh

        mesh = trimesh.load(f"{image_location}.stl")
        os.remove(f"{image_location}.stl")
        return mesh

    if renderer == "Wavefront":
        if image_name:
            shutil.copyfile(f"{image_location}.obj", image_name)
            shutil.copyfile(f"{image_location}.mtl", image_name.replace(".obj", ".mtl"))
        print(
            f"Find the rendered images at {image_name} and {image_name.replace('.obj', '.mtl')}."
        )
        return None

    # convert to png
    cmd = f"/usr/bin/convert {image_location}.tga {image_location}.png"
    if debug:
        print(cmd)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    (stdout, stderr) = proc.communicate()
    if debug:
        print(stdout.decode("utf-8"))
        print("\n")
        print(stderr.decode("utf-8"))

    # read image
    image = plt.imread(f"{image_location}.png")

    # write image if name has been provided
    if image_name:
        if os.path.isabs(image_name):
            shutil.copyfile(f"{image_location}.png", image_name)
        else:
            shutil.copyfile(f"{image_location}.png", os.getcwd() + f"/{image_name}")

    # remove temporary files
    if renderer == "tachyon":
        os.remove(f"{image_location}.dat")
    os.remove(f"{image_location}.tga")
    os.remove(f"{image_location}.png")
    # os.remove(f'{script_location}')

    # return matplotlib image object
    return image


def plot_cluster(
    trajs, pdb_path, png_path, cluster_no=None, col="_user_selected_points"
):
    # Third Party Imports
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(20, 20)

    if cluster_no is None:
        cluster_no = trajs.CVs[col].max()

    # prepare ax1 to make the two side histograms
    divider = make_axes_locatable(ax4)
    axHistx = divider.append_axes("top", size=1.2, pad=0.1)  # , sharex=ax1)
    axHisty = divider.append_axes("right", size=1.2, pad=0.1)  # , sharey=ax1)

    # some data management
    data = trajs.lowd
    where = np.where(trajs.CVs[col] == cluster_no)
    not_where = np.where(trajs.CVs[col] != cluster_no)
    x = data[:, 0]
    y = data[:, 1]

    # scatter everything grey and cluster blue
    ax1.scatter(*data[where].T)
    ax1.scatter(*data[not_where].T, c="grey", s=5)
    ax1.set_xlabel("x in a.u.")
    ax1.set_ylabel("y in a.u.")
    ax1.set_title(f"Scatter of low-dimensional data")

    # density
    bin_density = 46
    log_density = True

    # ax2 gets hexbin density
    # x_bins = np.linspace(x.min(), x.max(), bin_density)
    # y_bins = np.linspace(y.min(), y.max(), bin_density)
    H, xedges, yedges = np.histogram2d(x=x, y=y, bins=bin_density)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
    ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
    X, Y = np.meshgrid(xcenters, ycenters)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if log_density:
        with np.errstate(divide="ignore"):  # ignore division by zero error
            F = np.log(H)
    else:
        F = H
    # mappable = ax2.hexbin(x=X.ravel(), y=Y.ravel(), C=F.T.ravel(), cmap=plt.cm.turbo_r, extent=extent,
    #                       norm=mpl.colors.PowerNorm(1), gridsize=bin_density +1)
    cmap = plt.get_cmap("turbo").with_extremes(under="w")
    mappable = ax2.contourf(
        X, Y, H.T, cmap=cmap, levels=np.linspace(0.001, H.max(), 20)
    )
    ax2.set_xlabel("x in a.u.")
    ax2.set_ylabel("y in a.u.")
    ax2.set_title("Log density of points")

    # colorbar for ax2
    # colorbar
    # use the axes divider method to add colorbar
    ax_divider = make_axes_locatable(ax2)
    # add colorbaraxis to work with ticks and whatnot
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    # define colorbar norm. I like to work with values between 0 and 1
    # initialize colormap
    cb = plt.colorbar(mappable, cax=cax)
    cax.set_ylabel("Number of points")

    # cluster on ax4
    # x hist
    spines = [k for k in axHistx.spines.values()]
    spines[1].set_linewidth(0)
    spines[3].set_linewidth(0)
    axHistx.set_xticks([])
    H, edges, patches = axHistx.hist(data[:, 0][where], bins=50)
    axHistx.set_ylabel("count")
    axHistx.set_title("Scatter of Cluster")

    # y hist
    spines = [k for k in axHisty.spines.values()]
    spines[1].set_linewidth(0)
    spines[3].set_linewidth(0)
    axHisty.set_yticks([])
    H, edges, patches = axHisty.hist(
        data[:, 1][where], bins=50, orientation="horizontal"
    )
    axHisty.set_xlabel("count")

    # scatter data
    ax4.scatter(x=data[where, 0], y=data[where, 1])
    spines = [k for k in ax4.spines.values()]
    spines[3].set_linewidth(0)
    spines[1].set_linewidth(0)
    ax4.set_xlabel("x in a.u.")
    ax4.set_ylabel("y in a.u.")

    # annotate rms
    rms = np.np.floor(
        (1 / len(data[where]))
        * np.sum(
            (data[where, 0] - np.mean(data[where, 0])) ** 2
            + (data[where, 1] - np.mean(data[where, 1])) ** 2
        )
    )
    text = f"RMS = {np.round(rms, decimals=5)}"
    ax4.text(0.05, 0.95, text, transform=ax1.transAxes)

    # annotate geometric center
    centroid = [np.mean(x[where]), np.mean(y[where])]
    ax4.scatter(*centroid, s=50, c="C1")
    ax4.annotate(
        "geom. center",
        xy=centroid,
        xycoords="data",
        xytext=(0.95, 0.95),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="black", shrink=0.05, fc="w", ec="k", lw=2),
        horizontalalignment="right",
        verticalalignment="top",
        color="C1",
    )

    # annotate rmsd center
    # view, dummy_traj = gen_dummy_traj(trajs, cluster_no, max_frames=100, col=col)
    # index, distances, centroid = rmsd_centroid_of_cluster(dummy_traj, parallel=False)
    # idx = np.round(np.linspace(0, len(where) - 1, 100)).astype(int)
    # where = where[idx]
    # centroid = data[where[0][::5][index]]
    # ax4.scatter(*centroid, s=50, c='C2')
    # ax4.annotate('rmsd center',
    #             xy=centroid, xycoords='data',
    #             xytext=(0.95, 0.85), textcoords='axes fraction',
    #             arrowprops=dict(facecolor='black', shrink=0.05, fc="w", ec="k", lw=2),
    #             horizontalalignment='right', verticalalignment='top', color='C2')

    # make vmd snapshot
    try:
        image = render_vmd(
            pdb_path, drawframes=True, renderer="tachyon", debug=False, scale=1.5
        )
        ax3.imshow(image)
        [k.set_linewidth(0) for k in ax3.spines.values()]
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Image of cluster")
    except:
        ax3.annotate("VMD Rendering not possible", (0.5, 0.5))
        pass

    # # calculate distances between rmsd centroid and all other points
    # distances = scipy.spatial.distance.cdist(centroid.reshape(1, 2), np.stack([x, y]).T)
    # H, edges, patches = ax3.hist(distances.flatten(), color='C1')
    # ax3.set_title("Distances to rmsd centroid.")
    # ax3.set_xlabel("Distance in a.u.")
    # ax3.set_ylabel("Count")

    plt.suptitle(f"Cluster {cluster_no}")
    plt.savefig(png_path, transparent=False)
    plt.close(fig)
