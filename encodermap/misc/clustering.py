# -*- coding: utf-8 -*-
# encodermap/misc/clustering.py
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
"""Functions for building clusters."""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy

# Third Party Imports
import numpy as np
from optional_imports import _optional_import


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
ngl = _optional_import("nglview")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from typing import TYPE_CHECKING, Optional, TypedDict

# Third Party Imports
from typing_extensions import NotRequired


if TYPE_CHECKING:
    # Third Party Imports
    from mdtraj import Topology, Trajectory
    from pandas import DataFrame, Series

    # Local Folder Imports
    from ..trajinfo import SingleTraj, TrajEnsemble

    class ClusterDict(TypedDict):
        """The output of the `cluster_to_dict()` function."""

        stacked: Trajectory
        joined_per_top: dict[Topology, Trajectory]
        ensemble: TrajEnsemble
        series: Series
        joined: NotRequired[Trajectory]


################################################################################
# Globals
################################################################################


__all__: list[str] = []


################################################################################
# Public Functions
################################################################################


def rmsd_centroid_of_cluster(
    traj: Trajectory,
    parallel: bool = True,
    atom_indices: Optional[np.ndarray] = None,
) -> tuple[int, np.ndarray, Trajectory]:
    """Computes the rmsd centroid of a trajectory.

    Args:
        traj (mdtraj.Trajectory): The trajectory.
        parallel (bool): Use OpenMP to calculate each of the RMSDs in
            parallel over multiple cores.
        atom_indices

    Returns:
        tuple: A tuple containing:
            - index (int): The index of the centroid.
            - distances (np.ndarray): The RMSD distance matrix with shape
                traj.n_frames x traj.n_frames
            - centroid (mdtraj.Trajectory): The traj of the centroid.

    References:
        Uses the algorithm found on http://mdtraj.org/latest/examples/centroids.html

    """
    if not np.any(atom_indices):
        atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != "H"]
    distances = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        distances[i] = md.rmsd(
            traj, traj, i, atom_indices=atom_indices, parallel=parallel
        )
    beta = 1
    index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
    centroid = traj[index]
    return index, distances, centroid


def cluster_to_dict(
    trajs: TrajEnsemble,
    align_string: str = "name CA",
    ref_align_string: str = "name CA",
    base_traj: Optional[Trajectory] = None,
) -> ClusterDict:
    """Creates a dictionary with joined/stacked trajectory frames.

    Examples:
        >>> import encodermap as em
        >>> import numpy as np
        >>> trajs = em.TrajEnsemble(
        ...     [
        ...         em.SingleTraj.from_pdb_id("1YUG"),
        ...         em.SingleTraj.from_pdb_id("1YUF"),
        ...     ]
        ... )
        >>> cluster_membership = np.full(shape=(trajs.n_frames, ), fill_value=-1, dtype=int)
        >>> cluster_membership[::2] = 0
        >>> trajs.load_CVs(cluster_membership, "cluster_membership")
        >>> cluster = trajs.cluster(cluster_id=0)
        >>> cluster_dict = em.misc.clustering.cluster_to_dict(cluster)

        Only, when the trajectories have all the same number of atoms, then
        we can join them all along the timne axis. And only then does the
        `'joined'` key appear in the output.

        >>> trajs.top  # doctest: +ELLIPSIS
        [<mdtraj.Topology with 1 chains, 50 residues, 720 atoms, 734 bonds at...>]
        >>> cluster_dict.keys()
        dict_keys(['stacked', 'joined_per_top', 'ensemble', 'series', 'joined'])

        The `'stacked'` key contains all frames stacked along the atom axis.

        >>> cluster_dict["stacked"]  # doctest: +ELLIPSIS
        <mdtraj.Trajectory with 1 frames, 11520 atoms, 800 residues, without unitcells at...>

    Args:
        trajs (TrajEnsemble): A TrajEnsemble of a cluster. See the documentation
            of :func:`encodermap.TrajEnsmeble.cluster` for more info.
        align_string (str): The align string of the parent traj.
        ref_align_string (str): The align string for the reference.
        base_traj (Trajectory, optional): A parent trajectory for when all
            trajs in `trajs` have the same number of atoms. In that case,
            the atomic coordinates can be applied to the base_traj.

    Returns:
        ClusterDict: A dictionary with joined/stacked MDTraj trajectories.

    """
    ds = trajs._CVs
    y = None
    if len(ds) == 1:
        col = list(ds.keys())[0]
        cluster_id = ds[col].values
        cluster_id = cluster_id[~np.isnan(cluster_id)]
        cluster_id = np.unique(cluster_id.astype(int))
        assert (
            len(cluster_id) == 1
        ), f"The CV '{col}' has ambiguous cluster_ids: {cluster_id}."
        cluster_id = cluster_id[0]
    else:
        for name, data_var in ds.items():
            x = data_var.values
            x = x[~np.isnan(x)]
            x = np.mod(x, 1)
            if np.all(x == 0):
                col = name
                y = x.copy()
            if np.all(x == 0) and len(np.unique(x)) == 1:
                col = name
                cluster_id = np.unique(x)[0]
                break
        else:
            if y is None:
                raise Exception(
                    f"Could not find a CV with integer values that defines a cluster "
                    f"membership. Make sure to `trajs.load_CVs()` a numpy array with "
                    f"cluster memberships."
                )
            else:
                raise Exception(f"The CV '{col}' has ambiguous cluster_ids: {y}.")

    series: DataFrame = (
        ds.stack({"frame": ("traj_num", "frame_num")})
        .transpose("frame", ...)
        .dropna("frame", how="all")[col]
        .to_pandas()
    )
    assert len(keys := list(series.keys())) == 1
    series = series[keys[0]]

    # if frames have the same xyz, we can join them
    joined = None
    joined_per_top = {}
    all_trajs: list[Trajectory] = []
    if all([t.n_atoms == trajs[0].n_atoms for t in trajs]):
        # superpose all
        for i, traj in enumerate(trajs):
            if traj.top in joined_per_top:
                ref = joined_per_top[traj.top].get_single_frame(0)
            else:
                ref = traj.get_single_frame(0)
            superposed = traj.superpose(
                reference=ref,
                frame=0,
                atom_indices=traj.top.select(align_string),
                ref_atom_indices=ref.top.select(ref_align_string),
            )
            if traj.top in joined_per_top:
                joined_per_top[traj.top] += superposed
            else:
                joined_per_top[traj.top] = superposed
            all_trajs.append(superposed)

        parent_traj = base_traj
        if parent_traj is None:
            parent_traj = all_trajs[0].traj

        # divmod
        try:
            no_of_iters, rest = divmod(
                sum([t.n_frames for t in all_trajs]), parent_traj.n_frames
            )
        except Exception as e:
            raise Exception(
                f"Can not build a dummy trajectory. Maybe you selected the "
                f"wrong cluster num. Here's the original Error: {e}"
            )
        for i in range(no_of_iters + 1):
            if i == 0:
                dummy_traj = copy.deepcopy(parent_traj)
            elif i == no_of_iters:
                dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj)[:rest])
            else:
                dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj))

        # set the xyz
        i = 0
        for traj in all_trajs:
            for frame in traj:
                dummy_traj[0].xyz = frame.xyz

        joined = dummy_traj

    # stack
    for i, traj in enumerate(trajs):
        for j, frame in enumerate(traj):
            if i == 0 and j == 0:
                stacked = copy.deepcopy(frame.traj)
            else:
                stacked = stacked.stack(frame.traj)

    out: ClusterDict = {
        "stacked": stacked,
        "joined_per_top": joined_per_top,
        "ensemble": trajs,
        "series": series,
    }
    if joined is not None:
        out["joined"] = joined

    return out
