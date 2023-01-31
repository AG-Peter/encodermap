# -*- coding: utf-8 -*-
# encodermap/misc/clustering.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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


import copy
import warnings

import matplotlib as mpl
import numpy as np

from .._optional_imports import _optional_import
from .errors import BadError

##############################################################################
# Optional Imports
##############################################################################


md = _optional_import("mdtraj")
ngl = _optional_import("nglview")


################################################################################
# Globals
################################################################################


__all__ = ["gen_dummy_traj", "get_cluster_frames"]


################################################################################
# Public Functions
################################################################################


def _get_joined_trajs(
    trajs, cluster_no, shorten=True, max_frames=-1, col="cluster_membership"
):
    # where can be int or np.ndarray
    if isinstance(cluster_no, (int, np.int_)):
        where = np.where(trajs.CVs[col] == cluster_no)[0]
    else:
        where = cluster_no

    # stride the where check to make calculations faster
    if max_frames != -1:
        idx = np.round(np.linspace(0, len(where) - 1, max_frames)).astype(int)
        where = where[idx]

    # if shorten change where, so that it contains only 10 structures
    if shorten:
        idx = np.round(np.linspace(0, len(where) - 1, 10)).astype(int)
        where = where[idx]
        assert len(where) == 10

    # append to a list
    joined_trajs = []
    for i, point in enumerate(where):
        try:
            joined_trajs.append(trajs.get_single_frame(point).traj)
        except IndexError:
            print(point)
            raise
    return joined_trajs, where


def get_cluster_frames(
    trajs,
    cluster_no,
    align_string="name CA",
    nglview=False,
    stack_atoms=False,
    shorten=False,
    max_frames=-1,
    superpose=True,
    col="cluster_membership",
    subunit="",
    ball_and_stick=False,
    cmap="viridis",
):
    if not isinstance(cluster_no, (int, np.int_)):
        raise NotImplementedError()

    joined_trajs, where = _get_joined_trajs(trajs, cluster_no, shorten, max_frames, col)

    # preset nglview and only compute it if nglview = True
    view = None

    # if the trajs contain more atoms, we remove as much residues, until they have the same number of CA
    if len(set([t.n_atoms for t in joined_trajs])) > 1:
        smallest_n_residues = min([t.n_residues for t in joined_trajs])
        aligns = [
            t.top.select(f"{align_string} and resid < {smallest_n_residues}")
            for t in joined_trajs
        ]
    else:
        aligns = [t.top.select(f"{align_string}") for t in joined_trajs]

    # if superpose superpose the trajs
    if superpose:
        if isinstance(superpose, bool):
            ref_frame = copy.deepcopy(joined_trajs[0])
        else:
            ref_frame = superpose

        try:
            for i, traj in enumerate(joined_trajs):
                joined_trajs[i] = traj.superpose(
                    ref_frame,
                    atom_indices=aligns[i],
                    ref_atom_indices=aligns[0],
                )
        except AttributeError as e:
            raise BadError(
                f"You provided some wrong datatype or a misformatted string into the argument align_string. Here's the original error: {e}"
            )

    if subunit:
        raise NotImplementedError()

    # only stacking possible here
    if not stack_atoms:
        raise Exception(
            "Cannot build a time resolved traj from topologies with differing atom count."
        )
    if nglview:
        if trajs.common_str:
            colors = mpl.cm.get_cmap(cmap).copy()(
                np.linspace(0, 1, len(trajs.common_str))
            )
        else:
            colors = mpl.cm.get_cmap(cmap).copy()(np.linspace(0, 1, len(joined_trajs)))

        component = 0
        for i, (frame, w) in enumerate(zip(joined_trajs, where)):
            if trajs.common_str:
                c = colors[trajs.common_str.index(trajs.get_single_frame(w).common_str)]
            else:
                c = colors[i]
            # c = '0x' + mpl.colors.rgb2hex(c)[1:7].upper()
            c = mpl.colors.rgb2hex(c)
            if i == 0:
                view = ngl.show_mdtraj(frame)
            else:
                view.add_component(frame)
            view.clear_representations(component=component)
            if ball_and_stick:
                view.add_hyperball(selection="backbone", component=component, color=c)
            else:
                view.add_ribbon(selection="backbone", component=component, color=c)
            component += 1

    return view, joined_trajs


def gen_dummy_traj(
    trajs,
    cluster_no,
    align_string="name CA",
    nglview=False,
    stack_atoms=False,
    shorten=False,
    max_frames=-1,
    superpose=True,
    col="cluster_membership",
    subunit="",
    ref_align_string="name CA",
    base_traj=None,
):
    """Makes a dummy traj from an encodermap trajectory which contains
    trajectories with different topology.

    This function takes an encodermap.TrajEnsemble object and returns mdtraj
    trajectories for clustered data. This function can concatenate trajs even
    if the topology of trajecotries in the TrajEnsemble class is different. The
    topology of this dummy traj will be wrong, but the atomic positions are
    correct.

    This function constructs a traj of length cluster_membership.count(cluster_no)
    with the topology of the first frame of this cluster
    (trajs.get_single_frame(cluster_membership.index(cluster_no))) and changes the
    atomic coordinates of this traj based on the other frames in this cluster.

    Note:
        If you have loaded the encodermap functions with the 'no_load'
        backend a second call to this function with the same parameters will
        be faster, because the trajectory frames have been loaded to memory.

    Args:
        trajs (encodermap.TrajEnsemble): The trajs which were clustered.
        cluster_no (Union[int, int, np.ndarray, list]): The cluster_no of the cluster to make the dummy traj from.
            Can be:
            * int or int: The cluster will be found by using the trajs own cluster_membership in the trajs pandas dataframe.
            * np.array or list: If list or np.array is provided multiple clusters are returned and colored according to clsuter_no.
        align_string (str, optional): Use this mdtraj atom selection string to align the frames
            of the dummy traj. Defaults to 'name CA'.
        nglview (bool, optional): Whether to return a tuple of an nglview.view object and the traj
            or not. Defaults to False.
        stack_atoms (bool, optional): Whether to stack all frames into a single frame with
            mutliple structures in it. This option is useful, if you want to
            generate a picture of interpenetrating structures. Defaults to False.
        shorten (bool, optional): Whether to return all structures or just a subset of
            roughly ten structures. Defaults to False.
        max_frames (int, optional): Only return so many frames. If set to -1 all frames will
            be returned. Defaults to -1.
        superpose (Union(bool, mdtraj.Trajectory), optional): Whether the frames of the returned traj should be superposed
            to frame 0 of the traj. If an mdtraj Trajectory is provided this trajectory is used to superpose. Defaults to True.
        subunit (str, optional): When you want to only visualize an ensemble of certain parts of your protein but keep some
            part stationary (see `align_str`), you can provide a mdtraj selection string. This part of the
            protein will only be rendered from the first frame. The other parts will be rendered as an ensemble
            of structures (either along atom (`stack_atoms` = True) or time (`stack_atoms` = False)). Defaults to ''.
        ref_align_string (str, optional): When the type of `superpose` is mdtraj.Trajectory with a different topology
            than `trajs`, you can give a different align string into this argument. Defaults to 'name CA'.
        base_traj (Union[None, mdtraj.Trajectory], optional): An mdtraj.Trajectory that will be set to the coordinates from
            trajs, instead of trajs[0]. Normally, the first traj in `trajs` (trajs[0]) will be used as a base traj.
            It will be extended into the time-direction until it has the desired number of frames (shorten=True; 10,
            max_frames=N, N; etc.). If you don't want to use this traj but something else, you can feed this option
            an mdtraj.Trajectory object. Defaults to None.

    Returns:
        tuple: A tuple containing:

            view (nglview.view): The nglview.view object if nglview == True,
                is None otherwise.
            dummy_traj (mdtraj.Trajectory): The mdtraj trajectory with wrong
                topology but correct atomic positions.

    See also:
        See the render_vmd function in this document
        to render an image of the returned traj.

    """
    if isinstance(cluster_no, (int, np.int_)):
        return _gen_dummy_traj_single(
            trajs,
            cluster_no,
            align_string,
            nglview,
            stack_atoms,
            shorten,
            max_frames,
            superpose,
            col,
            subunit,
            ref_align_string,
            base_traj,
        )
    elif isinstance(cluster_no, (list, np.ndarray)):
        dummy_trajs, views = [], []
        for i in cluster_no:
            v, dt = _gen_dummy_traj_single(
                trajs,
                i,
                align_string,
                nglview,
                stack_atoms,
                shorten,
                max_frames,
                superpose,
                col,
                subunit,
                ref_align_string,
                base_traj,
            )
            dummy_trajs.append(dt)
            views.append(v)
            if stack_atoms:
                dummy_traj = dummy_trajs[0]
                view = views[0]
                for frame in dummy_trajs[1:]:
                    dummy_traj = dummy_traj.stack(frame)
                    if not subunit:
                        view.add_trajectory(dummy_traj)
                    else:
                        raise NotImplementedError("Not yet Implemented")
            else:
                raise NotImplementedError(
                    "Joining along time axes with multiple cluster's doesn't seem to make sense. Make a proposal how to handle this!"
                )
        return views, dummy_trajs
    else:
        raise TypeError(
            f"`cluster_no` must be int or list. You supplied {type(cluster_no)}."
        )


def _gen_dummy_traj_single(
    trajs,
    cluster_no,
    align_string="name CA",
    nglview=False,
    stack_atoms=False,
    shorten=False,
    max_frames=-1,
    superpose=True,
    col="cluster_membership",
    subunit="",
    ref_align_string="name CA",
    base_traj=None,
):
    """Called when only one cluster is needed."""

    joined_trajs, where = _get_joined_trajs(trajs, cluster_no, shorten, max_frames, col)

    # preset nglview and only compute it if nglview = True
    view = None

    # use traj[0] of the trajs list as the traj from which the topology will be used
    # or use base_traj, if provided
    if base_traj is None:
        if isinstance(trajs[0].index, slice):
            parent_traj = md.load(
                trajs.locations[0], top=trajs.top_files[0], stride=trajs[0].index.step
            )[: len(joined_trajs)]
        else:
            parent_traj = md.load(trajs.locations[0], top=trajs.top_files[0])[
                : len(joined_trajs)
            ]
    else:
        parent_traj = base_traj

    # print some info
    if align_string:
        sel = parent_traj.top.select(align_string)
        print(
            f"Provided alignment string results in {len(sel)} atoms. First atom is {parent_traj.top.atom(sel[0])}. Last atom is {parent_traj.top.atom(sel[-1])}."
        )

    # join the correct number of trajs
    # by use of the divmod method, the frames parent_traj traj will be
    # appended for a certain amount, until the remainder of the division
    # is met by that time, the parent traj will be sliced to fill the correct number of frames
    try:
        no_of_iters, rest = divmod(len(where), parent_traj.n_frames)
    except Exception as e:
        raise Exception(
            f"Can not buid a dummy trajectory. Maybe you selected the wronmg cluster num. Here's the original Error: {e}"
        )
    for i in range(no_of_iters + 1):
        if i == 0:
            dummy_traj = copy.deepcopy(parent_traj)
        elif i == no_of_iters:
            dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj)[:rest])
        else:
            dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj))

    # some checks
    assert len(where) == dummy_traj.n_frames
    assert len(where) == len(joined_trajs)

    # change the xyz coordinates of dummy_traj according to the frames in joined trajs
    for i, traj in enumerate(joined_trajs):
        dummy_traj.xyz[i] = traj.xyz

    # if superpose superpose the trajs
    if superpose:
        if isinstance(superpose, bool):
            ref_frame = copy.deepcopy(dummy_traj[0])
        else:
            ref_frame = superpose
        try:
            _ = dummy_traj.superpose(
                ref_frame,
                atom_indices=dummy_traj.top.select(align_string),
                ref_atom_indices=ref_frame.top.select(ref_align_string),
            )
        except AttributeError as e:
            raise BadError(
                f"You provided some wrong datatype or a misformatted string into the argument align_string. Here's the original error: {e}"
            )

    # if stack_atoms is true overwrite dummy_traj
    if stack_atoms:
        tmp_ = copy.deepcopy(dummy_traj)
        for i, frame in enumerate(tmp_):
            if i == 0:
                dummy_traj = copy.deepcopy(frame)
            else:
                if subunit:
                    sel_all = frame.top.select("all")
                    sel_subunit = frame.top.select(subunit)
                    not_subunit = np.setdiff1d(sel_all, sel_subunit)
                    frame = frame.atom_slice(not_subunit)
                dummy_traj = dummy_traj.stack(frame)

    # make nglview
    if nglview and not subunit:
        view = ngl.show_mdtraj(dummy_traj)
    if nglview and subunit:
        warnings.simplefilter("ignore")
        view = ngl.show_mdtraj(tmp_[0])
        view.clear_representations()
        view.add_representation("cartoon", color="red")
        for frame in tmp_[1:]:
            sel_all = frame.top.select("all")
            sel_subunit = frame.top.select(subunit)
            not_subunit = np.setdiff1d(sel_all, sel_subunit)
            frame = frame.atom_slice(not_subunit)
            try:
                _ = view.add_trajectory(frame)
            except KeyError as e:
                # for some reason it doen't want some atoms. Let's remove them.
                for a in frame.top.atoms:
                    if str(a) == str(e):
                        offender_index = a.index
                        break
                without_offender = np.setdiff1d(
                    frame.top.select("all"), [offender_index]
                )
                frame = frame.atom_slice(without_offender)
                _ = view.add_trajectory(frame)

    # return
    warnings.simplefilter("default")
    return view, dummy_traj


def rmsd_centroid_of_cluster(traj, parallel=True, atom_indices=None):
    """Computes the rmsd centroid of a trajectory.

    Args:
        traj (mdtraj.Trajectory): The trajectory.

    Returns:
        tuple: A tuple containing:

            index (int): The index of the centroid.
            distances (np.ndarray): The RMSD distance matrix with shape
                traj.n_frames x traj.n_frames
            centroid (mdtraj.Trajectory): The traj of the centroid.

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
