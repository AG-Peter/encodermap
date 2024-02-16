#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/long_training.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
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
"""This script download simulation data from M1 connected Ubiquitin dimers
 from the University of Konstanz's data repository
and runs a long training using the new EncoderMap version. This script can
be used as a standalone to run these trainings on faster
computers (HPC, render farms, etc.).

"""
################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy
import os
import pkgutil
import shutil
import warnings
from pathlib import Path
from typing import Literal, Optional

# Third Party Imports
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import scipy
from click import Option, UsageError, command, option
from rich.console import Console
from tqdm.rich import tqdm


################################################################################
# Helpers
################################################################################


class MutuallyExclusiveOption(Option):
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        help = kwargs.get("help", "")
        if self.mutually_exclusive:
            ex_str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = help + (
                " NOTE: This argument is mutually exclusive with "
                " arguments: [" + ex_str + "]."
            )
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(self.name, ", ".join(self.mutually_exclusive))
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(ctx, opts, args)


################################################################################
# tf1/tf2 functions
################################################################################


def _main_tf1(
    console: Console,
    dataset: Literal["linear_dimers", "two_state"],
    output_dir: Optional[str | Path] = None,
    overwrite: bool = False,
    total_steps: int = 50_000,
) -> None:
    if dataset != "linear_dimers":
        raise Exception(f"Dataset '{dataset}' currently not available for tf1.")
    console.log(f"Importing the tf1 version of EncoderMap.")
    import encodermap.encodermap_tf1 as em  # isort: skip

    # Third Party Imports
    import tensorflow.compat.v1 as tf

    # Encodermap imports
    from encodermap.kondata import get_from_kondata

    tf.disable_eager_execution()

    # get the location of the main encodermap
    package = pkgutil.get_loader("encodermap")
    emfile = package.get_filename()
    molname = dataset

    # download the data
    console.log("Downloading data from KonDATA.")
    if output_dir is None:
        output_dir = Path(emfile).parent.parent / "tests"
        if not output_dir.is_dir():
            raise Exception(
                f"EncoderMap isn't installed in developer mode (pip install -e). "
                f"Clone the EncoderMap repository and reinstall it via "
                f"`pip install -e .`. That way we can create files in the test"
                f"directory ({output_dir}) and run this training"
            )
        output_dir /= f"data/{dataset}"

    get_from_kondata(
        "linea_dimers",
        output_dir,
        mk_parentdir=True,
        silence_overwrite_message=True,
        tqdm_class=tqdm,
    )

    console.log(f"Downloaded {len(list(output_dir.rglob('*')))} files from KonDATA.")

    # load data
    structure_path = output_dir / "01.pdb"
    trajectory_paths = [str(output_dir / f"{i:02d}.xtc") for i in range(1, 13)]
    console.log(
        f"Creating a MDAnalysis Universe from {len(trajectory_paths)} "
        f"trajectories using these files:\n{trajectory_paths}"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uni = mda.Universe(structure_path, trajectory_paths)
    selected_atoms = uni.select_atoms(
        "backbone or name H or name O1 or (name CD and resname PRO)"
    )
    console.log(
        "We now load the data with the MolData class. "
        "This operation is cached and will be faster next time."
    )
    moldata = em.MolData(selected_atoms, cache_path=output_dir / "cache")

    # define parameters
    console.log("Defining parameters")

    # First, we want to train without C_alpha cost.
    # Finally, we want to activate the C_alpha cost to improve the long-range order of the generated conformations
    parameters = em.ADCParameters()
    if overwrite:
        main_path = output_dir / "long_training"
        if main_path.is_dir():
            for fileobj in main_path.rglob("*"):
                if fileobj.is_file():
                    if fileobj.suffix != ".png":
                        fileobj.unlink()
                if fileobj.is_dir():
                    shutil.rmtree(fileobj)
        main_path.mkdir(parents=True, exist_ok=True)
        parameters.main_path = str(main_path)
    else:
        parameters.main_path = em.misc.run_path(output_dir / "runs/{}".format(molname))

    parameters.cartesian_cost_scale = 0
    parameters.cartesian_cost_variant = "mean_abs"
    parameters.cartesian_cost_scale_soft_start = (
        int(total_steps / 10 * 9),
        int(total_steps / 10 * 9) + total_steps // 50,
    )
    parameters.cartesian_pwd_start = (
        1  # Calculate pairwise distances starting form the second backbone atom ...
    )
    parameters.cartesian_pwd_step = (
        3  # for every third atom. These are the C_alpha atoms
    )

    parameters.dihedral_cost_scale = 1
    parameters.dihedral_cost_variant = "mean_abs"

    parameters.distance_cost_scale = 0  # no distance cost in dihedral space
    parameters.cartesian_distance_cost_scale = (
        100  # instead, we use distance cost in C_alpha distance space
    )
    parameters.cartesian_dist_sig_parameters = [400, 10, 5, 1, 2, 5]

    parameters.checkpoint_step = max(1, int(total_steps / 10))
    parameters.l2_reg_constant = 0.001
    parameters.center_cost_scale = 0
    parameters.id = molname

    # save a picture with mapped distances
    picture_file = Path(parameters.main_path) / "distance_mapping.png"
    console.log(
        f"Saving a picture demonstrating how distances are mapped "
        f"to {picture_file}."
    )
    console.log(
        f"For that we need to calculate the pairwise distances of the "
        f"input CA coordinates with tensorflow."
    )
    CA_pos = moldata.central_cartesians[
        ::1000, parameters.cartesian_pwd_start :: parameters.cartesian_pwd_step
    ]
    console.log(
        f"The array of CA positions has the shape {CA_pos.shape}. The positions "
        f"of the first three CA-atoms in the first three frames are:\n{CA_pos[:3, :3]}."
    )
    with tf.Graph().as_default():
        pwd = em.misc.pairwise_dist(CA_pos, flat=True)
        with tf.Session() as sess:
            pwd = sess.run(pwd)
    console.log(
        f"The pairwise distances have a shape of {pwd.shape}. This shape arises "
        f"from the number of simulation frames (n_frames = {pwd.shape[0]}), "
        f"the number of CA atoms (n_CA = {CA_pos.shape[1]}), which are used to "
        f"construct n pairwise distances, where n is nCr(n_CA, 2) = {pwd.shape[1]}. "
        f"The first 5 frames of the first 5 pairwise distances are:\n{pwd[:5, :5]}"
    )
    axes = em.plot.distance_histogram(
        pwd, float("inf"), parameters.cartesian_dist_sig_parameters
    )
    plt.savefig(picture_file)
    assert picture_file.is_file()
    console.log(f"Display the file with:\n$ display {picture_file}")

    # Get references from dummy model
    console.log(
        f"Training a dummy model to get references for the angle_cost, dihedral_cost,"
        f"and cartesian_cost."
    )
    dummy_parameters = copy.deepcopy(parameters)
    dummy_parameters.main_path = em.misc.create_dir(
        os.path.join(parameters.main_path, "dummy")
    )
    dummy_parameters.n_steps = int(len(moldata.dihedrals) / parameters.batch_size)
    dummy_parameters.summary_step = 1

    e_map = em.AngleDihedralCartesianEncoderMapDummy(dummy_parameters, moldata)
    e_map.train()
    e_map.close()
    e_map = None

    costs = em.misc.read_from_log(
        os.path.join(dummy_parameters.main_path, "train"),
        ["cost/angle_cost", "cost/dihedral_cost", "cost/cartesian_cost"],
    )
    means = []
    for values in costs:
        means.append(np.mean([i.value for i in values]))
    parameters.angle_cost_reference = means[0]
    parameters.dihedral_cost_reference = means[1]
    parameters.cartesian_cost_reference = means[2]
    console.log(
        f"The cost references came out to be:\n"
        f"angle_cost_reference: {means[0]}\n"
        f"dihedral_cost_reference: {means[1]}\n"
        f"cartesian_cost_reference: {means[2]}\n"
    )
    np.savetxt(
        os.path.join(dummy_parameters.main_path, "adc_cost_means.txt"), np.array(means)
    )

    # run training
    console.log("Will now run a training. First without the CA cost.")
    parameters.n_steps = parameters.cartesian_cost_scale_soft_start[0]
    e_map = em.AngleDihedralCartesianEncoderMap(parameters, moldata)
    e_map.train()
    e_map.close()
    e_map = None

    console.log("Now with the CA cost.")
    parameters.n_steps = total_steps - parameters.cartesian_cost_scale_soft_start[0]
    parameters.cartesian_cost_scale = 1
    ckpt_path = os.path.join(
        parameters.main_path,
        "checkpoints",
        "step{}.ckpt".format(parameters.cartesian_cost_scale_soft_start[0]),
    )

    e_map = em.AngleDihedralCartesianEncoderMap(
        parameters, moldata, checkpoint_path=ckpt_path
    )
    e_map.train()

    # project
    picture_file = Path(parameters.main_path) / "lowd_projection.png"
    projected = e_map.encode(moldata.dihedrals)
    H, xedges, yedges = np.histogram2d(*projected.T, bins=500)
    plt.close("all")
    ax = plt.imshow(
        -np.log(H.T),
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    cbar = plt.colorbar()
    cbar.set_label("-ln(p)", labelpad=0)
    plt.title("Lowd projection")
    e_map.close()
    e_map = None
    plt.savefig(picture_file)
    console.log(f"Find a low-dimensional projection at: {picture_file}.")
    console.log("Finished.")


def _main_tf2(
    console: Console,
    dataset: Literal["linear_dimers", "two_state"],
    output_dir: Optional[str | Path] = None,
    overwrite: bool = False,
    total_steps: int = 50_000,
) -> None:
    if dataset not in ["linear_dimers", "two_state"]:
        raise Exception(f"Dataset '{dataset}' currently not available for tf1.")
    console.log(f"Importing the tf2 version of EncoderMap.")
    import encodermap as em  # isort: skip

    # download the data
    console.log("Downloading data from KonDATA.")
    if output_dir is None:
        if output_dir is None:
            output_dir = Path(em.__file__).parent.parent / "tests"
            if not output_dir.is_dir():
                raise Exception(
                    f"EncoderMap isn't installed in developer mode (pip install -e). "
                    f"Clone the EncoderMap repository and reinstall it via "
                    f"`pip install -e .`. That way we can create files in the test"
                    f"directory ({output_dir}) and run this training"
                )
            output_dir /= f"data/{dataset}"

    em.get_from_kondata(
        dataset,
        output_dir,
        mk_parentdir=True,
        silence_overwrite_message=True,
        tqdm_class=tqdm,
    )

    console.log(f"Downloaded {len(list(output_dir.rglob('*')))} files from KonDATA.")

    # create trajs
    console.log("Loading MD data.")
    trajs_file = output_dir / "trajs.h5"
    trajs = em.load(trajs_file)
    console.log(
        f"Loaded a TrajectoryEnsemble with {trajs.n_frames} total frames from file {trajs_file}."
    )

    # define parameters
    console.log("Defining parameters.")
    if overwrite:
        main_path = output_dir / "long_training"
        if main_path.is_dir():
            for fileobj in main_path.rglob("*"):
                if fileobj.is_file():
                    if fileobj.suffix != ".png":
                        fileobj.unlink()
                if fileobj.is_dir():
                    shutil.rmtree(fileobj)
        main_path.mkdir(parents=True, exist_ok=True)
    else:
        main_path = em.misc.run_path(output_dir / "runs/{}".format(dataset))
    parameters = dict(
        n_steps=total_steps,
        main_path=main_path,
        cartesian_cost_scale=1,
        cartesian_cost_variant="mean_abs",
        cartesian_cost_scale_soft_start=(
            int(total_steps / 10 * 9),
            int(total_steps / 10 * 9) + total_steps // 50,
        ),
        cartesian_pwd_start=1,
        cartesian_pwd_step=3,
        dihedral_cost_scale=1,
        dihedral_cost_variant="mean_abs",
        distance_cost_scale=0,
        cartesian_distance_cost_scale=100,
        cartesian_dist_sig_parameters=[40, 10, 5, 1, 2, 5],
        checkpoint_step=max(1, int(total_steps / 10)),
        l2_reg_constant=0.001,
        center_cost_scale=0,
        tensorboard=True,
    )
    parameters = em.ADCParameters(**parameters)
    console.log(f"Created these parameters:\n{str(parameters)}")

    # save a picture with mapped distances
    picture_file = Path(parameters.main_path) / "distance_mapping.png"
    console.log(
        f"Saving a picture demonstrating how distances are "
        f"mapped to {picture_file}."
    )
    console.log(
        f"For that we need to calculate the pairwise distances of "
        f"the input CA coordinates with scipy."
    )
    CA_pos = trajs.CVs["central_cartesians"][
        ::1000, parameters.cartesian_pwd_start :: parameters.cartesian_pwd_step
    ]
    console.log(
        f"The position of the first three CA-atoms in the first three frames "
        f"are: {trajs._CVs.central_cartesians[0, :3, :3]}."
    )
    pwd = []
    for i in CA_pos:
        pwd.append(scipy.spatial.distance.pdist(i))
    pwd = np.vstack(pwd)
    console.log(
        f"The pairwise distances have a shape of {pwd.shape}. This shape arises "
        f"from the number of simulation frames (n_frames = {pwd.shape[0]}), "
        f"the number of CA atoms (n_CA = {CA_pos.shape[1]}), which are used to "
        f"construct n pairwise distances, where n is nCr(n_CA, 2) = {pwd.shape[1]}. "
        f"The first 5 frames of the first 5 pairwise distances are:\n{pwd[:5, :5]}"
    )
    ax = em.plot.distance_histogram(
        pwd, float("inf"), parameters.cartesian_dist_sig_parameters
    )
    plt.savefig(picture_file)
    assert picture_file.is_file()
    console.log(f"Display the file with:\n$ display {picture_file}")

    # In the tf2 version of EncoderMap, training for references and cartesian
    # training are combined.
    console.log(f"Creating an AngleDihedralCartesianEncoderMap instance.")
    e_map = em.AngleDihedralCartesianEncoderMap(
        trajs=trajs,
        parameters=parameters,
        read_only=False,
    )
    e_map.add_images_to_tensorboard(image_step=total_steps // 50)
    e_map.train()

    # project
    picture_file = Path(parameters.main_path) / "lowd_projection.png"
    projected = e_map.encode(trajs.central_dihedrals)
    H, xedges, yedges = np.histogram2d(*projected.T, bins=500)
    plt.close("all")
    ax = plt.imshow(
        -np.log(H.T),
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    cbar = plt.colorbar()
    cbar.set_label("-ln(p)", labelpad=0)
    plt.title("Lowd projection")
    plt.savefig(picture_file)
    console.log(f"Find a low-dimensional projection at: {picture_file}.")
    console.log("Finished.")


################################################################################
# Main
################################################################################


@command()
@option(
    "--output-dir",
    default=None,
    help=(
        "In which directory to put the data-files and the training. Defaults to "
        "EncoderMap's git repository in the subdirectory tests/data/{dataset}."
    ),
)
@option(
    "-tf1",
    help="Use the tensorflow1 version of EncoderMap.",
    is_flag=True,
    default=False,
    show_default=True,
)
@option(
    "--disable-gpu",
    help="Disable the GPU by setting the appropriate env variable.",
    is_flag=True,
    default=False,
    show_default=True,
)
@option(
    "--overwrite/--no-overwrite",
    "-o/-no",
    help="Overwrite the existing directory.",
    is_flag=True,
    default=False,
    show_default=True,
)
@option(
    "--steps",
    "-s",
    help="The number of steps to run the training for. Defaults to 50,000.",
    type=int,
    default=50_000,
)
@option(
    "--dataset",
    "-ds",
    help="The dataset to use. Defaults to the linear dimer one.",
    type=str,
    default="linear_dimers",
)
def main(
    output_dir: Optional[str | Path] = None,
    tf1: bool = False,
    disable_gpu: bool = False,
    overwrite: bool = False,
    steps: int = 50_000,
    dataset: str = "linear_dimers",
) -> None:
    tf_version = "tf2" if not tf1 else "tf1"
    console = Console()
    with console.status(
        f"Running a long EncoderMap using {tf_version} with the {dataset} dataset for {steps} steps."
    ) as status:
        if disable_gpu:
            console.log(f"Disabling the GPU.")
            # Standard Library Imports
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if not tf1:
            return _main_tf2(
                console,
                dataset,
                output_dir,
                overwrite,
                total_steps=steps,
            )
        else:
            return _main_tf1(
                console,
                dataset,
                output_dir,
                overwrite,
                total_steps=steps,
            )


if __name__ == "__main__":
    main()
