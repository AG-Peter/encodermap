# -*- coding: utf-8 -*-
# tests/test_tf1_tf2_deterministic.py
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


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy
import functools
import os
import re
import unittest
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Union

# Third Party Imports
import mdtraj as md
import numpy as np
import tensorflow as tf
import xarray as xr
from mdtraj.geometry.dihedral import indices_phi, indices_psi
from rich.console import Console
from tensorflow.python.framework.errors_impl import DataLossError
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm

# Encodermap imports
from conftest import expensive_test
from encodermap.encodermap_tf1.backmapping import chain_in_plane as chain_in_plane_tf1
from encodermap.encodermap_tf1.backmapping import (
    dihedrals_to_cartesian_tf as dihedral_to_cartesian_tf1,
)
from encodermap.encodermap_tf1.backmapping import (
    guess_amide_H,
    guess_amide_O,
    merge_cartesians,
)
from encodermap.encodermap_tf1.misc import pairwise_dist
from encodermap.encodermap_tf1.misc import variable_summaries as variable_summaries_tf1
from encodermap.misc.rotate import mdtraj_rotate
from encodermap.models.models import (
    MyBiasInitializer,
    MyKernelInitializer,
    gen_functional_model,
)
from encodermap.trajinfo.info_single import Capturing


try:
    # Local Folder Imports
    from .test_autoencoder import for_all_test_methods, log_successful_test
except ImportError:
    # Encodermap imports
    from test_autoencoder import for_all_test_methods, log_successful_test

try:
    # Local Folder Imports
    from .test_featurizer import assert_allclose_periodic
except ImportError:
    # Encodermap imports
    from test_featurizer import assert_allclose_periodic


################################################################################
# Helper
################################################################################


def check_matching_keys(
    dict_a: dict[str, np.ndarray],
    dict_b: dict[str, np.ndarray],
    exc: Exception,
    key: str = "gradients",
) -> str:
    # find keys in dummy_e_map_tf1._accumulated_gradients and deterministic_dummy_training_tf1
    for i, (key1, item1) in enumerate(dict_a.items()):
        for j, (key2, item2) in enumerate(dict_b.items()):
            if key not in key2:
                continue
            if item1.shape != item2.shape:
                continue
            if np.allclose(item1, item2):
                return (
                    f"Don't know, why {str(exc)} was raised. I found matching values at:\n\n"
                    f"`dict_a`: {key1=}, `dict_b` {key2=}."
                )
    else:
        raise Exception(
            f"There are no keys (without '{key}' in `dict_a`, that match any values in `dict_b`."
        ) from exc


def equal_array_of_tuples(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
) -> bool:
    """Compares a sequence of np.ndarray and returns whether the arrays in the sequence are identical.

    Args:
        a (Sequence[np.ndarray]): The first sequence of np.ndarrays.
        b (Sequence[np.ndarray]): The 2nd sequence of np.ndarrays.

    Returns:
        bool: Whether the arrays in the sequence are equal.

    """
    assert len(a) == len(b)
    for i, j in zip(a, b):
        if not np.array_equal(i, j):
            return False
    return True


def data_loss_iterator(gen: Generator) -> Generator:
    """Takes a Generator, that might raise StopIteration, DataLossError and catches it and breaks the loop."""
    iter_ = 0
    while True:
        try:
            yield next(gen)
        except (StopIteration, DataLossError):
            print(f"Received loop-breaking exception after {iter_} steps.")
            break
        iter_ += 1


def _synchronize_deterministic_gens(
    gen1: Generator,
    gen2: Generator,
    testclass: tf.test.TestCase,
) -> None:
    """Takes two generators (tf.data.Dataset), that should return deterministic samples, checks them with the help
    of `testclass` and synchronizes them, should they return asynchronous samples.

    Args:
        gen1 (Generator): The first generator.
        gen2 (Generator): The 2nd generator.
        testclass (tf.test.TestCase): Instance of the TestCase class,
            so, that `testclass.assertAllEqual can be used.`

    """
    ds1_take2 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ]
    )
    ds1_take1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    ds2_take1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    ds2_take2 = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    g1 = list(gen1.take(3).as_numpy_iterator())
    g2 = list(gen2.take(3).as_numpy_iterator())
    matrix = np.empty((3, 3))
    for i, a in enumerate(g1):
        for j, b in enumerate(g2):
            matrix[i, j] = equal_array_of_tuples(a, b)
    testclass.assertEqual(g1[0][0].shape, g2[1][0].shape)
    testclass.assertNotAllEqual(
        g1[0][0], g2[1][0], msg="The generators are returning always the same batch."
    )
    testclass.assertNotAllEqual(
        g1[1][0], g2[2][0], msg="The generators are returning always the same batch."
    )
    testclass.assertNotAllEqual(
        g1[1][1], g2[2][1], msg="The generators are returning always the same batch."
    )
    if np.array_equal(matrix, np.identity(3)):
        testclass.console.log(
            f"The generators at {gen1=} and {gen2=} are IN SYNC.", style="bold magenta"
        )
        pass
    elif np.array_equal(matrix, ds2_take2):
        testclass.console.log(
            f"The generators at {gen1=} and {gen2=} are OUT OF SYNC. gen1 is ahead by 2.",
            style="bold magenta",
        )
        gen2.take(2)
    elif np.array_equal(matrix, ds2_take1):
        testclass.console.log(
            f"The generators at {gen1=} and {gen2=} are OUT OF SYNC. gen1 is ahead by 1.",
            style="bold magenta",
        )
        gen2.take(1)
    elif np.array_equal(matrix, ds1_take1):
        testclass.console.log(
            f"The generators at {gen1=} and {gen2=} are OUT OF SYNC. gen2 is ahead by 1.",
            style="bold magenta",
        )
        gen1.take(1)
    elif np.array_equal(matrix, ds1_take2):
        testclass.console.log(
            f"The generators at {gen1=} and {gen2=} are OUT OF SYNC. gen2 is ahead by 2.",
            style="bold magenta",
        )
        gen1.take(2)
    else:
        raise Exception(f"Gens out of sync:\n{matrix}")

    for i, (ds1, ds2) in enumerate(
        zip(gen1.as_numpy_iterator(), gen2.as_numpy_iterator())
    ):
        if i >= 5:
            break
        for j, (d1, d2) in enumerate(zip(ds1, ds2)):
            testclass.assertAllEqual(
                d1,
                d2,
                msg=(
                    f"Arrays differ at iteration {i=}. There is currently no way to resolve this issue of desyncing "
                    f"tf.data.Datasets. Running the test multiple times results in a 3 out of 5 success rate. The "
                    f"dataset already uses the `options = tf.data.Options(); options.deterministic = True; "
                    f"dataset = dataset.with_options(options)` code snipped found in https://www.tensorflow.org/api_docs/python/tf/data/Options "
                    f"but still produces this error. Feel free to raise this error over at tensorflow. "
                    f"THIS IS INSIDE THE _synchronize_deterministic_gens FUNCTION WHICH WAS MEANT TO FIX THIS ISSUE."
                ),
                success_msg=(
                    f"Inside the `_synchronize_deterministic_gens` functions at iteration {i=}. the data of shape {d1.shape=} {d2.shape=} "
                    f"are identical."
                ),
            )


def _create_artificial_two_state_system(
    output_dir: Union[str, Path],
    output_dir_make: bool = False,
    pdb_id: str = "linear_dimers",
    select_chain: int = 0,
    initial_dihedral: Optional[tuple[int, int, int, int]] = None,
    state_2_rotation: float = 90.0,
    factor: float = 0.001,
    variation_method: Literal[
        "gaussian_wiggle", "gaussian_dihedrals"
    ] = "gaussian_dihedrals",
    n_frames: int = 5000,
    early_stop: bool = False,
) -> None:
    # make output dir
    output_dir = Path(output_dir)
    if output_dir_make:
        output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.is_dir(), (
        f"The output directory {output_dir} does not exist. Set `output_dir_make` "
        f"to True to create it."
    )

    # define the url and load
    if pdb_id == "linear_dimers":
        # Encodermap imports
        from encodermap.kondata import get_from_kondata

        _output_dir = Path(
            get_from_kondata(
                "linear_dimers", mk_parentdir=True, silence_overwrite_message=True
            )
        )
        pdb_file = _output_dir / "01.pdb"
        assert pdb_file.is_file()
        state1 = md.load(str(pdb_file))
    else:
        url = f"https://files.rcsb.org/view/{pdb_id.upper()}.pdb"
        state1 = md.load_pdb(url)

    # maybe remove some chains
    sel = state1.top.select(f"chainid {select_chain}")
    try:
        state1 = state1.atom_slice(sel)
    except Exception as e:
        raise Exception(f"{sel=}") from e

    # define the initial dihedral
    if initial_dihedral is None and pdb_id == "1AO6":
        initial_dihedral = [2386, 2390, 2391, 2392]
    elif initial_dihedral is None and pdb_id == "linear_dimers":
        initial_dihedral = [759, 761, 763, 768]
    elif pdb_id != "1AO6" and initial_dihedral is None:
        raise Exception(
            f"Please specify the 0-based indices of the atoms that make up the "
            f"dihedral to rotate the structure around and create a second state."
        )
    dihedral_atoms = np.array([initial_dihedral])

    # crate state2 by rotation
    state2 = deepcopy(
        mdtraj_rotate(state1, [state_2_rotation], dihedral_atoms, deg=True)
    )

    if early_stop:
        print(
            f"Early stopping and saving: "
            f"{output_dir / 'state1.pdb'}"
            f"{output_dir / 'state2.pdb'}"
        )
        state1.save_pdb(str(output_dir / "state1.pdb"))
        state2.save_pdb(str(output_dir / "state2.pdb"))
        return

    # vary the positions by the variation method
    if variation_method == "gaussian_wiggle":
        for i in range(n_frames):
            frame1 = deepcopy(state1)
            frame2 = deepcopy(state2)
            frame1.xyz += np.random.normal(0, factor * 50, size=frame1.xyz.shape)
            frame2.xyz += np.random.normal(0, factor * 50, size=frame2.xyz.shape)
            if i == 0:
                frame1.save_pdb(str(output_dir / "state1.pdb"))
                frame2.save_pdb(str(output_dir / "state2.pdb"))
                traj1 = deepcopy(frame1)
                traj2 = deepcopy(frame2)
            else:
                traj1 = traj1.join(deepcopy(frame1))
                traj2 = traj2.join(deepcopy(frame2))
        else:
            traj1.save_xtc(str(output_dir / "state1.xtc"))
            traj2.save_xtc(str(output_dir / "state2.xtc"))
    else:
        psi_indices = indices_psi(state1.top)
        phi_indices = indices_phi(state1.top)
        indices = np.hstack(
            [psi_indices, phi_indices],
        ).reshape((-1,) + psi_indices.shape[1:])
        existing_angles_state1 = md.compute_dihedrals(state1, indices)
        existing_angles_state2 = md.compute_dihedrals(state2, indices)
        traj1 = mdtraj_rotate(
            state1,
            angles=existing_angles_state1
            + np.random.normal(
                -np.pi * factor,
                np.pi * factor,
                size=(n_frames, len(indices)),
            ),
            indices=indices,
            drop_proline_angles=True,
            delete_sulfide_bridges=True,
        )
        traj2 = mdtraj_rotate(
            state2,
            angles=existing_angles_state2
            + np.random.normal(
                -np.pi * factor,
                np.pi * factor,
                size=(n_frames, len(indices)),
            ),
            indices=indices,
            drop_proline_angles=True,
            delete_sulfide_bridges=True,
        )
        traj1[0].save_pdb(str(output_dir / "state1.pdb"))
        traj2[0].save_pdb(str(output_dir / "state2.pdb"))
        traj1.save_xtc(str(output_dir / "state1.xtc"))
        traj2.save_xtc(str(output_dir / "state2.xtc"))


def _setup_network_dummy(self):
    self.inputs = self.data_iterator.get_next()
    if self.p.use_backbone_angles:
        self.main_inputs = tf.concat([self.inputs[0], self.inputs[1]], axis=1)
    else:
        self.main_inputs = self.inputs[1]
    self.main_inputs = tf.compat.v1.placeholder_with_default(
        self.main_inputs, self.main_inputs.shape
    )
    self.regularizer = tf.keras.regularizers.l2(self.p.l2_reg_constant)
    encoded = self._encode(self.main_inputs)
    self.latent = tf.compat.v1.placeholder_with_default(encoded, encoded.shape)
    variable_summaries_tf1("latent", self.latent)
    self.generated = self._generate(self.latent)

    self.generated_dihedrals = tf.tile(
        np.expand_dims(np.mean(self.train_moldata.dihedrals, axis=0), axis=0),
        [tf.shape(self.main_inputs)[0], 1],
    )
    self.generated_angles = tf.tile(
        np.expand_dims(np.mean(self.train_moldata.angles, axis=0), axis=0),
        [tf.shape(self.main_inputs)[0], 1],
    )

    mean_lengths = np.expand_dims(np.mean(self.train_moldata.lengths, axis=0), axis=0)
    self.chain_in_plane = chain_in_plane_tf1(mean_lengths, self.generated_angles)
    self.cartesian = dihedral_to_cartesian_tf1(
        self.generated_dihedrals + np.pi, self.chain_in_plane
    )
    tf.compat.v1.summary.tensor_summary("debug/_generated_cartesians", self.cartesian)
    self.amide_H_cartesian = guess_amide_H(
        self.cartesian, self.train_moldata.central_atoms.names
    )
    self.amide_O_cartesian = guess_amide_O(
        self.cartesian, self.train_moldata.central_atoms.names
    )

    self.cartesian_with_guessed_atoms = merge_cartesians(
        self.cartesian,
        self.train_moldata.central_atoms.names,
        self.amide_H_cartesian,
        self.amide_O_cartesian,
    )
    tf.compat.v1.summary.tensor_summary("debug/_input_cartesians", self.inputs[2])
    self.input_cartesian_pairwise_dist = pairwise_dist(
        self.inputs[2][
            :,
            self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
        ],
        flat=True,
    )
    tf.compat.v1.summary.tensor_summary(
        "debug/_input_cartesians_pairwise", self.input_cartesian_pairwise_dist
    )
    self.gen_cartesian_pairwise_dist = pairwise_dist(
        self.cartesian[
            :,
            self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
        ],
        flat=True,
    )
    tf.compat.v1.summary.tensor_summary(
        "debug/_generated_cartesians_pairwise", self.gen_cartesian_pairwise_dist
    )

    self.clashes = tf.compat.v1.count_nonzero(
        pairwise_dist(self.cartesian, flat=True) < 1, axis=1, dtype=tf.float32
    )
    tf.summary.scalar("clashes", tf.reduce_mean(self.clashes))


# this subclass of AngleDihedralCartesianEncoderMap holds some intermediate values
def get_custom_adc_encodermap(
    tf: Any,
    initial_weights: Union[
        Literal["random", "ones", "deterministic"], dict[str, np.ndarray]
    ] = "random",
    check_weights: bool = True,
    dummy: bool = False,
    saver: bool = False,
) -> Any:
    # Encodermap imports
    from encodermap.encodermap_tf1.angle_dihedral_cartesian_encodermap import (
        AngleDihedralCartesianEncoderMap,
        AngleDihedralCartesianEncoderMapDummy,
    )
    from encodermap.encodermap_tf1.misc import add_layer_summaries, variable_summaries

    cls = []
    if dummy:
        cls.append(AngleDihedralCartesianEncoderMapDummy)
    else:
        cls.append(AngleDihedralCartesianEncoderMap)
    cls.insert(0, tf.test.TestCase)

    class CustomADCEncodermap(*cls):
        def __init__(
            self,
            parameters,
            train_data=None,
            validation_data=None,
            checkpoint_path=None,
            n_inputs=None,
            read_only=False,
            seed=None,
            debug=False,
            train_data_len: Union[None, int] = None,
            shapes: Optional[list[Sequence[int]]] = None,
            train_moldata_with_angles: Any = None,
            trajs: Optional[Any] = None,
            initial_step: int = 0,
            **kwargs,
        ):
            self._encoder_layers = {}
            self._called_encoder_layers = {}
            self._decoder_layers = {}
            self._called_decoder_layers = {}
            self._kernel_initializers = []
            assert shapes is not None
            self._shapes = shapes
            assert train_data_len is not None
            self._len = train_data_len
            assert train_moldata_with_angles is not None
            self.train_moldata = train_moldata_with_angles
            assert trajs is not None
            self.trajs = trajs
            self._initial_step = initial_step
            self._accumulated_weights = {}
            self.p = parameters
            self.n_inputs = n_inputs
            # if seed is provided, weights and biases are fixed to ensure reproducibility
            self.seed = seed
            tf.random.set_random_seed(self.seed)
            self.debug = debug
            if not read_only:
                self.p.save()
            print(
                "Output files are saved to {}".format(self.p.main_path),
                "as defined in 'main_path' in the parameters.",
            )

            self.train_data = train_data
            self.validation_data = validation_data

            self._prepare_data()
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.dataset = self.trajs.tf_dataset(
                    self.p.batch_size,
                    deterministic=True,
                    prefetch=False,
                )
                self.global_step = tf.train.create_global_step()

                self._setup_data_iterator()
                if not hasattr(self, "data_iterator"):
                    self.fail(
                        msg=f"'CustomADCEncoderMap' has not attr 'data_iterator'."
                    )

                self._setup_network()

                with tf.name_scope("cost"):
                    self.cost = 0
                    self._setup_cost()
                    tf.summary.scalar("combined_cost", self.cost)

                # Setup Optimizer:
                self.optimizer = tf.train.AdamOptimizer(self.p.learning_rate)
                self.gradients = self.optimizer.compute_gradients(self.cost)
                for i, g in enumerate(self.gradients):
                    if g[0] is not None:
                        tf.summary.tensor_summary(f"gradients_{g[1].name}", g[0])
                self.optimize = self.optimizer.apply_gradients(
                    self.gradients, global_step=self.global_step
                )

                self.merged_summaries = tf.summary.merge_all()

                # Setup Session
                if self.p.gpu_memory_fraction == 0:
                    gpu_options = tf.GPUOptions(allow_growth=True)
                else:
                    gpu_options = tf.GPUOptions(
                        per_process_gpu_memory_fraction=self.p.gpu_memory_fraction
                    )
                self.sess = tf.Session(
                    config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph
                )
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(
                    self.data_iterator.initializer,
                    #     feed_dict={
                    #         p: d for p, d in zip(self.data_placeholders, self.train_data)
                    #     },
                )
                if not read_only:
                    self.train_writer = tf.summary.FileWriter(
                        os.path.join(self.p.main_path, "train"), self.sess.graph
                    )
                    if self.validation_data is not None:
                        self.validation_writer = tf.summary.FileWriter(
                            os.path.join(self.p.main_path, "validation"),
                            self.sess.graph,
                        )
                if initial_weights == "deterministic" and check_weights:
                    weights_should_be = [
                        0.0045964746,
                        -0.032322593,
                        0.06147788,
                        -1.0974659,
                        0.0490791,
                        -0.01832466,
                    ]
                    weights_shapes_should_be = [
                        (906, 128),
                        (128, 128),
                        (128, 2),
                        (2, 128),
                        (128, 128),
                        (128, 906),
                    ]
                    biases_should_be = [
                        0.10608495,
                        0.08391656,
                        0.13059087,
                        0.031738803,
                        0.12442134,
                        0.09088181,
                    ]
                    biases_shapes_should_be = [
                        (128,),
                        (128,),
                        (2,),
                        (128,),
                        (128,),
                        (906,),
                    ]
                    for i, layer in enumerate(
                        list(self._encoder_layers.values())
                        + list(self._decoder_layers.values())
                    ):
                        w, b = self.sess.run(layer.weights)
                        self.assertAlmostEqual(w[0, 0], weights_should_be[i], places=5)
                        self.assertAlmostEqual(b[0], biases_should_be[i], places=5)
                        self.assertEqual(w.shape, weights_shapes_should_be[i])
                        self.assertEqual(b.shape, biases_shapes_should_be[i])

                if saver:
                    self.saver = tf.train.Saver(max_to_keep=100)

        def _prepare_data(self):
            pass

        def _cartesian_cost(self):
            if self.p.cartesian_cost_scale is not None:
                if self.p.cartesian_cost_variant == "mean_square":
                    cartesian_cost = tf.reduce_mean(
                        tf.square(
                            self.input_cartesian_pairwise_dist
                            - self.gen_cartesian_pairwise_dist
                        )
                    )
                elif self.p.cartesian_cost_variant == "mean_abs":
                    cartesian_cost = tf.reduce_mean(
                        tf.abs(
                            self.input_cartesian_pairwise_dist
                            - self.gen_cartesian_pairwise_dist
                        )
                    )
                elif self.p.cartesian_cost_variant == "mean_norm":
                    cartesian_cost = tf.reduce_mean(
                        tf.norm(
                            self.input_cartesian_pairwise_dist
                            - self.gen_cartesian_pairwise_dist,
                            axis=1,
                        )
                    )
                else:
                    raise ValueError(
                        "cartesian_cost_variant {} not available".format(
                            self.p.dihedral_to_cartesian_cost_variant
                        )
                    )
                tf.summary.scalar("debug/_cartesian_cost_raw", cartesian_cost)
                cartesian_cost /= self.p.cartesian_cost_reference
                tf.summary.scalar("cartesian_cost", cartesian_cost)
                tf.summary.scalar("debug/_cartesian_cost_referenced", cartesian_cost)
                if self.p.cartesian_cost_scale != 0:
                    if self.p.cartesian_cost_scale_soft_start[0] is None:
                        self.cost += self.p.cartesian_cost_scale * self.cartesian_cost_r
                    else:
                        a = self.p.cartesian_cost_scale_soft_start[0]
                        b = self.p.cartesian_cost_scale_soft_start[1]
                        cost_scale = tf.case(
                            [
                                (
                                    tf.less(self.global_step + self._initial_step, a),
                                    lambda: tf.constant(0, tf.float32),
                                ),
                                (
                                    tf.greater(
                                        self.global_step + self._initial_step, b
                                    ),
                                    lambda: tf.constant(
                                        self.p.cartesian_cost_scale, tf.float32
                                    ),
                                ),
                            ],
                            default=lambda: self.p.cartesian_cost_scale
                            / (b - a)
                            * (
                                tf.cast(
                                    self.global_step + self._initial_step, tf.float32
                                )
                                - a
                            ),
                        )
                        tf.summary.scalar("cartesian_cost_scale", cost_scale)
                        tf.summary.scalar(
                            "debug/_cartesian_cost_referenced_and_scaled",
                            cost_scale * cartesian_cost,
                        )
                        self.cost += cost_scale * cartesian_cost

        def train(self):
            self._step_ = 0
            gradient_names = []
            for i, name in enumerate(
                list(self._encoder_layers.keys()) + list(self._decoder_layers.keys())
            ):
                if i == 0:
                    gradient_names.append(f"gradients_dense/kernel_0")
                    gradient_names.append(f"gradients_dense/bias_0")
                else:
                    gradient_names.append(f"gradients_dense_{i}/kernel_0")
                    gradient_names.append(f"gradients_dense_{i}/bias_0")
            assert len(gradient_names) == len(self.gradients)
            for _ in tqdm(range(self.p.n_steps)):
                # if fixed seed, we need the summaries at global_step == 1
                # to run unittests on them
                # add stuff to the accumulated dicts
                if self._step_ <= 200 or self._step_ % 10 == 0:
                    i = 1
                    for name, layer in self._encoder_layers.items():
                        w, b = layer.weights
                        w = self.sess.run(w)
                        b = self.sess.run(b)
                        self._accumulated_weights[
                            f"step_{self._step_}_encoder/dense_{i}_weights/values"
                        ] = w
                        self._accumulated_weights[
                            f"step_{self._step_}_encoder/dense_{i}_biases/values"
                        ] = b
                        i += 1
                    for name, layer in self._decoder_layers.items():
                        w, b = layer.weights
                        w = self.sess.run(w)
                        b = self.sess.run(b)
                        self._accumulated_weights[
                            f"step_{self._step_}_decoder/dense_{i}_weights/values"
                        ] = w
                        self._accumulated_weights[
                            f"step_{self._step_}_decoder/dense_{i}_biases/values"
                        ] = b
                        i += 1

                # this line replaces a self.sess.run(self.optimzie)
                # because we want to save summaries for every step
                _, summary_values = self.sess.run(
                    (self.optimize, self.merged_summaries)
                )
                self.train_writer.add_summary(summary_values, self._step_)
                if self.validation_data is not None:
                    summary_values = self.sess.run(
                        self.merged_summaries,
                        feed_dict={
                            self.main_inputs: self._random_batch(self.validation_data)
                        },
                    )
                    self.validation_writer.add_summary(summary_values, self._step_)

                self._step_ += 1
            else:
                if saver:
                    self.saver.save(
                        self.sess,
                        os.path.join(
                            self.p.main_path,
                            "checkpoints",
                            "step{}.ckpt".format(self._step()),
                        ),
                    )
            self.train_writer.flush()

        def _setup_data_iterator(self):
            self.data_iterator = tf.compat.v1.data.make_initializable_iterator(
                self.dataset
            )

        def _encode(self, inputs):
            with tf.name_scope("encoder"):
                if self.p.periodicity < float("inf"):
                    if self.p.periodicity != 2 * np.pi:
                        inputs = inputs / self.p.periodicity * 2 * np.pi
                    self.unit_circle_inputs = tf.concat(
                        [tf.sin(inputs), tf.cos(inputs)], 1
                    )
                    current_layer = self.unit_circle_inputs
                else:
                    current_layer = inputs
                if len(cls) > 1:
                    self.assertEqual(
                        len(self.p.n_neurons),
                        len(self.p.activation_functions) - 1,
                        msg="you need one activation function more then layers given in n_neurons",
                    )
                for i, (n_neurons, act_fun) in enumerate(
                    zip(self.p.n_neurons, self.p.activation_functions[1:])
                ):
                    if act_fun:
                        act_fun = getattr(tf.nn, act_fun)
                    else:
                        act_fun = None
                    variable_summaries(
                        "activation{}".format(i), current_layer, debug=True
                    )
                    if isinstance(initial_weights, str):
                        if initial_weights == "ones":
                            print(
                                "Using Constant(1) for kernel and bias initialization."
                            )
                            dense = tf.layers.Dense(
                                n_neurons,
                                activation=act_fun,
                                kernel_initializer=tf.keras.initializers.Constant(1),
                                kernel_regularizer=self.regularizer,
                                bias_initializer=tf.keras.initializers.Constant(1),
                            )
                        else:
                            if initial_weights == "random":
                                seed = None
                            else:
                                seed = 123456789101112 + i
                            print(f"TF1 Encoder layer {i}, seed={seed}")
                            print(
                                f"Using variance_scaling/random_normal for kernel and bias initialization with seed={seed}"
                            )
                            _kernel_initializer = tf.variance_scaling_initializer(
                                seed=seed
                            )
                            _bias_initializer = tf.random_normal_initializer(
                                0.1, 0.05, seed=seed
                            )
                            dense = tf.layers.Dense(
                                n_neurons,
                                activation=act_fun,
                                kernel_initializer=_kernel_initializer,
                                kernel_regularizer=self.regularizer,
                                bias_initializer=_bias_initializer,
                            )
                    elif isinstance(initial_weights, dict):
                        if i == 0:
                            w_name = "dense/kernel"
                            b_name = "dense/bias"
                        else:
                            w_name = f"dense_{i}/kernel"
                            b_name = f"dense_{i}/bias"
                        _kernel_initializer = MyKernelInitializer(
                            initial_weights[w_name]
                        )
                        _bias_initializer = MyBiasInitializer(initial_weights[b_name])
                        dense = tf.layers.Dense(
                            n_neurons,
                            activation=act_fun,
                            kernel_initializer=_kernel_initializer,
                            kernel_regularizer=self.regularizer,
                            bias_initializer=_bias_initializer,
                        )
                    self._kernel_initializers.append(_kernel_initializer)
                    self._encoder_layers[f"encoder_{i}"] = dense
                    current_layer = dense(current_layer)
                    self._called_encoder_layers[f"encoder_{i}"] = current_layer
                    add_layer_summaries(dense, debug=self.debug)
            return current_layer

        def _generate(self, inputs):
            with tf.name_scope("generator"):
                current_layer = inputs
                if self.p.periodicity < float("inf"):
                    n_neurons_with_inputs = [
                        self.main_inputs.shape[1] * 2
                    ] + self.p.n_neurons
                else:
                    n_neurons_with_inputs = [
                        self.main_inputs.shape[1]
                    ] + self.p.n_neurons
                for i, (n_neurons, act_fun) in enumerate(
                    zip(
                        n_neurons_with_inputs[-2::-1],
                        self.p.activation_functions[-2::-1],
                    )
                ):
                    if act_fun:
                        act_fun = getattr(tf.nn, act_fun)
                    else:
                        act_fun = None
                    if isinstance(initial_weights, str):
                        if initial_weights == "ones":
                            print(
                                "Using Constant(1) for kernel and bias initialization."
                            )
                            dense = tf.layers.Dense(
                                n_neurons,
                                activation=act_fun,
                                kernel_initializer=tf.keras.initializers.Constant(1),
                                kernel_regularizer=self.regularizer,
                                bias_initializer=tf.keras.initializers.Constant(1),
                            )
                        else:
                            if initial_weights == "random":
                                seed = None
                            else:
                                seed = 121110987654321 + i
                            print(f"TF1 Decoder layer {i}, seed={seed}")
                            print(
                                f"Using variance_scaling/random_normal for kernel and bias initialization with seed={seed}"
                            )
                            _kernel_initializer = tf.variance_scaling_initializer(
                                seed=seed
                            )
                            _bias_initializer = tf.random_normal_initializer(
                                0.1, 0.05, seed=seed
                            )
                            dense = tf.layers.Dense(
                                n_neurons,
                                activation=act_fun,
                                kernel_initializer=_kernel_initializer,
                                kernel_regularizer=self.regularizer,
                                bias_initializer=_bias_initializer,
                            )
                    elif isinstance(initial_weights, dict):
                        w_name = f"dense_{i + len(self._encoder_layers)}/kernel"
                        b_name = f"dense_{i + len(self._encoder_layers)}/bias"
                        _kernel_initializer = MyKernelInitializer(
                            initial_weights[w_name]
                        )
                        _bias_initializer = MyBiasInitializer(initial_weights[b_name])
                        dense = tf.layers.Dense(
                            n_neurons,
                            activation=act_fun,
                            kernel_initializer=_kernel_initializer,
                            kernel_regularizer=self.regularizer,
                            bias_initializer=_bias_initializer,
                        )
                    self._kernel_initializers.append(_kernel_initializer)
                    self._decoder_layers[f"decoder_{i}"] = dense
                    current_layer = dense(current_layer)
                    self._called_decoder_layers[f"decoder_{i}"] = current_layer
                if self.p.periodicity < float("inf"):
                    split = self.main_inputs.shape[1]
                    current_layer = tf.atan2(
                        current_layer[:, :split], current_layer[:, split:]
                    )
                    if self.p.periodicity != 2 * np.pi:
                        current_layer = current_layer / (2 * np.pi) * self.p.periodicity
                self._called_decoder_layers[f"decoder_{i}"] = current_layer
            return current_layer

    if dummy:
        CustomADCEncodermap._setup_network = _setup_network_dummy
    else:
        pass
        # CustomADCEncodermap._setup_network = _setup_network_not_dummy
    return CustomADCEncodermap


def retry_due_to_out_of_sync_datasets(msg_regex: str, n_times: int = 3) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while True:
                try:
                    out = func(*args, **kwargs)
                except Exception as e:
                    if not re.search(msg_regex, str(e)):
                        raise e
                    if retries >= n_times:
                        args[0].fail(
                            f"Ran the test {func.__name__} {retries} times, "
                            f"but it still fails. This is the last error:\n\n{e}"
                        )
                    else:
                        args[0].console.log(
                            f"The test {func.__name__} failed. I will retry "
                            f"{n_times - retries} times. This is the last error:"
                            f"\n\n{e}"
                        )
                        retries += 1
                else:
                    return out

        return wrapper

    return decorator


################################################################################
# Test suites
################################################################################


@for_all_test_methods(log_successful_test)
class TestTf1Tf2Deterministic(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.console = Console(width=150)

    def assertAllClosePeriodic(
        self,
        a: np.ndarray,
        b: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 0.0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        periodicity: float = 2 * np.pi,
    ) -> None:
        (a, b) = self.evaluate_if_both_tensors(a, b)
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        try:
            assert_allclose_periodic(
                actual=a,
                desired=b,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                err_msg=err_msg,
                verbose=verbose,
                periodicity=periodicity,
            )
        except AssertionError as e:
            self.fail(str(e))

    @expensive_test
    def test_deterministic_training(self):
        # self.calculate_loss_and_gradients_from_dataset(
        #     "linear_dimers",
        #     "deterministic",
        #     "deterministic",
        #     "train_and_check",
        #     "train_and_check",
        #     "train_and_check",
        #     "train_and_check",
        #     "train_and_check",
        #     "train_and_check",
        # )

        self.calculate_loss_and_gradients_from_dataset(
            "two_state",
            "deterministic",
            "deterministic",
            "skip",
            "skip",
            "skip",
            "skip",
            "skip",
            "train_and_check",
        )

    @retry_due_to_out_of_sync_datasets(msg_regex=r".*found.*")
    def calculate_loss_and_gradients_from_dataset(
        self,
        dataset: Literal[
            "linear_dimers", "TrpCage", "two_state", "random"
        ] = "linear_dimers",
        initial_weights: Literal["ones", "random", "deterministic"] = "deterministic",
        dataset_selection: Literal["deterministic", "random"] = "deterministic",
        train_tf1_dummy: Literal[
            "train_and_save", "train_and_check", "skip"
        ] = "train_and_check",
        train_tf2_dummy: Literal["train_and_check", "skip"] = "train_and_check",
        train_tf1_wo_cartesians: Literal[
            "train_and_save", "train_and_check", "skip"
        ] = "train_and_check",
        train_tf2_wo_cartesians: Literal["train_and_check", "skip"] = "train_and_check",
        train_tf1_w_cartesians: Literal[
            "train_and_save", "train_and_check", "skip"
        ] = "train_and_check",
        train_tf2_w_cartesians: Literal[
            "train_and_save", "train_and_check", "skip"
        ] = "train_and_check",
        save_tf1_checkpoints: bool = False,
    ):
        """This is a long test, that ensures compatibility between the tf1 and tf2
        versions of EncoderMap.

        """
        # import the tf1 version
        # Encodermap imports
        import encodermap as em
        import encodermap.encodermap_tf1 as em_tf1
        from encodermap.kondata import get_from_kondata

        # disable tf2 stuff for tf1 checkpoint_saver
        if save_tf1_checkpoints:
            train_tf2_dummy = "skip"
            train_tf2_wo_cartesians = "skip"
            train_tf2_w_cartesians = "skip"
            tf.compat.v1.disable_eager_execution()

        # some globals
        COST_REFERENCES = {
            "linear_dimers": {
                "cartesian_cost": 0.8194258863275702,
                "angle_cost": 0.046510757031765854,
                "dihedral_cost": 0.8560819138180126,
            },
            "two_state": {
                "cartesian_cost": 0.7904957471749722,
                "angle_cost": 0.003593107745146904,
                "dihedral_cost": 0.04003295340599158,
            },
        }

        # these npz files hold intermediate data
        output_dir = Path(__file__).resolve().parent / f"data/{dataset}"
        _npz_deterministic_data = output_dir / "extra_data/deterministic_subset.npz"
        _npz_deterministic_dummy_training_tf1 = (
            output_dir / "extra_data/deterministic_dummy_training_tf1.npz"
        )
        _npz_deterministic_wo_cartesians_training_tf1 = (
            output_dir / "extra_data/deterministic_wo_cartesians_training_tf1.npz"
        )
        _npz_deterministic_w_cartesians_training_tf1 = (
            output_dir / "extra_data/deterministic_w_cartesians_training_tf1.npz"
        )

        if dataset == "linear_dimers":
            _agnostic_params_total_steps = 400
            _agnostic_params_cost_scale_soft_start = (200, 300)
            _log_images_image_step = 100
            _agnostic_params_checkpoint_step = max(
                1, int(_agnostic_params_total_steps / 10)
            )
            _dummy_training_steps_should_be = 234
            _wo_cartesian_training_tf1_nsteps = 200
            _tf2_train_wo_cartesians_atol = 0.5
            _tf2_train_wo_cartesians_rtol = 1e-7
            _tf1_checkpoint = (
                output_dir / "checkpoints/wo_cartesians/step200.ckpt.index"
            )
            _tf2_checkpoint = (
                output_dir / "checkpoints/finished_training/saved_model_50000.keras"
            )
            # fmt: off
            _tf1_cost_scale_should_be = np.array(
                [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                 0.25, 0.5, 0.75, 1., 1., 1., 1.]
            )
            _tf2_cost_scale_should_be = np.array(
                [
                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                    0.08, 0.09, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0,
                ]
            )
            # fmt: on

            _ = get_from_kondata(
                dataset,
                output_dir,
                mk_parentdir=True,
                silence_overwrite_message=True,
                download_extra_data=True,
                download_checkpoints=True,
            )

            self.assertTrue(
                output_dir.is_dir(),
                success_msg=(
                    f"The datafiles to run this test are present at {output_dir}. Continuing."
                ),
            )
            self.assertTrue(
                _npz_deterministic_dummy_training_tf1.is_file(),
            )
            self.assertTrue(
                _npz_deterministic_wo_cartesians_training_tf1.is_file(),
            )
            self.assertTrue(
                _npz_deterministic_w_cartesians_training_tf1.is_file(),
            )
            self.assertTrue(
                _tf1_checkpoint.is_file(),
            )
            self.assertTrue(
                _tf2_checkpoint.is_file(),
            )
        elif dataset == "two_state":
            _agnostic_params_total_steps = 200
            _agnostic_params_cost_scale_soft_start = (100, 150)
            _log_images_image_step = 5
            _agnostic_params_checkpoint_step = 5
            _dummy_training_steps_should_be = 39
            _wo_cartesian_training_tf1_nsteps = 100
            _tf2_train_wo_cartesians_atol = 0.8
            _tf2_train_wo_cartesians_rtol = 0.2
            output_dir.mkdir(parents=True, exist_ok=True)
            _ = get_from_kondata(
                dataset,
                output_dir,
                mk_parentdir=True,
                silence_overwrite_message=True,
                download_extra_data=True,
                download_checkpoints=True,
            )
            self.assertTrue(
                (output_dir / "trajs.h5").is_file(),
            )
            _npz_deterministic_data.parent.mkdir(parents=True, exist_ok=True)
            _tf1_checkpoint = output_dir / "checkpoints/wo_cartesians/step10.ckpt.index"
            _tf1_cost_scale_should_be = np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
            _tf2_cost_scale_should_be = np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
        else:
            raise Exception(f"This dataset is currently not available for testing.")

        # em_tf1/em_tf2 agnostic parameters:
        agnostic_params = dict(
            dihedral_cost_scale=1,
            angle_cost_scale=1,
            cartesian_cost_scale=0,
            distance_cost_scale=1,
            cartesian_distance_cost_scale=1,
            center_cost_scale=1,
            l2_reg_constant=0.001,
            # Wondering why ADCParameters implements an auto_cost_scale
            auto_cost_scale=1,
            # the remaining parameters
            cartesian_cost_variant="mean_abs",
            cartesian_cost_scale_soft_start=_agnostic_params_cost_scale_soft_start,
            cartesian_pwd_start=1,
            cartesian_pwd_step=3,
            dihedral_cost_variant="mean_abs",
            cartesian_dist_sig_parameters=[400, 10, 5, 1, 2, 5],
            checkpoint_step=_agnostic_params_checkpoint_step,
            id="diUbq",
        )

        # instantiate parameters with all costs (except regularization) set to one
        parameters = em_tf1.ADCParameters()
        parameters.main_path = em_tf1.misc.run_path(str(output_dir / "runs"))
        self.console.log(
            f"Creating a em_tf1 instance of `ADCParameters` at {parameters.main_path}."
        )
        for key, val in agnostic_params.items():
            setattr(parameters, key, val)

        # Create a dataset
        # this circumvents the MolData class, but this way, we can
        # be deterministic with how data is fed into the network
        self.console.log(
            "Verifying the `deterministic=True` option of `TrajEnsemble.batch_iterator`."
        )
        if dataset == "linear_dimers":
            trajs = em.TrajEnsemble.from_dataset(output_dir / "trajs.h5")
            shapes = [(256, 454), (256, 453), (256, 456, 3), (256, 455), (256, 322)]
            if dataset_selection == "random":
                dataset_tf1 = trajs.batch_iterator(parameters.batch_size)
                dataset_tf2 = trajs.tf_dataset(parameters.batch_size)
            elif dataset_selection == "deterministic":
                # make a small test to assure deterministic sampling
                fail_message = (
                    "You chose the 'deterministic' sampling for the input dataset. "
                    "For this method, the initial seed is set to what you choose and "
                    "subsequent calls to `np.random.choice` will yield the same "
                    "subset of the dataset. To make sure this works correctly, "
                    "we check the seeded batches with a small dataset we saved to "
                    "disk on 2024-01-10. This test can also fail, when numpy "
                    "changes its seeding. To generate a new deterministic subsample "
                    "you can run this code:\n"
                    "import encodermap as em\n"
                    "import numpy as np\n"
                    f"trajs = em.load('{output_dir / 'trajs.h5'}')\n"
                    "samples = {}\n"
                    "for i, data in enumerate(trajs.batch_iterator(20, deterministic=True)):\n"
                    "    if i >= 5:\n"
                    "        break\n"
                    "    for j, d in enumerate(data):\n"
                    "        samples[f'{i}_{j}'] = d\n"
                    f"np.savez(\"{output_dir / 'extra_data/deterministic_subset.npz'}\", **samples)\n\n"
                    "After which you can rerun the test."
                )
                if not _npz_deterministic_data.is_file():
                    raise Exception(fail_message)
                _deterministic_data = np.load(_npz_deterministic_data)
                for i, data in enumerate(trajs.batch_iterator(20, deterministic=True)):
                    if i >= 5:
                        break
                    for j, d in enumerate(data):
                        self.assertAllEqual(
                            _deterministic_data[f"{i}_{j}"],
                            d,
                            msg=fail_message,
                            success_msg=(
                                f"The deterministic option for `TrajEnsemble.batch_iterator` yielded the same data "
                                f"from the {dataset} dataset at iteration {i} for datapoint {j} with shape {d.shape} as "
                                f"is stored in the file {_npz_deterministic_data} which was created during development "
                                f"on 2023-12-01."
                            ),
                        )
                dataset_tf1 = trajs.tf_dataset(
                    parameters.batch_size, deterministic=True, prefetch=False
                )
                trajs2 = deepcopy(trajs)
                dataset_tf2 = trajs2.tf_dataset(
                    parameters.batch_size, deterministic=True, prefetch=False
                )
        elif dataset == "random":
            raise NotImplementedError()
        elif dataset == "TrpCage":
            raise NotImplementedError()
        elif dataset == "two_state":
            trajs = em.TrajEnsemble.from_dataset(output_dir / "trajs.h5")
            shapes = [(256, 454), (256, 453), (256, 456, 3), (256, 455), (256, 322)]
            if dataset_selection == "random":
                dataset_tf1 = trajs.batch_iterator(parameters.batch_size)
                dataset_tf2 = trajs.tf_dataset(parameters.batch_size)
            elif dataset_selection == "deterministic":
                # make a small test to assure deterministic sampling
                fail_message = (
                    "You chose the 'deterministic' sampling for the input dataset. "
                    "For this method, the initial seed is set to what you choose and "
                    "subsequent calls to `np.random.choice` will yield the same "
                    "subset of the dataset. To make sure this works correctly, "
                    "we check the seeded batches with a small dataset we saved to "
                    "disk on 2024-02-02. This test can also fail, when numpy "
                    "changes its seeding. To generate a new deterministic subsample "
                    "you can run this code:\n"
                    "import encodermap as em\n"
                    "import numpy as np\n"
                    "from pathlib import Path\n"
                    f"Path(\"{output_dir / 'extra_data'}\").mkdir(parents=True, "
                    f"exist_ok=True)\n"
                    f"trajs = em.load('{output_dir / 'trajs.h5'}')\n"
                    "samples = {}\n"
                    "for i, data in enumerate(trajs.batch_iterator(20, "
                    "deterministic=True)):\n"
                    "    if i >= 5:\n"
                    "        break\n"
                    "    for j, d in enumerate(data):\n"
                    "        samples[f'{i}_{j}'] = d\n"
                    f"np.savez(\"{output_dir / 'extra_data/deterministic_subset.npz'}"
                    f'", **samples)\n\n'
                    "After which you can rerun the test."
                )
                if not _npz_deterministic_data.is_file():
                    raise Exception(fail_message)
                _deterministic_data = np.load(_npz_deterministic_data)
                for i, data in enumerate(trajs.batch_iterator(20, deterministic=True)):
                    if i >= 5:
                        break
                    for j, d in enumerate(data):
                        self.assertAllEqual(
                            _deterministic_data[f"{i}_{j}"],
                            d,
                            msg=fail_message,
                            success_msg=(
                                f"The deterministic option for `TrajEnsemble.batch_iterator` yielded the same data "
                                f"from the {dataset} dataset at iteration {i} for datapoint {j} with shape {d.shape} as "
                                f"is stored in the file {_npz_deterministic_data} which was created during development "
                                f"on 2023-12-01."
                            ),
                        )
                dataset_tf1 = trajs.tf_dataset(
                    parameters.batch_size, deterministic=True, prefetch=False
                )
                trajs2 = deepcopy(trajs)
                dataset_tf2 = trajs2.tf_dataset(
                    parameters.batch_size, deterministic=True, prefetch=False
                )
        else:
            raise ValueError(
                f"Argument `dataset` needs to be either 'linear_dimers' or 'random', not {dataset=}"
            )

        # make some tests for the tf1 and tf2 datasets
        # first use `take()` to compare them
        if not save_tf1_checkpoints:
            ds1 = list(dataset_tf1.take(1).as_numpy_iterator())[0]
            ds2 = list(dataset_tf2.take(1).as_numpy_iterator())[0]
            for i, (d1, d2) in enumerate(zip(ds1, ds2)):
                self.assertAllEqual(
                    d1,
                    d2,
                    msg=f"Failed before even checking iteration.",
                    success_msg=(
                        f"The first call to `TrajEnsemble.tf_dataset.take(1)` returned the same data for datapoint "
                        f"{i} with {d1.shape}-shaped data."
                    ),
                )
            self.console.log(
                "Verifying the `deterministic=True` option of `TrajEnsemble.tf_dataset`. "
                "This can sometimes fail, when tensorflow calls the method out of sync. "
                "There is an experimental function `_synchronize_deterministic_gens` to fix that. But this function "
                "is still under development."
            )

            # then synchronize the gens
            _synchronize_deterministic_gens(dataset_tf1, dataset_tf2, self)

        # set the fake central atoms
        fake_central_atoms = np.array(["N", "CA", "C"] * trajs.n_residues[0])

        # train the tf1 and tf2 dummies
        if train_tf1_dummy != "skip":
            # We still need a fake MolData, that has all
            # the angles and distances, because encodermap wants to mean all of
            # them, not just the batch.
            class FakeMolData:
                pass

            class FakeCentralAtoms:
                pass

            self.console.log(
                f"The argument `train_tf1_dummy` was set to {train_tf1_dummy}, which means, I will create two classes: "
                f"`FakeMoldData` and `FakeCentralAtoms`, which will hold all the necessary attributes, that "
                f"`encodermap.encodermap_tf1.moldata.MolData` needs. The `FakeCentralAtoms` is a dataclass, that holds "
                f"the names of the central atoms (which for em_tf1 is N, H, CA, C, O)."
            )

            fake_moldata = FakeMolData()
            # added .transpose("frame", ...) and removed .T
            angles = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_angles.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            # added .transpose("frame", ...) and removed .T
            lengths = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_distances.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            # added .transpose("frame", ...) and removed .T
            dihedrals = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_dihedrals.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            self.assertIsInstance(
                angles,
                np.ndarray,
                success_msg="The instance of the local variable `angles` is correct.",
            )
            self.assertEqual(
                angles.shape[-1],
                shapes[0][-1],
                msg=f"{angles.shape=}",
                success_msg=(
                    f"The shape of the numpy array stored in the local variable `angles` is correct. These angles can be "
                    f"used by em_tf1 to calculate mean angles of all samples, which for the {dataset} dataset is {len(angles)}. "
                    f"These mean angles will be used in conjunction with the mean lengths to calculate the cost references."
                ),
            )
            fake_moldata.angles = angles
            fake_moldata.lengths = lengths
            fake_moldata.dihedrals = dihedrals
            fake_moldata.central_atoms = FakeCentralAtoms()
            fake_moldata.central_atoms.names = fake_central_atoms

            # here we can create a heavily modified version of the tf1 ADCEncoderMap
            dummy_parameters = copy.deepcopy(parameters)
            dummy_parameters.main_path = em.misc.run_path(
                os.path.join(parameters.main_path, "dummy")
            )
            dummy_parameters.n_steps = int(
                len(fake_moldata.angles) / parameters.batch_size
            )
            self.assertEqual(
                dummy_parameters.n_steps,
                _dummy_training_steps_should_be,
                msg=(f"{len(fake_moldata.angles)=}, {parameters.batch_size=}"),
                success_msg=(
                    f"The training of the dummy emap_tf1 for the {dataset} dataset commences for the correct number of "
                    f"steps. The intention is to roughly iterate over all samples once to get the cost references. The "
                    f"{dataset} dataset has {len(angles)} samples. With a batch size of {dummy_parameters.batch_size=} "
                    f"the number of steps should be {dummy_parameters.n_steps}."
                ),
            )
            with Capturing() as output:
                dummy_e_map_tf1 = get_custom_adc_encodermap(
                    tf=tf.compat.v1,
                    initial_weights=initial_weights,
                    check_weights=dataset == "linear_dimers",
                    dummy=True,
                )(
                    parameters=dummy_parameters,
                    train_data=dataset_tf1,
                    train_data_len=trajs.n_frames,
                    train_moldata_with_angles=fake_moldata,
                    shapes=shapes,
                    trajs=trajs,
                )
            self.assertGreater(
                len(output),
                0,
                success_msg="The creation of the custom_ADC_encodermap printed successfully to stdout.",
            )
            for line in output:
                if "kernel" in line or "bias" in line:
                    if initial_weights == "ones":
                        self.assertIn(
                            "Constant(1)",
                            line,
                            msg=f"{output=}",
                            success_msg=(
                                f"The {initial_weights=} argument to `calculate_loss_and_gradients_from_dataset` makes "
                                f"the `get_custom_adc_encodermap` function print the type of kernel/bias-initializer used. "
                                f"The correct type of initializers were used."
                            ),
                        )
                    else:
                        self.assertIn(
                            "variance_scaling/random_normal",
                            line,
                            msg=f"{output=}",
                            success_msg=(
                                f"The {initial_weights=} argument to `calculate_loss_and_gradients_from_dataset` makes "
                                f"the `get_custom_adc_encodermap` function print the type of kernel/bias-initializer used. "
                                f"The correct type of initializers were used."
                            ),
                        )

            if train_tf1_dummy == "train_and_check":
                self.console.log(
                    f"Because `train_tf1_dummy` was set to 'train_and_check' I will now commence and train the "
                    f"`dummy_e_map_tf1`. Afterwards, I will compare the logged costs/weights/gradients with data saved "
                    f"in the file {_npz_deterministic_dummy_training_tf1}. This data was sved on 2023-12-01 during "
                    f"development."
                )
                # read the protobufs and compare
                # if the dataset is deterministic
                # the weights/gradients/losses need to be deterministic
                dummy_e_map_tf1.train()
                deterministic_dummy_training_tf1 = np.load(
                    _npz_deterministic_dummy_training_tf1
                )
                records_files = list(
                    (Path(dummy_parameters.main_path) / "train").glob("*")
                )
                cost_means = {
                    "dihedral_cost": [],
                    "angle_cost": [],
                    "cartesian_cost": [],
                }

                visited_tags = set()
                for records_file in records_files:
                    records = summary_iterator(str(records_file))
                    for i, summary in enumerate(data_loss_iterator(records)):
                        if summary.step >= 10:
                            if summary.step % 10 != 0:
                                continue
                        for j, v in enumerate(summary.summary.value):
                            visited_tags.add(v.tag)
                            if "dense" in v.tag and (
                                "values" in v.tag or "gradients" in v.tag
                            ):
                                name = f"step_{summary.step}_{v.tag}"

                                # compare the weights/biases/gradients of a previous training with the one
                                # logged to tensorboard
                                ar = tf.make_ndarray(v.tensor)
                                self.assertAllClose(
                                    ar,
                                    deterministic_dummy_training_tf1[name],
                                    rtol=1e-2,
                                    success_msg=(
                                        f"The tensorboard data with tag {v.tag} in file {records_file} at step {summary.step} "
                                        f"matches the data in the file {_npz_deterministic_dummy_training_tf1} with the tag "
                                        f"{name}."
                                    ),
                                )

                                # compare the weights/biases/gradients saved during training to the ones
                                # logged to tensorboard
                                if "gradients" in name:
                                    pass
                                else:
                                    ar2 = dummy_e_map_tf1._accumulated_weights[name]
                                    try:
                                        self.assertAllClose(
                                            ar,
                                            ar2,
                                            msg=f"The values of {name=} and {v.tag=} are different at step {summary.step=}.",
                                            success_msg=(
                                                f"The tensorboard data with tag {v.tag} in file {records_file} at step {summary.step} "
                                                f"matches the data saved during training with the tag {name}."
                                            ),
                                        )
                                    except AssertionError as e:
                                        if "gradients" not in v.tag:
                                            raise e
                                        self.console.log(
                                            f"The comparison of the {v.simple_value=} with {v.tag=} at step {summary.step=} "
                                            f"with {name=} {dummy_e_map_tf1._accumulated_gradients[name]=} failed. I will now "
                                            f"check the whole dicts of the {_npz_deterministic_dummy_training_tf1} file and"
                                            f"`dummy_e_map_tf1._accumulated_gradients` for matching values."
                                        )
                                        o = check_matching_keys(
                                            dummy_e_map_tf1._accumulated_gradients,
                                            deterministic_dummy_training_tf1,
                                            e,
                                        )
                                        raise Exception(o) from e

                            # compare the costs from a previous training with the ones
                            # logged to tensorboard
                            if "cost" in v.tag and "debug" not in v.tag:
                                name = f"step_{summary.step}_{v.tag}"
                                self.assertAlmostEqual(
                                    deterministic_dummy_training_tf1[name],
                                    v.simple_value,
                                    places=2,
                                    msg=(
                                        f"The values in the tfrecords file {records_file} with {v.tag=}, {name=} "
                                        f"{summary.step=} {v.simple_value=} does not "
                                        f"coincide with the value stored in {_npz_deterministic_dummy_training_tf1}: "
                                        f"{deterministic_dummy_training_tf1[name]=}"
                                    ),
                                    success_msg=(
                                        f"The tensorboard data with tag {v.tag} in file {records_file} at step {summary.step} "
                                        f"matches the data in the file {_npz_deterministic_dummy_training_tf1} with the tag "
                                        f"{name}."
                                    ),
                                )
                                if "dihedral_cost" in v.tag:
                                    cost_means["dihedral_cost"].append(v.simple_value)
                                if "angle_cost" in v.tag:
                                    cost_means["angle_cost"].append(v.simple_value)
                                if (
                                    "cartesian_cost" in v.tag
                                    and "cartesian_cost_scale" not in v.tag
                                ):
                                    cost_means["cartesian_cost"].append(v.simple_value)
                            if "debug" in v.tag and summary.step in [
                                0,
                                1,
                                2,
                                5,
                                10,
                                20,
                            ]:
                                self.console.log(
                                    f"At step {summary.step}, I found the tag {v.tag}"
                                )
                                val = v.simple_value
                                if isinstance(val, float):
                                    self.console.log(v.tag, val)
                                elif val.ndim == 2:
                                    self.console.log(v.tag, v.simple_value[:2, :2])
                                elif val.ndim == 3:
                                    self.console.log(v.tag, v.simple_value[:2, :5, -1])
                                else:
                                    self.console.log(v.tag, val)
                self.assertAlmostEqual(
                    np.mean(cost_means["dihedral_cost"]),
                    COST_REFERENCES[dataset]["dihedral_cost"],
                    places=3,
                    msg=f"{cost_means=}",
                    success_msg=(
                        f"The dihedral cost means are preserved in this em_tf1_dummy training run and "
                        f"{_npz_deterministic_dummy_training_tf1}."
                    ),
                )
                self.assertAlmostEqual(
                    np.mean(cost_means["angle_cost"]),
                    COST_REFERENCES[dataset]["angle_cost"],
                    places=3,
                    msg=f"{cost_means=}",
                    success_msg=(
                        f"The angle cost means are preserved in this em_tf1_dummy training run "
                        f"and {_npz_deterministic_dummy_training_tf1}."
                    ),
                )
                self.assertAlmostEqual(
                    np.mean(cost_means["cartesian_cost"]),
                    COST_REFERENCES[dataset]["cartesian_cost"],
                    places=2,
                    msg=f"{cost_means=}",
                    success_msg=(
                        f"The cartesian cost means are preserved in this em_tf1_dummy training run "
                        f"and {_npz_deterministic_dummy_training_tf1}."
                    ),
                )
                dummy_e_map_tf1.close()
                self.console.log(
                    "Training of encodermap_tf1/AngleDihedralCartesianEncoderMapDummy "
                    "is deterministic."
                )
            elif train_tf1_dummy == "train_and_save":
                deterministic_dummy_training = {}

                for i, layer in enumerate(
                    list(dummy_e_map_tf1._encoder_layers.values())
                    + list(dummy_e_map_tf1._decoder_layers.values())
                ):
                    w, b = dummy_e_map_tf1.sess.run(layer.weights)
                    deterministic_dummy_training[f"init_weights_{i}"] = w
                    deterministic_dummy_training[f"init_bias_{i}"] = b

                dummy_e_map_tf1.train()
                dummy_e_map_tf1.close()
                records_files = list(
                    (Path(dummy_parameters.main_path) / "train").glob("*")
                )
                self.assertGreater(
                    len(records_files),
                    0,
                    msg="No records files to read.",
                    success_msg=f"Records files {records_files} have been written.",
                )
                for records_file in records_files:
                    records = summary_iterator(str(records_file))
                    for i, summary in enumerate(data_loss_iterator(records)):
                        if summary.step >= 200:
                            if summary.step % 10 != 0:
                                continue
                        for j, v in enumerate(summary.summary.value):
                            if "dense" in v.tag and (
                                "values" in v.tag or "gradients" in v.tag
                            ):
                                name = f"step_{summary.step}_{v.tag}"
                                ar = tf.make_ndarray(v.tensor)
                                deterministic_dummy_training[name] = ar
                            if "cost" in v.tag:
                                name = f"step_{summary.step}_{v.tag}"
                                deterministic_dummy_training[name] = v.simple_value

                _npz_deterministic_dummy_training_tf1.unlink(missing_ok=True)
                np.savez(
                    _npz_deterministic_dummy_training_tf1,
                    **deterministic_dummy_training,
                )
                # Encodermap imports
                from encodermap.encodermap_tf1.misc import read_from_log

                costs = read_from_log(
                    os.path.join(dummy_parameters.main_path, "train"),
                    ["cost/angle_cost", "cost/dihedral_cost", "cost/cartesian_cost"],
                )
                means = []
                for values in costs:
                    means.append(np.mean([i.value for i in values]))
                self.fail(
                    msg=(
                        f"I've trained an encodermap_tf1/AngleDihedralCartesianEncoderMapDummy for "
                        f"{dummy_parameters.n_steps} steps and saved the "
                        f"weights/activations/gradients/losses to {_npz_deterministic_dummy_training_tf1}. You can "
                        f"now run this function with `train_tf1_dummy='train_and_check'` "
                        f"to see, whether the training is deterministic. The cost means "
                        f"(angle, dihedral, cartesian) are: "
                        f"{means=}."
                    )
                )
            else:
                raise ValueError(
                    f"`train_tf1_dummy needs to be one of 'skip', 'train_and_check', "
                    f"'train_and_save'. You provided: {train_tf1_dummy=}."
                )

        else:
            self.console.log(
                f"I will not train an encodermap_tf1/AngleDihedralCartesianEncoderMapDummy. "
                f"If you want to prove the deterministic training, set "
                f"`train_tf1_dummy='train_and_check'` and see for yourself."
            )

        if train_tf2_dummy == "train_and_check":
            parameters_tf2 = em.ADCParameters(**agnostic_params)
            parameters_tf2.main_path = em.misc.run_path(str(output_dir / "runs"))
            model = gen_functional_model(
                input_shapes=[s[1:] for s in shapes],
                parameters=parameters_tf2,
                kernel_initializer="deterministic",
                bias_initializer="deterministic",
                write_summary=False,
            )

            # check the weights and biases
            deterministic_dummy_training = np.load(
                _npz_deterministic_dummy_training_tf1
            )
            i = 0
            for layer in model.encoder_model.layers + model.decoder_model.layers:
                if "dense" not in layer.__class__.__name__.lower():
                    continue
                self.assertAllClose(
                    layer.get_weights()[0],
                    deterministic_dummy_training[f"init_weights_{i}"],
                    rtol=1e-5,
                    success_msg=(
                        f"The initial weights of a encodermap_tf2/AngleDihedralCartesianEncoderMap matches the initial "
                        f"weights saved in the file {_npz_deterministic_dummy_training_tf1}. This means, that the starting "
                        f"weights for the tf1 and tf2 version are identical. Although the dummy training does not depend "
                        f"on initial weights."
                    ),
                )
                self.assertAllClose(
                    layer.get_weights()[1],
                    deterministic_dummy_training[f"init_bias_{i}"],
                    rtol=1e-5,
                    success_msg=(
                        f"The initial bias of a encodermap_tf2/AngleDihedralCartesianEncoderMap matches the initial "
                        f"bias saved in the file {_npz_deterministic_dummy_training_tf1}. This means, that the starting "
                        f"bias for the tf1 and tf2 version are identical. Although the dummy training does not depend "
                        f"on initial bias."
                    ),
                )
                i += 1

            emap = em.AngleDihedralCartesianEncoderMap(
                trajs=trajs,
                parameters=parameters_tf2,
                model=model,
                dataset=dataset_tf2,
            )

            result = emap.train_for_references()
            self.assertEqual(
                len(result["dihedral_cost"]),
                _dummy_training_steps_should_be,
                success_msg=(
                    f"Training for references with em_tf2 took the expected {_dummy_training_steps_should_be} steps."
                ),
            )

            self.assertAlmostEqual(
                np.mean(result["dihedral_cost"]),
                COST_REFERENCES[dataset]["dihedral_cost"],
                places=3,
                msg=f"{result['dihedral_cost']=}",
                success_msg=(
                    f"The dihedral cost means are preserved in this em_tf2_dummy training run "
                    f"and {_npz_deterministic_dummy_training_tf1}."
                ),
            )
            self.assertAlmostEqual(
                np.mean(result["angle_cost"]),
                COST_REFERENCES[dataset]["angle_cost"],
                places=3,
                msg=f"{result['angle_cost']=}",
                success_msg=(
                    f"The angle cost means are preserved in this em_tf2_dummy training run "
                    f"and {_npz_deterministic_dummy_training_tf1}."
                ),
            )
            self.assertAlmostEqual(
                np.mean(result["cartesian_cost"]),
                COST_REFERENCES[dataset]["cartesian_cost"],
                places=1,
                msg=f"{result['cartesian_cost']=}",
                success_msg=(
                    f"The cartesian cost means are preserved in this em_tf2_dummy training run "
                    f"and {_npz_deterministic_dummy_training_tf1}."
                ),
            )
            self.console.log(
                "Training of encodermap_tf2/AngleDihedralCartesianEncoderMapDummy "
                "is deterministic."
            )
        elif train_tf2_dummy == "skip":
            self.console.log(
                f"I will not train an encodermap_tf2/AngleDihedralCartesianEncoderMap "
                f"in dummy mode. If you want to prove the deterministic training, set "
                f"`train_tf1_dummy='train_and_check'` and see for yourself."
            )
        else:
            raise ValueError(
                f"`train_tf2_dummy needs to be one of 'train_and_check' or "
                f"'skip'. You provided: {train_tf2_dummy=}."
            )

        # dummy training finished here comes small runs of real training
        agnostic_params["cartesian_cost_scale"] = 1
        agnostic_params["dihedral_cost_reference"] = COST_REFERENCES[dataset][
            "dihedral_cost"
        ]
        agnostic_params["angle_cost_reference"] = COST_REFERENCES[dataset]["angle_cost"]
        agnostic_params["cartesian_cost_reference"] = COST_REFERENCES[dataset][
            "cartesian_cost"
        ]
        agnostic_params["n_steps"] = agnostic_params["cartesian_cost_scale_soft_start"][
            0
        ]

        # consume the datasets if skipped
        if train_tf1_dummy == "skip":
            dataset_tf1.take(_dummy_training_steps_should_be)
        if train_tf2_dummy == "skip":
            dataset_tf2.take(_dummy_training_steps_should_be)

        if train_tf1_wo_cartesians in ["train_and_save", "train_and_check"]:

            class FakeMolData:
                pass

            class FakeCentralAtoms:
                pass

            fake_moldata = FakeMolData()
            # added .transpose("frame", ...) and removed .T
            angles = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_angles.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            # added .transpose("frame", ...) and removed .T
            lengths = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_distances.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            # added .transpose("frame", ...) and removed .T
            dihedrals = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_dihedrals.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            self.assertIsInstance(
                angles,
                np.ndarray,
                success_msg=("The type of angles is correct (np.ndarray)."),
            )
            self.assertEqual(
                angles.shape[-1],
                shapes[0][-1],
                msg=f"{angles.shape=}",
                success_msg=f"The angles have the expected shape: {angles.shape=}.",
            )
            fake_moldata.angles = angles
            fake_moldata.lengths = lengths
            fake_moldata.dihedrals = dihedrals
            fake_moldata.central_atoms = FakeCentralAtoms()
            fake_moldata.central_atoms.names = fake_central_atoms

            # here we can create a heavily modified version of the tf1 ADCEncoderMap
            parameters_ = copy.deepcopy(parameters)
            for k, v in agnostic_params.items():
                setattr(parameters_, k, v)
            with Capturing() as output:
                wo_cartesians_e_map_tf1 = get_custom_adc_encodermap(
                    tf=tf.compat.v1,
                    initial_weights=initial_weights,
                    check_weights=dataset == "linear_dimers",
                    dummy=False,
                )(
                    parameters=parameters_,
                    train_data=dataset_tf1,
                    train_data_len=trajs.n_frames,
                    train_moldata_with_angles=fake_moldata,
                    shapes=shapes,
                    trajs=trajs,
                )

            if train_tf1_wo_cartesians == "train_and_save":
                deterministic_wo_cartesians_training = {}

                for i, layer in enumerate(
                    list(wo_cartesians_e_map_tf1._encoder_layers.values())
                    + list(wo_cartesians_e_map_tf1._decoder_layers.values())
                ):
                    w, b = wo_cartesians_e_map_tf1.sess.run(layer.weights)
                    deterministic_wo_cartesians_training[f"init_weights_{i}"] = w
                    deterministic_wo_cartesians_training[f"init_bias_{i}"] = b

                wo_cartesians_e_map_tf1.train()
                records_files = list(
                    (Path(wo_cartesians_e_map_tf1.p.main_path) / "train").glob("*")
                )
                self.assertGreater(
                    len(records_files),
                    0,
                    msg="No records files to read.",
                    success_msg=f"Records files ({records_files}) have been written.",
                )
                for records_file in records_files:
                    records = summary_iterator(str(records_file))
                    for i, summary in enumerate(data_loss_iterator(records)):
                        if summary.step >= 10:
                            if summary.step % 50 != 0:
                                continue
                        for j, v in enumerate(summary.summary.value):
                            if "dense" in v.tag and (
                                "values" in v.tag or "gradients" in v.tag
                            ):
                                name = f"step_{summary.step}_{v.tag}"
                                ar = tf.make_ndarray(v.tensor)
                                deterministic_wo_cartesians_training[name] = ar
                            if "cost" in v.tag:
                                name = f"step_{summary.step}_{v.tag}"
                                deterministic_wo_cartesians_training[
                                    name
                                ] = v.simple_value

                # can't get final weights. The layers still have their initial
                # weights saved in them.
                # I can't save checkpoints in this tf runtime because of
                # eager/graph mode, so I ran one training in graph mode and
                # used its checkpoint to reconstruct the weights
                # I saved the file to output_dir / "tf1_checkpoints/wo_cartesians/step200.ckpt"
                # raise Exception("get final weights")
                # for i, layer in enumerate(
                #     list(wo_cartesians_e_map_tf1._encoder_layers.values())
                #     + list(wo_cartesians_e_map_tf1._decoder_layers.values())
                # ):
                #     w, b = wo_cartesians_e_map_tf1.sess.run(layer.weights)
                #     deterministic_wo_cartesians_training[f"final_weights_{i}"] = w
                #     deterministic_wo_cartesians_training[f"final_bias_{i}"] = b
                wo_cartesians_e_map_tf1.close()
                np.savez(
                    _npz_deterministic_wo_cartesians_training_tf1,
                    **deterministic_wo_cartesians_training,
                )
                self.fail(
                    msg=(
                        f"I've trained an encodermap_tf1/AngleDihedralCartesianEncoderMap"
                        f" for w/o cartesian cost for {parameters_.n_steps} steps and saved the "
                        f"weights/activations/gradients/losses to "
                        f"{_npz_deterministic_wo_cartesians_training_tf1}. You can "
                        f"now run this function with `train_tf1_dummy='train_and_check'` "
                        f"to see, whether the training is deterministic."
                    )
                )
            elif train_tf1_wo_cartesians == "train_and_check":
                # here we can create a heavily modified version of the tf1 ADCEncoderMap
                parameters_ = copy.deepcopy(parameters)
                for k, v in agnostic_params.items():
                    setattr(parameters_, k, v)
                parameters_.n_steps = _wo_cartesian_training_tf1_nsteps
                with Capturing() as output:
                    wo_cartesians_e_map_tf1 = get_custom_adc_encodermap(
                        tf=tf.compat.v1,
                        initial_weights=initial_weights,
                        check_weights=dataset == "linear_dimers",
                        dummy=False,
                        saver=save_tf1_checkpoints,
                    )(
                        parameters=parameters_,
                        train_data=dataset_tf1,
                        train_data_len=trajs.n_frames,
                        train_moldata_with_angles=fake_moldata,
                        shapes=shapes,
                        trajs=trajs,
                    )
                wo_cartesians_e_map_tf1.train()
                wo_cartesians_e_map_tf1.close()
                deterministic_wo_cartesians_training = np.load(
                    _npz_deterministic_wo_cartesians_training_tf1
                )
                records_files = list((Path(parameters_.main_path) / "train").glob("*"))

                for records_file in records_files:
                    records = summary_iterator(str(records_file))
                    for i, summary in enumerate(data_loss_iterator(records)):
                        if summary.step >= 10:
                            break
                        for j, v in enumerate(summary.summary.value):
                            if "dense" in v.tag and (
                                "values" in v.tag or "gradients" in v.tag
                            ):
                                name = f"step_{summary.step}_{v.tag}"
                                ar = tf.make_ndarray(v.tensor)
                                self.assertAllClose(
                                    ar,
                                    deterministic_wo_cartesians_training[name],
                                    atol=5e-1,
                                    success_msg=(
                                        f"The value of {v.tag} at step {summary.step} "
                                        f"is identical within an atol of 0.5 of the current "
                                        f"em_tf1_wo_cartesians and the content of the file "
                                        f"{_npz_deterministic_dummy_training_tf1}"
                                    ),
                                )
                            if "cost" in v.tag:
                                name = f"step_{summary.step}_{v.tag}"
                                self.assertAllClose(
                                    deterministic_wo_cartesians_training[name],
                                    v.simple_value,
                                    atol=0.5,
                                    success_msg=(
                                        f"The value of {v.tag} at step {summary.step} "
                                        f"is identical within an atol of 0.5 of the current "
                                        f"em_tf1_wo_cartesians and the content of the file "
                                        f"{_npz_deterministic_dummy_training_tf1}"
                                    ),
                                    msg=(
                                        f"The value of {v.tag} at step {summary.step} "
                                        f"is NOT identical within an atol of 0.5 of the current "
                                        f"em_tf1_wo_cartesians and the content of the file "
                                        f"{_npz_deterministic_dummy_training_tf1}"
                                    ),
                                )
                self.console.log(
                    "Training of encodermap_tf1/AngleDihedralCartesianEncoderMap "
                    "w/o cartesian cost is deterministic."
                )
        elif train_tf1_wo_cartesians == "skip":
            self.console.log(
                f"I will not train an encodermap_tf1/AngleDihedralCartesianEncoderMap "
                f"w/o cartesians. If you want to prove the deterministic training, set "
                f"`train_tf1_wo_cartesians='train_and_check'` and see for yourself."
            )
        else:
            raise ValueError(
                f"`train_tf1_wo_cartesians needs to be one of 'train_and_save', "
                f"'train_and_check', or 'skip'. You provided: {train_tf1_wo_cartesians=}."
            )

        if train_tf2_wo_cartesians == "train_and_check":
            parameters_tf2 = em.ADCParameters(**agnostic_params)
            parameters_tf2.main_path = em.misc.run_path(str(output_dir / "runs"))
            parameters_tf2.n_steps = (
                _agnostic_params_total_steps - _agnostic_params_cost_scale_soft_start[0]
            )
            parameters_tf2.tensorboard = True
            parameters_tf2.summary_step = 1
            model = gen_functional_model(
                input_shapes=[s[1:] for s in shapes],
                parameters=parameters_tf2,
                kernel_initializer="deterministic",
                bias_initializer="deterministic",
                write_summary=False,
            )

            # check the weights and biases
            deterministic_wo_cartesians_training = np.load(
                _npz_deterministic_wo_cartesians_training_tf1
            )
            deterministic_dummy_training = np.load(
                _npz_deterministic_dummy_training_tf1
            )
            i = 0
            for layer in model.encoder_model.layers + model.decoder_model.layers:
                if "dense" not in layer.__class__.__name__.lower():
                    continue
                self.assertAllClose(
                    layer.get_weights()[0],
                    deterministic_dummy_training[f"init_weights_{i}"],
                    msg=f"weights: {i=} {layer=}",
                    rtol=1e-3,
                    success_msg=(
                        f"The weights of the layer {layer} of the "
                        f"tf2 model and the init weights of "
                        f"the em_tf1_wo_cartesians model are equal within rtol=1e-3."
                    ),
                )
                self.assertAllClose(
                    layer.get_weights()[1],
                    deterministic_dummy_training[f"init_bias_{i}"],
                    msg=f"weights: {i=} {layer=}",
                    rtol=1e-3,
                    success_msg=(
                        f"The bias of the layer {layer} of the "
                        f"tf2 model and the init weights of "
                        f"the em_tf1_wo_cartesians model are equal within rtol=1e-3."
                    ),
                )
                i += 1

            emap = em.AngleDihedralCartesianEncoderMap(
                trajs=trajs,
                parameters=parameters_tf2,
                model=model,
                dataset=dataset_tf2,
            )

            emap.add_images_to_tensorboard(image_step=_log_images_image_step)

            with Capturing() as output:
                emap.train()
            self.assertTrue(
                any(["References are already provided" in l for l in output]),
                success_msg=(
                    f"The em_tf2 AngleDihedralCartesianEncoderMap class correctly "
                    f"recognizes that references are already set."
                ),
            )

            deterministic_wo_cartesians_training = np.load(
                _npz_deterministic_wo_cartesians_training_tf1
            )

            keys = set()
            for key in deterministic_wo_cartesians_training.keys():
                if "step_0" in key and "cost" in key:
                    keys.add(key)

            mappings = {
                "cost/cartesian_cost": "Cost/Cost_4/Cartesian Cost before scaling",
                "cost/angle_cost": "Cost/Cost_1/Angle Cost",
                "cost/center_cost": "Cost/Cost_7/Center Cost",
                "cost/distance_cost": "Cost/Cost_5/Distance Cost",
                "cost/cartesian_distance_cost": "Cost/Cost_6/Cartesian Distance Cost",
                "cost/combined_cost": "Cost/Combined Cost",
                "cost/dihedral_cost": "Cost/Cost/Dihedral Cost",
                "cost/cartesian_cost_scale": "Cost/Cost_3/Cartesian Cost current scaling",
            }
            mappings = {v: k for k, v in mappings.items()}

            record_files = (Path(emap.p.main_path) / "train").glob("*")
            for records_file in record_files:
                records = summary_iterator(str(records_file))
                for i, summary in enumerate(data_loss_iterator(records)):
                    for j, v in enumerate(summary.summary.value):
                        if "Cost" in v.tag:
                            if v.tag not in mappings:
                                continue
                            tf1_tag = f"step_{summary.step}_{mappings[v.tag]}"
                            if tf1_tag not in deterministic_wo_cartesians_training:
                                continue
                            tf1_value = float(
                                deterministic_wo_cartesians_training[tf1_tag]
                            )
                            ar = float(tf.make_ndarray(v.tensor))
                            self.assertAllClose(
                                tf1_value,
                                ar,
                                atol=_tf2_train_wo_cartesians_atol,
                                rtol=_tf2_train_wo_cartesians_rtol,
                                msg=f"{tf1_tag:<35} TF1: {tf1_value:.4f} TF2: {ar:.4f} is NOT OK",
                                success_msg=(
                                    f"The cost {v.tag} at step {summary.step} is "
                                    f"equal within 0.5 of the em_tf1 and em_tf2 "
                                    f"versions of the w/o cartesian training."
                                ),
                            )

            self.console.log(
                "Training of encodermap_tf2/AngleDihedralCartesianEncoderMap "
                "w/o cartesian cost is deterministic. Returning..."
            )
        elif train_tf2_wo_cartesians == "skip":
            print(
                f"I will not train an encodermap_tf2/AngleDihedralCartesianEncoderMap "
                f"w/o cartesians. If you want to prove the deterministic training, set "
                f"`train_tf2_wo_cartesians='train_and_check'` and see for yourself."
            )
        else:
            raise ValueError(
                f"`train_tf2_wo_cartesians` needs to be one of 'train_and_check' or "
                f"'skip'. You provided: {train_tf2_wo_cartesians=}."
            )

        if save_tf1_checkpoints:
            checkpoint_file = (
                Path(parameters.main_path).resolve()
                / f"checkpoints/step{parameters_.n_steps}.ckpt.index"
            )
            if checkpoint_file.is_file():
                _checkpoint_dir = _tf1_checkpoint.parent
                _checkpoint_dir.mkdir(parents=True, exist_ok=True)
                moves = {}
                for file in checkpoint_file.parent.glob("*"):
                    new_name = _checkpoint_dir / file.name
                    file.rename(new_name)
                    moves[str(file)] = str(new_name)
                self.fail(
                    f"I've trained a tf1_wo_cartesians model and saved the "
                    f"checkpoint files to {_checkpoint_dir}. Here are the file "
                    f"renames I've carried out: {moves}.\nYou can now rerun "
                    f"the test with `save_tf1_checkpoint` set to False "
                    f"and use the weights from this checkpoint file."
                )
            else:
                self.fail(
                    f"I've trained a tf1_wo_cartesians model and tried to save "
                    f"checkpoint files to {checkpoint_file.parent}, but no "
                    f".ckpt.index file was created. Here is a glob of the dir: "
                    f"{checkpoint_file.glob('*')=}."
                )
        else:
            self.assertTrue(
                _tf1_checkpoint.is_file(),
                success_msg=(
                    f"The file {_tf1_checkpoint} exists. This file is needed to "
                    f"load weights from an already trained model, because the "
                    f"tf1 checkpoint loader/restorer does not work in eager mode, "
                    f"but the remainder of the tests need to run in eager mode."
                ),
                msg=(
                    f"The file {_tf1_checkpoint} does not exist. This file is needed "
                    f"to check and reload already trained weights. Set the argument "
                    f"`save_tf1_checkpoint` to True and run this test again. This "
                    f"will automatically save a checkpoint for later use."
                ),
            )

        # consume the datasets if skipped
        if train_tf1_wo_cartesians == "skip":
            dataset_tf1.take(_wo_cartesian_training_tf1_nsteps)
        if train_tf2_wo_cartesians == "skip":
            dataset_tf2.take(_wo_cartesian_training_tf1_nsteps)

        agnostic_params["n_steps"] = (
            _agnostic_params_total_steps
            - agnostic_params["cartesian_cost_scale_soft_start"][0]
        )
        agnostic_params["cartesian_cost_scale"] = 1

        if train_tf1_w_cartesians in ["train_and_save", "train_and_check"]:

            class FakeMolData:
                pass

            class FakeCentralAtoms:
                pass

            fake_moldata = FakeMolData()
            # added .transpose("frame", ...) and removed .T
            angles = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_angles.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            # added .transpose("frame", ...) and removed .T
            lengths = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_distances.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            # added .transpose("frame", ...) and removed .T
            dihedrals = (
                xr.open_dataset(
                    output_dir / "trajs.h5",
                    engine="h5netcdf",
                    group="CVs",
                )
                .central_dihedrals.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .values
            )
            self.assertIsInstance(
                angles, np.ndarray, success_msg="Angles is correct type (np.ndarray)."
            )
            self.assertEqual(
                angles.shape[-1],
                shapes[0][-1],
                msg=f"{angles.shape=}",
                success_msg=(
                    f"The shape of the array angles in the FakeMolData class is "
                    f"correct {angles.shape=}."
                ),
            )
            fake_moldata.angles = angles
            fake_moldata.lengths = lengths
            fake_moldata.dihedrals = dihedrals
            fake_moldata.central_atoms = FakeCentralAtoms()
            fake_moldata.central_atoms.names = fake_central_atoms

            # read the weights from the provided wo_cartesians checkpoint
            # this had to be done, because there was no way to consolidate
            # eager and graph execution regarding the saver
            # Third Party Imports
            from tensorflow.python.training import py_checkpoint_reader

            w_cartesians_init_weights = {}
            _tf1_checkpoint = (
                _tf1_checkpoint.parent / _tf1_checkpoint.stem
            ).with_suffix(".ckpt")
            reader = py_checkpoint_reader.NewCheckpointReader(str(_tf1_checkpoint))
            for i in range(6):
                for type_ in ["kernel", "bias"]:
                    if i == 0:
                        key = f"dense/{type_}"
                    else:
                        key = f"dense_{i}/{type_}"
                    w_cartesians_init_weights[key] = reader.get_tensor(key)

            # here we can create a heavily modified version of the tf1 ADCEncoderMap
            parameters_ = copy.deepcopy(parameters)
            parameters_.main_path = em_tf1.misc.run_path(str(output_dir / "runs"))
            for k, v in agnostic_params.items():
                setattr(parameters_, k, v)
            with Capturing() as output:
                w_cartesians_e_map_tf1 = get_custom_adc_encodermap(
                    tf=tf.compat.v1,
                    initial_weights=w_cartesians_init_weights,
                    check_weights=dataset == "linear_dimers",
                    dummy=False,
                )(
                    parameters=parameters_,
                    train_data=dataset_tf1,
                    train_data_len=trajs.n_frames,
                    train_moldata_with_angles=fake_moldata,
                    shapes=shapes,
                    trajs=trajs,
                    initial_step=_agnostic_params_cost_scale_soft_start[0],
                )

            if train_tf1_w_cartesians == "train_and_save":
                deterministic_w_cartesians_training = {}

                for i, layer in enumerate(
                    list(w_cartesians_e_map_tf1._encoder_layers.values())
                    + list(w_cartesians_e_map_tf1._decoder_layers.values())
                ):
                    if i == 0:
                        key = f"dense/"
                    else:
                        key = f"dense_{i}/"
                    w, b = w_cartesians_e_map_tf1.sess.run(layer.weights)
                    deterministic_w_cartesians_training[f"init_weights_{i}"] = w
                    deterministic_w_cartesians_training[f"init_bias_{i}"] = b
                    self.assertAllClose(
                        w,
                        w_cartesians_init_weights[key + "kernel"],
                        success_msg=(
                            f"The init_weights which are the final weights of the "
                            f"w/o cartesian run are correctly implemented in this "
                            f"em_tf1 w/ cartesians run, which was done by implementing "
                            f"a custom kernel initializer."
                        ),
                    )
                    self.assertAllClose(
                        b,
                        w_cartesians_init_weights[key + "bias"],
                        success_msg=(
                            f"The init_weights which are the final biases of the "
                            f"w/o cartesian run are correctly implemented in this "
                            f"em_tf1 w/ cartesians run, which was done by implementing "
                            f"a custom bias initializer."
                        ),
                    )

                w_cartesians_e_map_tf1.train()
                w_cartesians_e_map_tf1.close()
                records_files = list(
                    (Path(w_cartesians_e_map_tf1.p.main_path) / "train").glob("*")
                )
                self.assertGreater(
                    len(records_files),
                    0,
                    msg="No records files to read.",
                    success_msg=(
                        f"Successfully written records files: {records_files=}"
                    ),
                )
                for records_file in records_files:
                    records = summary_iterator(str(records_file))
                    for i, summary in enumerate(data_loss_iterator(records)):
                        if summary.step >= 10:
                            if summary.step % 25 != 0 and summary.step != 200:
                                continue
                        for j, v in enumerate(summary.summary.value):
                            if "dense" in v.tag and (
                                "values" in v.tag or "gradients" in v.tag
                            ):
                                name = f"step_{summary.step}_{v.tag}"
                                ar = tf.make_ndarray(v.tensor)
                                deterministic_w_cartesians_training[name] = ar
                            if "cost" in v.tag:
                                name = f"step_{summary.step}_{v.tag}"
                                deterministic_w_cartesians_training[
                                    name
                                ] = v.simple_value
                outfile = _npz_deterministic_w_cartesians_training_tf1
                np.savez(outfile, **deterministic_w_cartesians_training)
                self.fail(
                    msg=(
                        f"I've trained an encodermap_tf1/AngleDihedralCartesianEncoderMap"
                        f" for w/ cartesian cost for {parameters_.n_steps} steps and saved the "
                        f"weights/activations/gradients/losses to {outfile}. You can "
                        f"now run this function with `train_tf1_w_cartesians='train_and_check'` "
                        f"to see, whether the training is deterministic."
                    )
                )
            elif train_tf1_w_cartesians == "train_and_check":
                # here we can create a heavily modified version of the tf1 ADCEncoderMap
                with Capturing() as output:
                    w_cartesians_e_map_tf1 = get_custom_adc_encodermap(
                        tf=tf.compat.v1,
                        initial_weights=w_cartesians_init_weights,
                        check_weights=dataset == "linear_dimers",
                        dummy=False,
                    )(
                        parameters=parameters_,
                        train_data=dataset_tf1,
                        train_data_len=trajs.n_frames,
                        train_moldata_with_angles=fake_moldata,
                        shapes=shapes,
                        trajs=trajs,
                        initial_step=_agnostic_params_cost_scale_soft_start[0],
                    )
                for i, layer in enumerate(
                    list(w_cartesians_e_map_tf1._encoder_layers.values())
                    + list(w_cartesians_e_map_tf1._decoder_layers.values())
                ):
                    if i == 0:
                        key = f"dense/"
                    else:
                        key = f"dense_{i}/"
                    w, b = w_cartesians_e_map_tf1.sess.run(layer.weights)
                    self.assertAllClose(
                        w,
                        w_cartesians_init_weights[key + "kernel"],
                        success_msg=(
                            f"The init_weights which are the final weights of the "
                            f"w/o cartesian run are correctly implemented in this "
                            f"em_tf1 w/ cartesians run, which was done by implementing "
                            f"a custom kernel initializer."
                        ),
                    )
                    self.assertAllClose(
                        b,
                        w_cartesians_init_weights[key + "bias"],
                        success_msg=(
                            f"The init_weights which are the final weights of the "
                            f"w/o cartesian run are correctly implemented in this "
                            f"em_tf1 w/ cartesians run, which was done by implementing "
                            f"a custom kernel initializer."
                        ),
                    )
                w_cartesians_e_map_tf1.train()
                w_cartesians_e_map_tf1.close()
                deterministic_w_cartesians_training = np.load(
                    _npz_deterministic_w_cartesians_training_tf1
                )
                records_files = list((Path(parameters_.main_path) / "train").glob("*"))
                cartesian_cost_scale = []

                for records_file in records_files:
                    records = summary_iterator(str(records_file))
                    for i, summary in enumerate(data_loss_iterator(records)):
                        for j, v in enumerate(summary.summary.value):
                            if summary.step >= 10:
                                if summary.step % 25 != 0 and summary.step != 200:
                                    continue
                            if "dense" in v.tag and (
                                "values" in v.tag or "gradients" in v.tag
                            ):
                                name = f"step_{summary.step}_{v.tag}"
                                ar = tf.make_ndarray(v.tensor)
                                self.assertAllClose(
                                    ar,
                                    deterministic_w_cartesians_training[name],
                                    atol=9e-1,
                                    success_msg=(
                                        f"The value/gradient of the kernel/bias of "
                                        f"the dense layer {v.tag} at step {summary.step} is preserved between "
                                        f"two em_tf1 w/ cartesian runs."
                                    ),
                                )
                            if "cost" in v.tag:
                                name = f"step_{summary.step}_{v.tag}"
                                self.assertAllClose(
                                    deterministic_w_cartesians_training[name],
                                    v.simple_value,
                                    atol=9e-1,
                                    msg=(
                                        f"The value of the cost {v.tag} at step {summary.step} is "
                                        f"NOT preserved between two em_tf1 w/ cartesian runs. The file "
                                        f"used to compare is: {deterministic_w_cartesians_training.fid=} {records_file=}."
                                    ),
                                    success_msg=(
                                        f"The value of the cost {v.tag} at step {summary.step} is "
                                        f"preserved between two em_tf1 w/ cartesian runs."
                                    ),
                                )
                            if v.tag == "cost/cartesian_cost_scale":
                                cartesian_cost_scale.append(v.simple_value)
                cartesian_cost_scale = np.array(cartesian_cost_scale)
                should_be = _tf1_cost_scale_should_be
                self.assertAllClose(
                    cartesian_cost_scale,
                    should_be,
                    success_msg=(
                        f"The cartesian cost scale in this em_tf1 w/ cartesians "
                        f"run increases as expected."
                    ),
                )
                self.console.log(
                    "Training of encodermap_tf1/AngleDihedralCartesianEncoderMap "
                    "w/ cartesian cost is deterministic."
                )
            elif train_tf1_w_cartesians == "skip":
                self.console.log(
                    f"I will not train an encodermap_tf1/AngleDihedralCartesianEncoderMap "
                    f"w/ cartesians. If you want to prove the deterministic training, set "
                    f"`train_tf1_w_cartesians='train_and_check'` and see for yourself."
                )

        if train_tf2_w_cartesians == "train_and_check":
            parameters_tf2 = em.ADCParameters(**agnostic_params)
            parameters_tf2.main_path = em.misc.run_path(str(output_dir / "runs"))
            parameters_tf2.tensorboard = True
            parameters_tf2.summary_step = 1
            parameters_tf2.current_training_step = (
                _agnostic_params_total_steps
                - agnostic_params["cartesian_cost_scale_soft_start"][0]
            )
            parameters_tf2.n_steps = _agnostic_params_total_steps
            # read the weights from the provided wo_cartesians checkpoint
            # this had to be done, because there was no way to consolidate
            # eager and graph execution regarding the saver
            # Third Party Imports
            from tensorflow.python.training import py_checkpoint_reader

            w_cartesians_init_weights = {}
            reader = py_checkpoint_reader.NewCheckpointReader(str(_tf1_checkpoint))
            for i in range(6):
                for type_ in ["kernel", "bias"]:
                    if i == 0:
                        key = f"dense/{type_}"
                    else:
                        key = f"dense_{i}/{type_}"
                    w_cartesians_init_weights[key] = reader.get_tensor(key)

            model = gen_functional_model(
                input_shapes=[s[1:] for s in shapes],
                parameters=parameters_tf2,
                kernel_initializer=w_cartesians_init_weights,
                bias_initializer=w_cartesians_init_weights,
                write_summary=False,
            )

            # check the weights and biases
            i = 0
            for layer in model.encoder_model.layers + model.decoder_model.layers:
                if "dense" not in layer.__class__.__name__.lower():
                    continue
                if i == 0:
                    name = "dense"
                else:
                    name = f"dense_{i}"
                self.assertAllClose(
                    layer.get_weights()[0],
                    w_cartesians_init_weights[f"{name}/kernel"],
                    msg=f"weights: {i=} {layer=}",
                    rtol=1e-3,
                    success_msg=(
                        f"The init_weights which are the final weights of the "
                        f"w/o cartesian run are correctly implemented in this "
                        f"em_tf2 w/ cartesians run, which was done by implementing "
                        f"a custom kernel initializer."
                    ),
                )
                self.assertAllClose(
                    layer.get_weights()[1],
                    w_cartesians_init_weights[f"{name}/bias"],
                    msg=f"bias: {i=} {layer=}",
                    rtol=1e-3,
                    success_msg=(
                        f"The init_weights which are the final bias of the "
                        f"w/o cartesian run are correctly implemented in this "
                        f"em_tf2 w/ cartesians run, which was done by implementing "
                        f"a custom bias initializer."
                    ),
                )
                i += 1

            emap = em.AngleDihedralCartesianEncoderMap(
                trajs=trajs,
                parameters=parameters_tf2,
                model=model,
                dataset=dataset_tf2,
            )

            with Capturing() as output:
                emap.train()
            self.assertTrue(
                any(["References are already provided" in l for l in output]),
                success_msg=(
                    f"em_tf2 AngleDihedralCartesianEncoderMap correctly recognizes "
                    f"that the parameters already have references in them."
                ),
            )

            deterministic_w_cartesians_training = np.load(
                _npz_deterministic_w_cartesians_training_tf1
            )

            keys = set()
            for key in deterministic_w_cartesians_training.keys():
                if "step_0" in key and "cost" in key:
                    keys.add(key)

            mappings = {
                "cost/cartesian_cost": "Cost/Cost_4/Cartesian Cost before scaling",
                "cost/angle_cost": "Cost/Cost_1/Angle Cost",
                "cost/center_cost": "Cost/Cost_7/Center Cost",
                "cost/distance_cost": "Cost/Cost_5/Distance Cost",
                "cost/cartesian_distance_cost": "Cost/Cost_6/Cartesian Distance Cost",
                "cost/combined_cost": "Cost/Combined Cost",
                "cost/dihedral_cost": "Cost/Cost/Dihedral Cost",
                "cost/cartesian_cost_scale": "Cost/Cost_3/Cartesian Cost current scaling",
            }
            mappings = {v: k for k, v in mappings.items()}

            record_files = (Path(emap.p.main_path) / "train").glob("*")
            cartesian_cost_scale = []
            for records_file in record_files:
                records = summary_iterator(str(records_file))
                for i, summary in enumerate(data_loss_iterator(records)):
                    for j, v in enumerate(summary.summary.value):
                        if "Cost" in v.tag:
                            if v.tag not in mappings:
                                continue
                            tf1_tag = f"step_{summary.step}_{mappings[v.tag]}"
                            if tf1_tag not in deterministic_w_cartesians_training:
                                continue
                            tf1_value = float(
                                deterministic_w_cartesians_training[tf1_tag]
                            )
                            ar = float(tf.make_ndarray(v.tensor))
                            self.assertAllClose(
                                tf1_value,
                                ar,
                                rtol=0.5,
                                msg=f"{tf1_tag:<35} TF1: {tf1_value:.4f} TF2: {ar:.4f} is NOT OK",
                                success_msg=f"Successfully compared the cost summary {v.tag} of tf1 and tf2 at step {summary.step}.",
                            )
                        if "current scaling" in v.tag and (
                            summary.step < 10 or summary.step % 25 == 0
                        ):
                            cartesian_cost_scale.append(tf.make_ndarray(v.tensor))
            cartesian_cost_scale = np.array(cartesian_cost_scale)
            should_be = _tf2_cost_scale_should_be
            self.assertAllClose(
                cartesian_cost_scale,
                should_be,
                msg=f"{agnostic_params}",
                success_msg=(f"The cartesian cost scale increases as expected."),
            )

            self.console.log(
                "Training of encodermap_tf2/AngleDihedralCartesianEncoderMap "
                "w/ cartesian cost is deterministic."
            )
        else:
            self.console.log(
                f"I will not train an encodermap_tf2/AngleDihedralCartesianEncoderMap "
                f"w/ cartesians. If you want to prove the deterministic training, set "
                f"`train_tf2_w_cartesians='train_and_check'` and see for yourself."
            )


if __name__ == "__main__":
    unittest.main()
