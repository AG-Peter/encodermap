# -*- coding: utf-8 -*-
# encodermap/models/layers.py
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
"""Module that implements custom layers. Mainly needed for handling periodicity,
backmapping or sparsity."""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import itertools
from collections.abc import Sequence
from math import pi
from typing import Any, Optional, Type, TypeVar, Union

# Third Party Imports
import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Layer

# Encodermap imports
from encodermap.encodermap_tf1.backmapping import chain_in_plane
from encodermap.loss_functions.loss_classes import testing
from encodermap.misc.backmapping import (
    dihedrals_to_cartesian_tf_layers,
    split_and_reverse_cartesians,
    split_and_reverse_dihedrals,
)
from encodermap.misc.distances import pairwise_dist
from encodermap.parameters.parameters import ADCParameters, Parameters


################################################################################
# Typing
################################################################################


BackMapLayerTransformationsType = TypeVar(
    "BackMapLayerTransformationsType",
    bound="BackMapLayerTransformations",
)
PeriodicOutputType = TypeVar(
    "PeriodicOutputType",
    bound="PeriodicOutput",
)
MeanAnglesType = TypeVar(
    "MeanAnglesType",
    bound="MeanAngles",
)
EncoderMapBaseLayerType = TypeVar(
    "EncoderMapBaseLayerType",
    bound="EncoderMapBaseLayer",
)
BackMapLayerType = TypeVar(
    "BackMapLayerType",
    bound="BackMapLayer",
)
BackMapLayerWithSidechainsType = TypeVar(
    "BackMapLayerWithSidechainsType",
    bound="BackMapLayerWithSidechains",
)


################################################################################
# Globals
################################################################################


__all__: list[str] = ["PeriodicInput", "PeriodicOutput", "MeanAngles", "BackMapLayer"]


################################################################################
# Layers
################################################################################


@tf.keras.utils.register_keras_serializable()
class EncoderMapBaseLayer(Layer):
    """EncoderMap's Base Layer, that implements saving and loading parameters.

    Classes that inherit from `EncoderMapBaseLayer` automatically receive
    parameters when deserialized.

    """

    def __init__(
        self,
        parameters: Union[Parameters, ADCParameters],
        print_name: str,
        trainable: bool = False,
    ) -> None:
        """Instantiate the layer.

        Args:
            parameters (Union[encocermap.parameters.Parameters, encocermap.parameters.ADCParameters]): An instance of
                encodermap's parameters.
            print_name (str): The name of this layer, as it should appear
                in summaries.
            trainable (bool): Whether this layer is trainable. As this layer
                has no kernel and/or bias. This argument has no influence.
                Defaults to False.

        """
        super().__init__(trainable)
        self.p = parameters
        self.print_name = print_name
        self._name = print_name
        self.my_trainable = trainable

    @classmethod
    def from_config(
        cls: Type[EncoderMapBaseLayerType],
        config: dict[Any, Any],
    ) -> EncoderMapBaseLayerType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[Any, Any]): A dictionary.

        Returns:
            EncoderMapBaseLayerType: An instance of the EncoderMapBaseLayer.

        """
        p = config.pop("p")
        if "cartesian_pwd_start" in p:
            p = ADCParameters(**p)
        else:
            p = Parameters(**p)
        return cls(parameters=p, **config)

    def get_config(self) -> dict[Any, Any]:
        """Serializes this keras serializable.

        Returns:
        dict[Any, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "print_name": self.print_name,
                "p": self.p.to_dict(),
                "trainable": self.my_trainable,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class PeriodicInput(EncoderMapBaseLayer):
    """Layer that handles periodic input. Needed, if angles are treated.
    Input angles will be split into sin and cos components,
    and a tensor with shape[0] = 2 * inp_shape[0] will be returned
    """

    def __init__(
        self,
        parameters: Union[Parameters, ADCParameters],
        print_name: str,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        """Instantiate the layer. Need parameters to get the info about the
        periodicity. Although angles are most often used, who knows what hyper-toroidal
        manifold your data lies in.

        Args:
            parameters (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]): An instance of
                encodermap's parameters.
            print_name (str): The name of this layer, as it should appear
                in summaries.
            trainable (bool): Whether this layer is trainable. As this layer
                has no kernel and/or bias. This argument has no influence.
                Defaults to False.

        """
        super().__init__(parameters, print_name, trainable)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call the layer."""
        outputs = inputs
        if self.p.periodicity != 2 * pi:
            outputs = outputs / self.p.periodicity * 2 * pi
        outputs = Concatenate(axis=1, name=f"{self.print_name}_Concat")(
            [
                tf.sin(outputs),
                tf.cos(outputs),
            ]
        )
        return outputs


@tf.keras.utils.register_keras_serializable()
class BackMapLayerWithSidechains(Layer):
    """Also backmaps sidechains. For that, we need a way to know which
    distances, angles, dihedrals belong to the backbone, and which belong to
    a sidechain. See the docstring of `encodermap.misc.backmapping._full_backmapping_np`
    for details.

    """

    def __init__(
        self,
        feature_description: Any,
    ) -> None:
        super().__init__()
        self.feature_description: dict[Any, Any] = feature_description

        # Definitions and Tests
        n_residues: int = max(list(feature_description[-1].keys()))
        assert np.array_equal(
            np.arange(1, n_residues + 1),
            np.sort(np.asarray(list(feature_description[-1].keys()))),
        ), (
            f"Currently the `feature_indices[-1]` dict needs to contain monotonous "
            f"increasing keys. Starting from 1 {feature_description[-1].keys()=}"
        )
        n_sidechains: int = sum(
            [v + 1 for v in feature_description[-1].values() if v > 0]
        )
        sum_sidechains = sum(list(feature_description[-1].values()))

        # this can be defined beforehand and then stacked as often, as a batch needs it
        self.init_xyz: tf.Tensor = tf.zeros(
            shape=(1, n_residues * 3 + n_sidechains, 3),
            dtype=tf.float32,
        )

        # first we create the central_distance indices
        central_distance_indices = np.tri(
            N=n_residues * 3 - 1,
            M=n_residues * 3,
            k=0,
        ).astype(bool)
        right_side_central_distance_indices = [
            np.full(shape=(1, n_sidechains), fill_value=False, dtype=bool)
        ]
        count = 0  # starts at the first atom of the central chan
        count2 = n_residues * 3 + 1  # starts at the first atom of the sidechain
        sidechain_cartesians_ind = []
        sidechain_positions_indices = []
        central_angle_index_triplets = np.vstack(
            [
                np.arange(0, n_residues * 3)[:-2],
                np.arange(0, n_residues * 3)[1:-1],
                np.arange(0, n_residues * 3)[2:],
            ]
        ).T.tolist()
        sidechain_angle_index_triplets = []
        central_dihedral_index_quadruplets = np.vstack(
            [
                np.arange(0, n_residues * 3)[:-3],
                np.arange(0, n_residues * 3)[1:-2],
                np.arange(0, n_residues * 3)[2:-1],
                np.arange(0, n_residues * 3)[3:],
            ]
        ).T.tolist()
        sidechain_dihedral_index_quadruplets = []

        # iterate over feature_description[-1] to get all indices and the right side
        # of the central cartesians
        for i, (residue, n_sidechains_in_residue) in zip(
            itertools.count(1, 3), feature_description[-1].items()
        ):
            if n_sidechains_in_residue == 0:
                if residue == 1 or residue == n_residues:
                    continue
                else:
                    right_side_central_distance_indices.append(t)
            else:
                sidechain_cartesians_ind.append(
                    np.arange(count, count + n_sidechains_in_residue)
                )
                sidechain_positions_indices.append(
                    [i]
                    + np.arange(count2 - 1, count2 + n_sidechains_in_residue).tolist()
                )
                for sidechain_i in range(n_sidechains_in_residue + 1):
                    if sidechain_i == 0:
                        # adds N-CA-CB
                        sidechain_angle_index_triplets.append(
                            [(residue - 1) * 3, (residue - 1) * 3 + 1, count2 - 1]
                        )
                        # adds N-CA-CB-CG
                        sidechain_dihedral_index_quadruplets.append(
                            [
                                (residue - 1) * 3,
                                (residue - 1) * 3 + 1,
                                count2 - 1,
                                count2,
                            ]
                        )
                    elif sidechain_i == 1:
                        # adds CA-CB-CG
                        sidechain_angle_index_triplets.append(
                            [(residue - 1) * 3 + 1, count2 - 1, count2]
                        )
                        # adds CA-CB-CG-CD
                        if sidechain_i < n_sidechains_in_residue:
                            sidechain_dihedral_index_quadruplets.append(
                                [(residue - 1) * 3 + 1, count2 - 1, count2, count2 + 1]
                            )
                    else:
                        # adds CB-CG-CD and so on
                        sidechain_angle_index_triplets.append(
                            [
                                count2 + sidechain_i - 3,
                                count2 + sidechain_i - 2,
                                count2 + sidechain_i - 1,
                            ]
                        )
                        if sidechain_i < n_sidechains_in_residue:
                            sidechain_dihedral_index_quadruplets.append(
                                [
                                    count2 + sidechain_i - 3,
                                    count2 + sidechain_i - 2,
                                    count2 + sidechain_i - 1,
                                    count2 + sidechain_i,
                                ]
                            )
                count += n_sidechains_in_residue + 1
                count2 += n_sidechains_in_residue + 1
                t = np.zeros(
                    shape=(3, n_sidechains),
                    dtype=bool,
                )
                t[:, :count] = True
                right_side_central_distance_indices.append(t)
        assert len(sidechain_angle_index_triplets) == n_sidechains
        assert len(sidechain_dihedral_index_quadruplets) == sum_sidechains, (
            f"I could not reconstruct the correct number of sidechain dihedral "
            f"quadruplets. The number of sidechain dihedrals requires the list "
            f"to have length {sum_sidechains}, but I created a list with "
            f"{len(sidechain_dihedral_index_quadruplets)}."
        )
        right_side_central_distance_indices.append(
            np.full(shape=(1, n_sidechains), fill_value=True, dtype=bool)
        )
        right_side_central_distance_indices = np.vstack(
            right_side_central_distance_indices
        )
        angle_index_triplets = np.vstack(
            central_angle_index_triplets + sidechain_angle_index_triplets
        )
        dihedral_index_quadruplets = np.vstack(
            central_dihedral_index_quadruplets + sidechain_dihedral_index_quadruplets
        )
        if sidechain_cartesians_ind != []:  # if sidechains
            _use_sidechains = True
            sidechain_cartesians_ind = np.concatenate(sidechain_cartesians_ind)
            central_distance_indices = np.hstack(
                [central_distance_indices, right_side_central_distance_indices]
            )
            side_distance_indices = [
                (np.tri(N=i + 1, M=i + 2, k=0) + 1)[:, 1:]
                for i in feature_description[-1].values()
                if i > 0
            ]
            side_distance_indices = (block_diag(*side_distance_indices) % 2) == 0
            left_side_side_distance_indices = (
                np.full(  # all atoms in the central chain are True
                    shape=(len(side_distance_indices), n_residues * 3),
                    fill_value=True,
                    dtype=bool,
                )
            )
            side_distance_indices = np.hstack(
                [left_side_side_distance_indices, side_distance_indices]
            )
            distance_indices = np.vstack(
                [central_distance_indices, side_distance_indices]
            )
        else:  # if no sidechains
            _use_sidechains = False
            distance_indices = central_distance_indices
        assert distance_indices.shape == (
            n_residues * 3 - 1 + n_sidechains,
            self.init_xyz.shape[1],
        ), (
            f"The shape of the distance index after stacking is unexpected.\n"
            f"Expected: {(n_residues * 3 - 1 + n_sidechains, self.init_xyz.shape[1])}\n"
            f"Actual: {distance_indices.shape}"
        )

        # now the angles
        central_angle_indices = central_distance_indices[1:]
        if _use_sidechains:  # if sidechains
            angle_indices = np.vstack(
                [central_distance_indices[1:], side_distance_indices]
            )
            side_angle_indices = side_distance_indices
        else:  # no sidechains
            angle_indices = central_distance_indices[1:]
        assert len(angle_indices) == len(distance_indices) - 1

        # and the dihedrals
        if _use_sidechains:  # if sidechains
            dihedral_indices = np.vstack(
                [
                    central_distance_indices[1:-1],
                    side_distance_indices[sidechain_cartesians_ind],
                ]
            )
            corrector = np.count_nonzero(
                list(feature_description[-1].values())
            )  # per reisude with sidechain dihedrals one less
        else:
            dihedral_indices = central_distance_indices[1:-1]
            corrector = 0
        assert len(dihedral_indices) == len(distance_indices) - 2 - corrector
        assert angle_index_triplets.shape[0] == angle_indices.shape[0]
        assert dihedral_index_quadruplets.shape[0] == dihedral_indices.shape[0], (
            f"The number of dihedral indices ({len(distance_indices)}) and quadruplets "
            f"does not match ({len(dihedral_index_quadruplets)})."
        )

        # create instance attributes with tf
        self._use_sidechains = tf.constant(_use_sidechains, dtype=tf.bool)
        self.n_sidechains_in_residue = tf.constant(
            np.array(
                [
                    self.feature_description[-1][k]
                    for k in sorted(self.feature_description[-1].keys())
                ]
            ).astype(np.int32),
            dtype=tf.int32,
        )

        # general
        self.up = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)
        self.down = tf.constant([[0.0, 0.0, -1.0]], dtype=tf.float32)

        # distances
        self.no_of_central_distances = central_distance_indices.shape[0]
        self.n_sidechains = n_sidechains
        self.central_distance_indices = tf.constant(
            central_distance_indices,
            shape=central_distance_indices.shape,
            dtype=tf.bool,
        )
        self.n_atoms = self.central_distance_indices.shape[1]

        # angles
        self.no_of_central_angles = central_angle_indices.shape[0]
        self.no_of_side_angles = side_angle_indices.shape[0]
        self.central_angle_indices = tf.constant(
            central_angle_indices, shape=central_angle_indices.shape, dtype=tf.bool
        )
        self.side_angle_indices = tf.constant(
            side_angle_indices, shape=side_angle_indices.shape, dtype=tf.bool
        )
        self.central_angle_index_triplets = tf.constant(
            np.asarray(central_angle_index_triplets),
            shape=(len(central_angle_index_triplets), 3),
            dtype=tf.int32,
        )
        self.sidechain_angle_index_triplets = tf.constant(
            np.asarray(sidechain_angle_index_triplets),
            shape=(len(sidechain_angle_index_triplets), 3),
            dtype=tf.int32,
        )

        # dihedrals
        self.no_of_dihedrals = dihedral_indices.shape[0]
        self.dihedral_indices = tf.constant(
            dihedral_indices, shape=dihedral_indices.shape, dtype=tf.bool
        )
        self.dihedral_index_quadruplets = tf.constant(
            np.asarray(dihedral_index_quadruplets),
            shape=(len(dihedral_index_quadruplets), 4),
            dtype=tf.int32,
        )

    def get_config(self) -> dict[Any, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[Any, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "feature_description": self.feature_description,
            }
        )
        return config

    @classmethod
    def from_config(
        cls: Type[BackMapLayerWithSidechainsType],
        config: dict[Any, Any],
    ) -> BackMapLayerWithSidechainsType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[Any, Any]): A dictionary.

        Returns:
            BackMapLayerType: An instance of the BackMapLayer.

        """
        feature_description = config.pop("feature_description")
        out = {int(k): v for k, v in feature_description.items()}
        for k, v in out.items():
            out[k] = {int(kv): vv for kv, vv in v.items()}
        return cls(feature_description=out)

    def call(self, inputs: tuple[tf.Tensor, ...]) -> tf.Tensor:
        # Unpack inputs
        (
            central_distances,
            central_angles,
            central_dihedrals,
            side_distances,
            side_angles,
            side_dihedrals,
        ) = inputs

        # concatenate the dihedrals
        dihedrals = tf.concat(
            [
                central_dihedrals,
                side_dihedrals,
            ],
            axis=1,
        )

        # distances
        xs_central = tf.TensorArray(
            dtype=tf.float32,
            size=self.no_of_central_distances + 1,
            clear_after_read=False,
        )
        ys_central = tf.TensorArray(
            dtype=tf.float32,
            size=self.no_of_central_distances + 1,
            clear_after_read=False,
        )
        xs_side = tf.TensorArray(
            dtype=tf.float32,
            size=self.n_sidechains,
            clear_after_read=False,
        )
        ys_side = tf.TensorArray(
            dtype=tf.float32,
            size=self.n_sidechains,
            clear_after_read=False,
        )
        xs_central = xs_central.write(
            0, tf.zeros((tf.shape(central_angles)[0],), dtype=tf.float32)
        )
        ys_central = ys_central.write(
            0, tf.zeros((tf.shape(central_angles)[0],), dtype=tf.float32)
        )
        # xs_central = [tf.zeros((tf.shape(central_angles)[0], ), dtype=tf.float32)]
        # ys_central = [tf.zeros((tf.shape(central_angles)[0], ), dtype=tf.float32)]
        # xs_side = []
        # ys_side = []

        residue = 0
        idx = 0
        j = 0
        for i in range(self.no_of_central_distances):
            # xs_central.append(xs[-1] + central_distances[:, 1])
            xs_central = xs_central.write(
                i + 1, xs_central.read(i) + central_distances[:, i]
            )
            # ys_central.append(tf.zeros((tf.shape(central_angles)[0], ), dtype=tf.float32))
            ys_central = ys_central.write(
                i + 1, tf.zeros((tf.shape(central_angles)[0],))
            )
            if idx == 0 and self._use_sidechains:
                n_sidechains = self.n_sidechains_in_residue[residue]
                if n_sidechains > 0:
                    for n in range(n_sidechains + 1):
                        # xs_side.append(xs_central.read(i))
                        xs_side = xs_side.write(j, xs_central.read(i + 1))
                        # ys_side.append(
                        #     tf.reduce_sum(
                        #         side_distances[:, j - n : j + 1],
                        #         axis=1,
                        #     )
                        # )
                        ys_side = ys_side.write(
                            j,
                            tf.reduce_sum(
                                side_distances[:, j - n : j + 1],
                                axis=1,
                            ),
                        )
                        j += 1
            idx += 1
            if idx >= 3:
                residue += 1
                idx = 0
        xs_central = tf.transpose(xs_central.stack(), perm=[1, 0])
        ys_central = tf.transpose(ys_central.stack(), perm=[1, 0])
        xs_side = tf.transpose(xs_side.stack(), perm=[1, 0])
        ys_side = tf.transpose(ys_side.stack(), perm=[1, 0])
        xs_side.set_shape((xs_central.shape[0], self.n_sidechains))
        ys_side.set_shape((xs_central.shape[0], self.n_sidechains))
        xs = tf.concat([xs_central, xs_side], axis=1)
        ys = tf.concat([ys_central, ys_side], axis=1)
        xyz_out = tf.stack(
            [
                xs,
                ys,
            ],
            axis=2,
        )
        xyz_out = tf.pad(
            tf.pad(
                xyz_out,
                ((0, 0), (0, 0), (0, 1)),
                constant_values=0,
            ),
            paddings=((0, 0), (0, 0), (0, 1)),
            constant_values=1,
        )

        # angles
        # Can't parallelize over angles (just over batch dimension)
        # because xyz_out is updated constantly and thus
        # xyz_out[..., -1] changes during iteration
        for i in range(self.no_of_central_angles):
            ind = self.central_angle_indices[i]
            angle_index = self.central_angle_index_triplets[i]
            ang = central_angles[:, i]
            direction = tf.repeat(
                self.up,
                repeats=tf.shape(ang)[0],
                axis=0,
            )
            abc = tf.transpose(
                tf.gather(
                    params=xyz_out,
                    indices=angle_index,
                    axis=1,
                    batch_dims=0,
                )[..., :3],
                perm=[1, 0, 2],
            )
            ba = abc[0] - abc[1]
            bc = abc[2] - abc[1]
            dot = tf.keras.backend.batch_dot(
                ba,
                bc,
            )
            prod = tf.expand_dims(_batch_fro(ba) * _batch_fro(bc), axis=1)
            t = tf.clip_by_value(dot / prod, clip_value_min=-1, clip_value_max=1)
            current_angle = tf.squeeze(tf.acos(t))
            angle = tf.abs(ang - current_angle)
            rotmat = _rotation_matrices(
                angle=angle,
                direction=direction,
                point=abc[1],
            )
            dynamic = tf.transpose(
                tf.gather(
                    params=xyz_out, indices=tf.where(~ind)[:, 0], axis=1, batch_dims=0
                ),
                perm=[0, 2, 1],
            )
            rotated = tf.transpose(
                tf.keras.backend.batch_dot(rotmat, dynamic),
                perm=[0, 2, 1],
            )
            static = tf.gather(
                params=xyz_out, indices=tf.where(ind)[:, 0], axis=1, batch_dims=0
            )
            new = tf.TensorArray(
                dtype=tf.float32,
                size=self.no_of_central_distances + 1 + self.n_sidechains,
                clear_after_read=False,
            )
            d = 0
            s = 0
            c = 0
            for j in ind:
                if j:
                    new = new.write(c, static[:, s])
                    s += 1
                else:
                    new = new.write(c, rotated[:, d])
                    d += 1
                c += 1
            xyz_out = tf.transpose(
                new.stack(),
                perm=[1, 0, 2],
            )

        # sidechains
        for i in range(self.no_of_side_angles):
            ind = self.side_angle_indices[i]
            angle_index = self.sidechain_angle_index_triplets[i]
            ang = side_angles[:, i]
            direction = tf.repeat(
                self.down,
                repeats=tf.shape(ang)[0],
                axis=0,
            )
            abc = tf.transpose(
                tf.gather(
                    params=xyz_out,
                    indices=angle_index,
                    axis=1,
                    batch_dims=0,
                )[..., :3],
                perm=[1, 0, 2],
            )
            ba = abc[0] - abc[1]
            bc = abc[2] - abc[1]
            dot = tf.keras.backend.batch_dot(
                ba,
                bc,
            )
            prod = tf.expand_dims(_batch_fro(ba) * _batch_fro(bc), axis=1)
            t = tf.clip_by_value(dot / prod, clip_value_min=-1, clip_value_max=1)
            current_angle = tf.squeeze(tf.acos(t))
            angle = tf.abs(ang - current_angle)
            rotmat = _rotation_matrices(
                angle=angle,
                direction=direction,
                point=abc[1],
            )
            dynamic = tf.transpose(
                tf.gather(
                    params=xyz_out, indices=tf.where(~ind)[:, 0], axis=1, batch_dims=0
                ),
                perm=[0, 2, 1],
            )
            rotated = tf.transpose(
                tf.keras.backend.batch_dot(rotmat, dynamic),
                perm=[0, 2, 1],
            )
            static = tf.gather(
                params=xyz_out, indices=tf.where(ind)[:, 0], axis=1, batch_dims=0
            )
            new = tf.TensorArray(
                dtype=tf.float32,
                size=self.no_of_central_distances + 1 + self.n_sidechains,
                clear_after_read=False,
            )
            d = 0
            s = 0
            c = 0
            for j in ind:
                if j:
                    new = new.write(c, static[:, s])
                    s += 1
                else:
                    new = new.write(c, rotated[:, d])
                    d += 1
                c += 1
            xyz_out = tf.transpose(
                new.stack(),
                perm=[1, 0, 2],
            )

        # dihedrals
        for i in range(self.no_of_dihedrals):
            ind = self.dihedral_indices[i]
            dihedral_index = self.dihedral_index_quadruplets[i]
            ang = dihedrals[:, i]
            abcd = tf.transpose(
                tf.gather(
                    params=xyz_out,
                    indices=dihedral_index,
                    axis=1,
                    batch_dims=0,
                )[..., :3],
                perm=[1, 0, 2],
            )
            direction = abcd[2] - abcd[1]
            b1 = abcd[1] - abcd[0]
            b2 = abcd[2] - abcd[1]
            b3 = abcd[3] - abcd[2]
            c1 = tf.linalg.cross(b2, b3)
            c2 = tf.linalg.cross(b1, b2)
            p1 = tf.reduce_sum((b1 * c1), axis=1)
            p1 *= tf.sqrt(tf.reduce_sum((b2 * b2), axis=1))
            p2 = tf.reduce_sum((c1 * c2), axis=1)
            current_angle = tf.atan2(p1, p2)
            angle = ang - current_angle
            rotmat = _rotation_matrices(
                angle=angle,
                direction=direction,
                point=abcd[1],
            )
            dynamic = tf.transpose(
                tf.gather(
                    params=xyz_out, indices=tf.where(~ind)[:, 0], axis=1, batch_dims=0
                ),
                perm=[0, 2, 1],
            )
            rotated = tf.transpose(
                tf.keras.backend.batch_dot(rotmat, dynamic),
                perm=[0, 2, 1],
            )
            static = tf.gather(
                params=xyz_out, indices=tf.where(ind)[:, 0], axis=1, batch_dims=0
            )
            new = tf.TensorArray(
                dtype=tf.float32,
                size=self.no_of_central_distances + 1 + self.n_sidechains,
                clear_after_read=False,
            )
            d = 0
            s = 0
            c = 0
            for j in ind:
                if j:
                    new = new.write(c, static[:, s])
                    s += 1
                else:
                    new = new.write(c, rotated[:, d])
                    d += 1
                c += 1
            xyz_out = tf.transpose(
                new.stack(),
                perm=[1, 0, 2],
            )

        return xyz_out[..., :3]


@tf.function
def _batch_fro(a: tf.Tensor) -> tf.Tensor:
    """Batch-wise Frobert norm, a.k.a. length of a vector."""
    return tf.sqrt(tf.reduce_sum(a**2, axis=1))


@tf.function
def _rotation_matrices(angle, direction, point) -> tf.Tensor:
    """Adapted from C. Gohlke's transformations.py.

    Batch-wise 4x4 rotation matrices.

    """
    sina = tf.sin(angle)
    cosa = tf.cos(angle)
    direction_u = _unit_vector(direction)

    # rotation matrix around unit vector
    R = tf.linalg.diag(tf.transpose([cosa, cosa, cosa]), k=0)
    R += tf.einsum("ki,kj->kij", direction_u, direction_u) * tf.expand_dims(
        tf.expand_dims(1.0 - cosa, -1), -1
    )
    direction_u *= tf.expand_dims(sina, -1)

    R_add = tf.TensorArray(
        dtype=tf.float32,
        size=tf.shape(angle)[0],
        clear_after_read=False,
    )
    for i in range(tf.shape(angle)[0]):
        d = direction_u[i]
        R_add = R_add.write(
            i, [[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]]
        )
    R_add = R_add.stack()
    R += R_add
    R.set_shape((angle.shape[0], 3, 3))

    # rotation around origin
    test = tf.expand_dims(
        tf.pad(
            point - tf.keras.backend.batch_dot(R, point),
            paddings=((0, 0), (0, 1)),
            constant_values=1,
        ),
        axis=-1,
    )
    R = tf.pad(R, ((0, 0), (0, 1), (0, 0)))
    M = tf.concat([R, test], axis=2)
    return M


@tf.function
def _unit_vector(vector: tf.Tensor) -> tf.Tensor:
    """Adapted from C. Gohlke's transformations.py"""
    length = tf.sqrt(tf.reduce_sum(vector**2, axis=1))
    return vector / tf.expand_dims(length, 1)


@tf.keras.utils.register_keras_serializable()
class BackMapLayer(Layer):
    """Layer that implements backmapping from torsions-angles-distances to Euclidean coordinates."""

    def __init__(self, left_split: int, right_split: int) -> None:
        """Instantiate the layer."""
        super().__init__()
        self._name = "BackmapLayer"
        self.left_split = left_split
        self.right_split = right_split

    @classmethod
    def from_config(
        cls: Type[BackMapLayerType],
        config: dict[Any, Any],
    ) -> BackMapLayerType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[Any, Any]): A dictionary.

        Returns:
            BackMapLayerType: An instance of the BackMapLayer.

        """
        left_split = config.pop("left_split")
        right_split = config.pop("right_split")
        return cls(left_split=left_split, right_split=right_split)

    def get_config(self) -> dict[Any, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[Any, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "left_split": self.left_split,
                "right_split": self.right_split,
            }
        )
        return config

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Call the layers, inputs should be a tuple shaped, so that it can be split into
        distances, angles, dihedrals = inputs
        """
        distances, angles, dihedrals = inputs
        # I don't know how negative distances can arrive at this step
        # but we replace them with the mean
        fixed_distances = tf.where(
            distances < 0.00001,
            tf.ones_like(distances) * tf.reduce_mean(distances),
            distances,
        )
        # mean lengths
        # back_mean_lengths = tf.expand_dims(tf.reduce_mean(inp_distances, 0), 0)
        out = tf.expand_dims(tf.reduce_mean(distances, 0), 0)

        # chain in plane
        # back_chain_in_plane = chain_in_plane(back_mean_lengths, out_angles)
        out = chain_in_plane(out, angles)

        # dihedrals to cartesian
        # back_cartesians = dihedrals_to_cartesian_tf(out_dihedrals + pi, back_chain_in_plane)
        out_dihedrals = tf.add(dihedrals, pi)
        out = dihedrals_to_cartesian_tf_layers(
            out_dihedrals,
            out,
            left_iteration_counter=self.left_split,
            right_iteration_counter=self.right_split,
        )
        return out


@tf.keras.utils.register_keras_serializable()
class BackMapLayerTransformations(Layer):
    """Experimental layer for using multimers with the ADCEMap."""

    def __init__(self, protein_lengths: Sequence[int]) -> None:
        """Instantiate the layer.

        Args:
            protein_lengths (Sequence[int]): The lengths of the proteins in the
                multimers. Based on this information, the input to `self.call`
                will be split.

        """
        self.protein_lengths = protein_lengths
        super().__init__()
        self._name = "BackmapLayerTransformations"

    @classmethod
    def from_config(
        cls: Type[BackMapLayerTransformationsType],
        config: dict[Any, Any],
    ) -> BackMapLayerTransformationsType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[Any, Any]): A dictionary.

        Returns:
            BackMapLayerTransformationsType: An instance of the BackMapLayerTransformations.

        """
        protein_lengths = config.pop("protein_lengths")
        return cls(protein_lengths=protein_lengths, **config)

    def get_config(self) -> dict[Any, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[Any, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "protein_lengths": self.protein_lengths,
            }
        )
        return config

    def call(self, inputs):
        """Call the layers, inputs should be a tuple shaped, so that it can be split into
        distances, angles, dihedrals, matrices = inputs
        """
        # Third Party Imports
        from tensorflow_graphics.rendering.utils import transform_homogeneous

        inp_distances, out_angles, out_dihedrals, matrices = inputs

        out_cartesians = []
        current_length = 0
        for i, protein_length in enumerate(self.protein_lengths):
            if current_length == 0:
                distance_ind = slice(0, protein_length * 3 - 1)
                angle_ind = slice(0, protein_length * 3 - 2)
                dihe_ind = slice(0, protein_length * 3 - 3)
            else:
                distance_ind = slice(
                    current_length * 3 - i,
                    current_length * 3 + protein_length * 3 - (i + 1),
                )
                angle_ind = slice(
                    current_length * 3 - (i + 1),
                    current_length * 3 + protein_length * 3 - (i + 2),
                )
                dihe_ind = slice(
                    current_length * 3 - (i + 2),
                    current_length * 3 + protein_length * 3 - (i + 3),
                )
                current_length += protein_length

            # index
            current_lengths = inp_distances[:, distance_ind]
            current_lengths = tf.expand_dims(tf.reduce_mean(current_lengths, 0), 0)
            current_angles = out_angles[:, angle_ind]
            current_dihedrals = out_dihedrals[:, dihe_ind]
            current_dihedrals = tf.add(current_dihedrals, pi)

            c = chain_in_plane(current_lengths, current_angles)
            c = dihedrals_to_cartesian_tf_layers(current_dihedrals, c)

            # for all other proteins apply homogeneous transformation matrices
            if i != 0:
                m = matrices[:, i - 1]
                c = transform_homogeneous(m, c)[..., :3]

            out_cartesians.append(c)

        out_cartesians = tf.concat(
            out_cartesians,
            axis=1,
        )
        return out_cartesians


@tf.keras.utils.register_keras_serializable()
class PeriodicOutput(EncoderMapBaseLayer):
    """Layer that reverses the PeriodicInputLayer."""

    def __init__(
        self,
        parameters: Union[Parameters, ADCParameters],
        print_name: str,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        """Instantiate the layer, We also need to know here what periodicity is needed.

        Args:
            parameters (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]): An instance of
                encodermap's parameters.
            print_name (str): The name of this layer, as it should appear
                in summaries.
            trainable (bool): Whether this layer is trainable. As this layer
                has no kernel and/or bias. This argument has no influence.
                Defaults to False.

        """
        super().__init__(parameters, print_name, trainable)

    def call(self, inputs):
        """Calls the layer. Inputs should be a tuple of (sin, cos) of the same angles"""
        outputs = inputs
        outputs = tf.atan2(*tf.split(outputs, 2, 1))

        if self.p.periodicity != 2 * pi:
            outputs = outputs / (2 * pi) * self.p.periodicity
        return outputs


@tf.keras.utils.register_keras_serializable()
class MeanAngles(Layer):
    """Layer that implements the mean of periodic angles."""

    def __init__(
        self,
        parameters: Union[Parameters, ADCParameters],
        print_name: str,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        """Instantiate the layer.

        Args:
            parameters (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]): An instance of
                encodermap's parameters.
            print_name (str): The name of this layer, as it should appear
                in summaries.
            trainable (bool): Whether this layer is trainable. As this layer
                has no kernel and/or bias. This argument has no influence.
                Defaults to False.

        """
        super().__init__(trainable=trainable)

    def call(self, inputs):
        """Call the layer"""
        return tf.tile(
            tf.expand_dims(
                tf.math.reduce_mean(inputs, 0),
                0,
            ),
            multiples=(tf.shape(inputs)[0], 1),
        )


@tf.keras.utils.register_keras_serializable()
class PairwiseDistances(EncoderMapBaseLayer):
    """Layer that implements pairwise distances for both cases, with and without sidechain reconstruction"""

    def __init__(
        self,
        parameters: Union[Parameters, ADCParameters],
        print_name: str,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        """Instantiate the layer.

        Args:
            parameters (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]): An instance of
                encodermap's parameters.
            print_name (str): The name of this layer, as it should appear
                in summaries.
            trainable (bool): Whether this layer is trainable. As this layer
                has no kernel and/or bias. This argument has no influence.
                Defaults to False.

        """
        super().__init__(parameters, print_name, trainable)
        if self.p.reconstruct_sidechains:
            assert hasattr(self.p, "sidechain_info"), (
                "The provided parameters ask for sidechains to be reconstructed, "
                "but don't contain a 'sidechain_info' attribute."
            )
            self.indices = None
            n_residues = max(list(self.p.sidechain_info[-1].keys()))
            self.indices = np.arange(n_residues * 3)[
                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step
            ]
            atom = n_residues * 3 + 1
            indices = []
            for residue, n_sidechains_in_residue in self.p.sidechain_info[-1].items():
                if n_sidechains_in_residue == 0:
                    continue
                if residue == 1:
                    atom += n_sidechains_in_residue
                else:
                    atom += n_sidechains_in_residue
                indices.append(atom)
            self.indices = np.concatenate([self.indices, indices])

    def get_config(self) -> dict[Any, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[Any, Any]: A dict with the serializable objects.

        """
        sidechain_info = self.p.sidechain_info
        config = super().get_config().copy()
        config.update(
            {
                "sidechain_info": sidechain_info,
            }
        )
        return config

    @classmethod
    def from_config(
        cls: Type[BackMapLayerWithSidechainsType],
        config: dict[Any, Any],
    ) -> BackMapLayerWithSidechainsType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[Any, Any]): A dictionary.

        Returns:
            BackMapLayerType: An instance of the BackMapLayer.

        """
        p = config.pop("p")
        if "cartesian_pwd_start" in p:
            p = ADCParameters(**p)
        else:
            p = Parameters(**p)
        sidechain_info = config.pop("sidechain_info")
        out = {int(k): v for k, v in sidechain_info.items()}
        for k, v in out.items():
            out[k] = {int(kv): vv for kv, vv in v.items()}
        p.sidechain_info = out
        return cls(parameters=p, **config)

    def call(self, inputs):
        """Call the layer"""
        if not self.p.reconstruct_sidechains:
            out = inputs[
                :,
                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
            ]
        else:
            out = tf.gather(
                params=inputs,
                indices=self.indices,
                axis=1,
                batch_dims=0,
            )
        out = pairwise_dist(out, flat=True)
        return out
