# -*- coding: utf-8 -*-
# encodermap/models/layers.py
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
"""Module that implements custom layers. Mainly needed for handling periodicity,
backmapping or sparsity."""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
from collections.abc import Sequence
from math import pi
from typing import Any, Optional, Type, TypeVar, Union

# Third Party Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Layer

# Local Folder Imports
from ..encodermap_tf1.backmapping import chain_in_plane
from ..misc import pairwise_dist
from ..misc.backmapping import (
    dihedrals_to_cartesian_tf_layers,
    split_and_reverse_cartesians,
    split_and_reverse_dihedrals,
)
from ..parameters.parameters import ADCParameters, Parameters, ParametersDict


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


################################################################################
# Globals
################################################################################


__all__ = ["PeriodicInput", "PeriodicOutput", "MeanAngles", "BackMapLayer"]


################################################################################
# Layers
################################################################################


# Part of development
# def dihedral_rotation_matrix(dihedral):
#     """
#     Create a rotation matrix for a dihedral angle.
#     """
#     # Ensure that dihedral is a NumPy array with the correct dtype
#     dihedral = np.array(dihedral, dtype=np.float32)
#
#     # Skew-symmetric matrix
#     skew_sym_matrix = tf.constant(
#         [
#             [0.0, -dihedral[0], 0.0],
#             [dihedral[0], 0.0, -dihedral[1]],
#             [0.0, dihedral[1], 0.0],
#         ],
#         dtype=tf.float32,
#     )
#
#     # Exponentiate the skew-symmetric matrix to get the rotation matrix
#     rotation_matrix = tf.eye(3, dtype=tf.float32) + tf.linalg.expm(skew_sym_matrix)
#
#     return rotation_matrix


@tf.keras.saving.register_keras_serializable()
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
            parameters (Union[Parameters, ADCParameters]): An instance of
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


@tf.keras.saving.register_keras_serializable()
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
            parameters (Union[Parameters, ADCParameters]): An instance of
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


@tf.keras.saving.register_keras_serializable()
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


@tf.keras.saving.register_keras_serializable()
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


@tf.keras.saving.register_keras_serializable()
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
            parameters (Union[Parameters, ADCParameters]): An instance of
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


@tf.keras.saving.register_keras_serializable()
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
            parameters (Union[Parameters, ADCParameters]): An instance of
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


@tf.keras.saving.register_keras_serializable()
class PairwiseDistances(EncoderMapBaseLayer):
    """Layer that implements pairwise distances."""

    def __init__(
        self,
        parameters: Union[Parameters, ADCParameters],
        print_name: str,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        """Instantiate the layer.

        Args:
            parameters (Union[Parameters, ADCParameters]): An instance of
                encodermap's parameters.
            print_name (str): The name of this layer, as it should appear
                in summaries.
            trainable (bool): Whether this layer is trainable. As this layer
                has no kernel and/or bias. This argument has no influence.
                Defaults to False.

        """
        super().__init__(parameters, print_name, trainable)

    def call(self, inputs):
        """Call the layer"""
        out = inputs[
            :,
            self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
        ]
        out = pairwise_dist(out, flat=True)
        return out
