# -*- coding: utf-8 -*-
# encodermap/models/layers.py
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
"""Module that implements custom layers. Mainly needed for handling periodicity,
backmapping or sparsity."""


################################################################################
# Imports
################################################################################


from __future__ import annotations

from math import pi

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Layer

from ..encodermap_tf1.backmapping import chain_in_plane
from ..misc import pairwise_dist
from ..misc.backmapping import (
    dihedrals_to_cartesian_tf_layers,
    split_and_reverse_cartesians,
    split_and_reverse_dihedrals,
)

################################################################################
# Globals
################################################################################


__all__ = ["PeriodicInput", "PeriodicOutput", "MeanAngles", "BackMapLayer", "Sparse"]


################################################################################
# Layers
################################################################################


class Sparse(Dense):
    """Simple subclass of tf.keras.layers.Dense, which implements sparse_dense_matmul"""

    def call(self, inputs):
        """Call the layer."""
        outputs = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs


class SparseReshape(tf.keras.layers.Reshape):
    """Layer that can reshapa a sparse Tensor."""

    def call(self, inputs):
        return tf.reshape(inputs, shape=(self.target_shape,))


class PeriodicInput(Layer):
    """Layer that handles periodic input. Needed, if angles are treated. Input angles
    will be split into sin and cos components and a tensor with shape[0] = 2 * inp_shape[0]
    will be returned
    """

    def __init__(self, parameters, print_name, trainable=False):
        """Instantiate the layer. Need parameters to get the info about the
        periodicity. Although angles are most often used, who knows what hyper-toroidal
        manifold your data lies in.

        """
        super().__init__(trainable)
        self.p = parameters
        self.print_name = print_name
        self._name = self.print_name + "_Periodic_Input"

    def call(self, inputs):
        """Call the layer."""
        outputs = inputs
        if self.p.periodicity != 2 * pi:
            outputs = Lambda(
                lambda x: x / self.p.periodicity * 2 * pi,
                name=f"{self.print_name}_Periodicity_to_2_pi",
            )(outputs)
        outputs = Concatenate(axis=1, name=f"{self.print_name}_Concat")(
            [
                Lambda(lambda x: tf.sin(x), name=f"{self.print_name}_sin")(outputs),
                Lambda(lambda x: tf.cos(x), name=f"{self.print_name}_cos")(outputs),
            ]
        )
        return outputs


class BackMapLayer(Layer):
    """Layer that implements backmapping from torsions-angles-distances to euclidean coordinates."""

    def __init__(self):
        """Instantiate the layer"""
        super().__init__()
        self._name = "Backmap_Layer"

    def call(self, inputs):
        """Call the layers, inputs should be a tuple shaped, so that it can be split into
        distances, angles, dihedrals = inputs
        """
        inp_distances, out_angles, out_dihedrals = inputs
        # mean lengths
        # back_mean_lengths = tf.expand_dims(tf.reduce_mean(inp_distances, 0), 0)
        out = Lambda(
            lambda x: tf.expand_dims(tf.reduce_mean(x, 0), 0), name="Back_Mean_Lengths"
        )(inp_distances)

        # chain in plane
        # back_chain_in_plane = chain_in_plane(back_mean_lengths, out_angles)
        out = Lambda(lambda x: chain_in_plane(x[0], x[1]), name="Back_Chain_in_Plane")(
            (out, out_angles)
        )

        # dihedrals to cartesian
        # back_cartesians = dihedrals_to_cartesian_tf(out_dihedrals + pi, back_chain_in_plane)
        out_dihedrals = Lambda(lambda x: tf.add(x, pi), name="Added_Pi")(out_dihedrals)
        out = Lambda(
            lambda x: dihedrals_to_cartesian_tf_layers(x[0], x[1]),
            name="Back_Cartesians",
        )((out_dihedrals, out))
        return out


class PeriodicOutput(Layer):
    """Layer that reverses the PeriodicInputLayer."""

    def __init__(self, parameters, print_name, trainable=False):
        """Instantiate the layer, We also need to know here, what periodicity is needed."""
        super().__init__(trainable)
        self.p = parameters
        self.print_name = print_name
        self._name = self.print_name + "_Periodic_Output"

    def call(self, inputs):
        """Calls the layer, Inputs shold be a tuple of (sin, cos) of the same angles"""
        outputs = inputs
        outputs = Lambda(
            lambda x: tf.atan2(*tf.split(x, 2, 1)),
            name=f"{self.print_name}_Fom_Unitcircle",
        )(outputs)
        if self.p.periodicity != 2 * pi:
            outputs = Lambda(
                lambda x: x / (2 * pi) * self.p.periodicity,
                name=f"{self.print_name}_Periodicity_from_2_pi",
            )(outputs)
        return outputs


class MeanAngles(Layer):
    """Layer that implements the mean of periodic angles."""

    def __init__(self, parameters, print_name, multiples_shape):
        """Instantiate the layer."""
        super().__init__()
        self.p = parameters
        self.print_name = print_name
        self.multiples_shape = multiples_shape
        self._name = self.print_name

    def call(self, inputs):
        """Call the layer"""
        outputs = Lambda(
            lambda x: tf.tile(
                tf.expand_dims(tf.math.reduce_mean(x, 0), 0),
                multiples=(self.multiples_shape, 1),
            ),
            name=self.print_name,
        )(inputs)
        return outputs
        # out_angles = tf.tile(np.expand_dims(np.mean(angles, 0), 0), multiples=(out_dihedrals.shape[0], 1))


class PairwiseDistances(Layer):
    """Layer that implements pairwise distances."""

    def __init__(self, parameters, print_name):
        """Instantiate the layer."""
        super().__init__()
        self.p = parameters
        self.print_name = print_name
        self._name = self.print_name + "_Pairwise"

    def call(self, inputs):
        """Call the layer"""
        out = inputs[
            :,
            self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
        ]
        out = Lambda(
            lambda x: pairwise_dist(x, flat=True),
            name=f"{self.print_name}_Pairwise_Distances",
        )(out)
        return out
