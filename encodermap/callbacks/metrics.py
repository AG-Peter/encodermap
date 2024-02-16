# -*- coding: utf-8 -*-
# encodermap/callbacks/metrics.py
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
"""Metrics are meta-variables that can be computed to observe the training but
are not directly linked to loss/cost and gradients.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
from typing import TYPE_CHECKING, Optional, Union

# Third Party Imports
import tensorflow as tf
import tensorflow_probability as tfp

# Local Folder Imports
from ..misc.distances import pairwise_dist


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Local Folder Imports
    from ..parameters import ADCParameters, Parameters


################################################################################
# Metrics
################################################################################


@tf.function
def tf_rmsd(y_true, y_pred, R, x, batch_size, n_atoms=None):
    if n_atoms is None:
        n_atoms = tf.shape(y_true)[1]
    # translate and rotate y_true using R and x
    trans = tf.broadcast_to(
        tf.expand_dims(x, axis=1),
        (batch_size, n_atoms, 3),
    ) + tf.keras.backend.batch_dot(
        y_true,
        R,
    )

    # difference between trans and y_pred
    diff = tf.math.sqrt(
        tf.reduce_sum(
            tf.square(
                tf.norm(
                    trans - y_pred,
                    axis=2,
                ),
            ),
            axis=1,
        )
        / tf.cast(n_atoms, "float32"),
    )

    return diff


@tf.function
def optimize_rmsd(y_true, y_pred, R, x):
    batch_size = tf.shape(y_true)[0]
    n_atoms = tf.shape(y_true)[1]

    loss_fn = lambda: tf_rmsd(y_true, y_pred, R, x, batch_size, n_atoms)

    tfp.math.minimize(
        loss_fn,
        10,
        tf.optimizers.Adam(learning_rate=0.1),
        jit_compile=True,
    )


class ADCClashMetric(tf.keras.metrics.Metric):
    def __init__(
        self,
        name: str = "ADCClashMetric",
        parameters: Optional[Union["Parameters", "ADCParameters"]] = None,
        **kwargs,
    ):
        super(ADCClashMetric, self).__init__(name=name, **kwargs)
        if parameters is None:
            parameters = Parameters()
        self.p = parameters
        self.clashes = self.add_weight(
            name="clashes", shape=self.p.batch_size, initializer="zeros"
        )
        # self.clashes = tf.Variable(
        #     initial_value=tf.ones(shape=(self.p.batch_size, ), dtype="float32"),
        #     name="clashes",
        #     shape=(self.p.batch_size,),
        #     dtype="float32",
        # )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> None:
        """Updates this metric.

        y_true (tf.Tensor): The train goal.
        y_pred (tf.Tensor): Current model output.

        """
        clashes = tf.math.count_nonzero(
            pairwise_dist(y_pred, flat=True) < 1, axis=1, dtype=tf.float32
        )
        self.clashes.assign(clashes)
        with tf.name_scope("Metrics"):
            tf.summary.scalar("Mean Clashes", tf.reduce_mean(self.clashes))
            tf.summary.histogram("Batch Clashes", self.clashes)

    def result(self):
        return self.clashes


class ADCRMSDMetric(tf.keras.metrics.Metric):
    def __init__(
        self,
        name: str = "ADCRMSDMetric",
        parameters: Optional[Union["Parameters", "ADCParameters"]] = None,
        **kwargs,
    ):
        super(ADCRMSDMetric, self).__init__(name=name, **kwargs)
        if parameters is None:
            parameters = Parameters()
        self.p = parameters
        self.rmsd = self.add_weight(
            name="rmsd", shape=self.p.batch_size, initializer="zeros"
        )
        self.R = tf.random.uniform((self.p.batch_size, 3, 3))
        self.x = tf.random.uniform((self.p.batch_size, 3))
        # self.rmsd = tf.Variable(
        #     initial_value=tf.ones(shape=(self.p.batch_size, ), dtype="float32"),
        #     name="rmsd",
        #     shape=(self.p.batch_size,),
        #     dtype="float32",
        # )

    def update_state(self, y_true, y_pred, sample_weight=None):
        optimize_rmsd(y_true, y_pred, self.R, self.x)
        rmsd = tf_rmsd(y_true, y_pred, self.R, self.x, self.p.batch_size)
        self.rmsd.assign(rmsd)
        with tf.name_scope("Metrics"):
            tf.summary.scalar("Mean RMSDs", tf.reduce_mean(self.rmsd))
            tf.summary.histogram("Batch RMSDs", self.rmsd)

    def result(self):
        return self.rmsd
