# -*- coding: utf-8 -*-
# encodermap/loss_functions/loss_classes.py
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
"""Losses for encodermap.

All losses in EncoderMap inherit from `tf.keras.losses.Loss` and thus can be
easily paired with other models.

"""
################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import functools
import inspect
import os
from collections.abc import Sequence
from typing import Any, Optional, Union

# Third Party Imports
import tensorflow as tf
import tensorflow.keras.backend as K

# Encodermap imports
from encodermap.misc.distances import (
    pairwise_dist,
    pairwise_dist_periodic,
    periodic_distance,
    sigmoid,
)
from encodermap.parameters.parameters import ADCParameters, Parameters


################################################################################
# Typing
################################################################################


################################################################################
# Globals
################################################################################


__all__: list[str] = ["DihedralLoss", "AngleLoss"]


################################################################################
# Utils
################################################################################


def testing(cls_or_func):
    if inspect.isclass(cls_or_func):
        orig_init = cls_or_func.__init__

        @functools.wraps(cls_or_func)
        def __init__(self, *args, **kwargs):
            if os.getenv("ENCODERMAP_TESTING", "False") != "True":
                raise Exception(
                    f"You are instantiating a em.testing class ({cls_or_func.__name__}). "
                    f"These classes are actively developed and not stable. If you "
                    f"know what you are doing, set the environment variable "
                    f"'EM_TESTING' to 'True'."
                )
            return orig_init(self, *args, **kwargs)

        cls_or_func.__init__ = __init__
        return cls_or_func
    else:

        @functools.wraps(cls_or_func)
        def newfunc(*args, **kwargs):
            if os.getenv("ENCODERMAP_TESTING", "False") != "True":
                raise Exception(
                    f"You are calling an em.testing function: ({cls_or_func.__name__}). "
                    f"These functions are actively developed and not stable. If you "
                    f"know what you are doing, set the environment variable "
                    f"'EM_TESTING' to 'True'."
                )
            return cls_or_func(*args, **kwargs)

        return newfunc


def _do_nothing(*args) -> None:
    """This function does nothing. One of the functions provided to tf.cond."""
    pass


def _summary_cost(
    name: str,
    cost: tf.Tensor,
) -> None:
    """This functions logs a scalar to a name. One of the functions provided to tf.cond.

    Args:
        name (str): The name to log the scalar as.
        cost (tf.Tensor): The scalar tensor to log.

    """
    tf.summary.scalar(name, cost)


################################################################################
# PublicClasses
################################################################################


@testing
@tf.keras.utils.register_keras_serializable()
class EncoderMapBaseLoss(tf.keras.losses.Loss):
    """EncoderMap's base loss. Serializes parameters and `self._train_counter`.

    It Can be subclassed to implement custom loss functions that have access to
    EncoderMap's parameter classes.

    """

    def __init__(
        self,
        parameters: Optional[Union[Parameters, ADCParameters]] = None,
        write_bool_cb: Optional[tf.keras.callbacks.Callback] = None,
        **kwargs,
    ) -> None:
        """Instantiate the Loss class.

        Most subclassed losses, don't need to overwrite this `__init__()`.

        Args:
            parameters (Optional[Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]]): The parameters
                this class will use to decide hwo to compute losses.

        """
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters

        if write_bool_cb is None:
            self.cb = None
            self.write_bool = K.constant(False, "bool", name="log_bool")
        else:
            self.cb = write_bool_cb
            self.write_bool = write_bool_cb.log_bool

        super().__init__()

    def call(self, loss_name, current_loss) -> None:
        """Use super().call(loss_name, current_loss) to log the current loss to
        tensorboard and advance the train counter.

        Args:
            loss_name (str): The name of the loss, as it should appear in Tensorboard.
            current_loss (float): The current value of the loss.

        Returns:
            float: The current loss.

        """
        tf.cond(
            self.write_bool,
            true_fn=lambda: _summary_cost(loss_name, current_loss),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return current_loss

    @classmethod
    def from_config(cls, config):
        p = config.pop("p")
        if "cartesian_pwd_start" in p:
            p = ADCParameters(**p)
        else:
            p = Parameters(**p)
        write_bool_cb = tf.keras.saving.deserialize_keras_object(
            config.pop("write_bool_cb")
        )
        return cls(parameters=p, write_bool_cb=write_bool_cb, **config)

    def get_config(self) -> dict[Any, Any]:
        config = super().get_config().copy()
        config.update(
            {
                "p": self.p.to_dict(),
                "write_bool_cb": tf.keras.saving.serialize_keras_object(self.cb),
            }
        )
        return config


@testing
@tf.keras.utils.register_keras_serializable()
class ADCBaseLoss(EncoderMapBaseLoss):
    """Base class for all Losses of the `AngleDihedralCartesianEncoderMap`.

    Replaces the default `Parameters()` with `ADCParameters()`.

    """

    def __init__(
        self,
        parameters: Optional[ADCParameters] = None,
        write_bool_cb: Optional[tf.keras.callbacks.Callback] = None,
        **kwargs,
    ) -> None:
        super().__init__(parameters, write_bool_cb)
        if parameters is None:
            self.p = ADCParameters()


@testing
@tf.keras.utils.register_keras_serializable()
class DihedralLoss(ADCBaseLoss):
    """EncoderMap's `DihedralLoss` for `AngleDihedralCartesianEncoderMap`.

    Uses the periodicity in `self.p` to compare the distances of input and
    output dihedrals. The `inp_dihedrals` are a tensor of size
    (batch_size, n_dihedrals), the `out_dihedrals` are a tensor of size
    (batch_size, n_dihedrals). The distances between two dihedrals are
    calculated with `d = tf.abs(inp_dihedrals - out_dihedrals)`. This array
    has the shape (batch_size, n_dihedrals). Because angles lie in a periodic
    space with periodicity (-pi, pi] this array needs to be adjusted to account
    for this with `tf.min(d, periodicity - d)`. The resulting array of shape
    (batch_size, n_dihedrals) will now be transformed based on the
    `dihedral_cost_variant`, which can be 'mean_square', 'mean_abs', or 'mean_norm'.

    """

    name = "DihedralLoss"

    def call(self, y_true: Sequence[tf.Tensor], y_pred: Sequence[tf.Tensor]) -> float:
        inp_dihedrals = y_pred[1]
        out_dihedrals = y_true[1]

        if self.p.dihedral_cost_scale is not None:
            if self.p.dihedral_cost_variant == "mean_square":
                dihedral_cost = tf.reduce_mean(
                    tf.square(
                        periodic_distance(
                            inp_dihedrals, out_dihedrals, self.p.periodicity
                        )
                    )
                )
            elif self.p.dihedral_cost_variant == "mean_abs":
                dihedral_cost = tf.reduce_mean(
                    tf.abs(
                        periodic_distance(
                            inp_dihedrals, out_dihedrals, self.p.periodicity
                        )
                    )
                )
            elif self.p.dihedral_cost_variant == "mean_norm":
                dihedral_cost = tf.reduce_mean(
                    tf.norm(
                        periodic_distance(
                            inp_dihedrals, out_dihedrals, self.p.periodicity
                        ),
                        axis=1,
                    )
                )
            else:
                raise ValueError(
                    f"dihedral_cost_variant {self.p.dihedral_cost_variant} not available"
                )
            dihedral_cost /= self.p.dihedral_cost_reference
            if self.p.dihedral_cost_scale != 0:
                dihedral_cost *= self.p.dihedral_cost_scale
        else:
            dihedral_cost = 0

        return super().call("Dihedral Cost", dihedral_cost)


@testing
@tf.keras.utils.register_keras_serializable()
class AngleLoss(ADCBaseLoss):
    """EncoderMap's `AngleLoss` for `AngleDihedralCartesianEncoderMap`.

    Uses the periodicity in `self.p` to compare the distances of input and
    output angles. The `inp_angles` are a tensor of size
    (batch_size, n_angles), the `out_angles` are a tensor of size
    (batch_size, n_angles). The distances between two angles are
    calculated with `d = tf.abs(inp_angles - out_angles)`. This array
    has the shape (batch_size, n_angles). Because angles lie in a periodic
    space with periodicity (-pi, pi] this array needs to be adjusted to account
    for this with `tf.min(d, periodicity - d)`. The resulting array of shape
    (batch_size, n_angles) will now be transformed based on the
    `angle_cost_variant`, which can be 'mean_square', 'mean_abs', or 'mean_norm'.

    """

    def call(self, y_true: Sequence[tf.Tensor], y_pred: Sequence[tf.Tensor]) -> float:
        inp_angles = y_pred[0]
        out_angles = y_true[0]

        if self.p.angle_cost_scale is not None:
            if self.p.angle_cost_variant == "mean_square":
                angle_cost = tf.reduce_mean(
                    tf.square(
                        periodic_distance(inp_angles, out_angles, self.p.periodicity)
                    )
                )
            elif self.p.angle_cost_variant == "mean_abs":
                angle_cost = tf.reduce_mean(
                    tf.abs(
                        periodic_distance(inp_angles, out_angles, self.p.periodicity)
                    )
                )
            elif self.p.angle_cost_variant == "mean_norm":
                angle_cost = tf.reduce_mean(
                    tf.norm(
                        periodic_distance(inp_angles, out_angles, self.p.periodicity),
                        axis=1,
                    )
                )
            else:
                raise ValueError(
                    f"angle_cost_variant {self.p.angle_cost_variant} not available"
                )
            angle_cost /= self.p.angle_cost_reference
            if self.p.angle_cost_scale != 0:
                angle_cost *= self.p.angle_cost_scale
        else:
            angle_cost = 0

        return super().call("Angle Cost", angle_cost)
