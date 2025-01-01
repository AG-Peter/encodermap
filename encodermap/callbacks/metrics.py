# -*- coding: utf-8 -*-
# encodermap/callbacks/metrics.py
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
"""Metrics are meta-variables that can be computed to observe the training but
are not directly linked to loss/cost and gradients.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

# Third Party Imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# Encodermap imports
from encodermap.misc.distances import pairwise_dist


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Local Folder Imports
    from ..parameters import ADCParameters, Parameters


################################################################################
# Metrics
################################################################################


WEIGHTS: list[float] = [14.0067, 24.305, 24.305]


################################################################################
# Metrics
################################################################################


def kabsch_weighted(
    P: np.ndarray, Q: np.ndarray, W: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Taken from Jimmy C. Kromann's RMSD (https://github.com/charnley/rmsd)
    Using the Kabsch algorithm with two sets of paired point P and Q.
    Each vector set is represented as an NxD matrix, where D is the
    dimension of the space.
    An optional vector of weights W may be provided.

    Note that this algorithm does not require that P and Q have already
    been overlayed by a centroid translation.

    The function returns the rotation matrix U, translation vector V,
    and RMS deviation between Q and P', where P' is:

        P' = P * U + V

    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Args:
        P (np.ndarray): Points A with shape (n_points, 3).
        Q (np.ndarray): Points B with shape (n_points, 3).
        W (np.ndarray): Weights with shape (n_points, ).

    Returns:
        float: The RMSD value in the same units, as the input points.

    """
    # Computation of the weighted covariance matrix
    CMP = np.zeros(3).astype("float32")
    CMQ = np.zeros(3).astype("float32")
    C = np.zeros((3, 3)).astype("float32")
    if W is None:
        W = np.ones(len(P)).astype("float32") / len(P)
    W = np.array([W, W, W]).T.astype("float32")
    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw = 3.0 / W.sum()
    n = len(P)
    for i in range(3):
        for j in range(n):
            for k in range(3):
                C[i, k] += P[j, i] * Q[j, k] * W[j, i]
    CMP = (P * W).sum(axis=0)
    CMQ = (Q * W).sum(axis=0)
    PSQ = (P * P * W).sum() - (CMP * CMP).sum() * iw
    QSQ = (Q * Q * W).sum() - (CMQ * CMQ).sum() * iw
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U = np.dot(V, W)
    msd = (PSQ + QSQ) * iw - 2.0 * S.sum()
    if msd < 0.0:
        msd = 0.0
    rmsd_ = np.sqrt(msd)
    V = np.zeros(3).astype("float32")
    for i in range(3):
        t = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V = V * iw
    return rmsd_.astype("float32")


def rmsd(a, b, translate):
    weights = np.tile(WEIGHTS, a.shape[0] // 3).astype("float32")
    if translate:
        a -= np.tile(np.expand_dims(np.mean(a, axis=-1), -1), (1, 3))
        b -= np.tile(np.expand_dims(np.mean(b, axis=-1), -1), (1, 3))
    return kabsch_weighted(a, b, weights)


def rmsd_numpy(a: np.ndarray, b: np.ndarray, translate: bool = True) -> np.ndarray:
    """Implements Kabsch-Umeyama algorithm.

    References:
        @article{kabsch1976solution,
            title={A solution for the best rotation to relate two sets of vectors},
            author={Kabsch, Wolfgang},
            journal={Acta Crystallographica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography},
            volume={32},
            number={5},
            pages={922--923},
            year={1976},
            publisher={International Union of Crystallography}
        }

    """
    result = [rmsd(i, j, translate=translate) for i, j in zip(a, b)]
    return np.asarray(result)


def kabsch_tf(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    # weights repeat N-CA-C
    weights = tf.tile(WEIGHTS, [tf.shape(a)[0] // 3])

    # center coordinates
    a -= tf.tile(tf.expand_dims(tf.reduce_mean(a, axis=-1), -1), (1, 3))
    b -= tf.tile(tf.expand_dims(tf.reduce_mean(b, axis=-1), -1), (1, 3))

    # predefine multipliers
    S_mul = tf.convert_to_tensor(
        [1, 1, -1],
        dtype=tf.float32,
    )

    # Computation of the weighted covariance matrix
    C = tf.zeros((3, 3), dtype=tf.float32)
    W = tf.tile(tf.expand_dims(weights, -1), [1, 3])
    iw = 3.0 / tf.reduce_sum(W)
    n = tf.shape(a)[0]
    for i in range(3):
        for j in range(n):
            for k in range(3):
                updates = (
                    C[i, k]
                    + tf.gather_nd(a, [j, i]) * tf.gather_nd(b, [j, k]) * W[j, i]
                )
                C = tf.tensor_scatter_nd_update(
                    tensor=C,
                    indices=[[i, k]],
                    updates=[updates],
                )
    CMP = tf.reduce_sum(a * W, axis=0)
    CMQ = tf.reduce_sum(b * W, axis=0)
    PSQ = tf.reduce_sum(tf.square(a) * W) - tf.reduce_sum(tf.square(CMP)) * iw
    QSQ = tf.reduce_sum(tf.square(b) * W) - tf.reduce_sum(tf.square(CMQ)) * iw
    C = (C - tf.tensordot(CMP, CMQ, axes=0) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    S, V, W = tf.linalg.svd(C, full_matrices=False)
    d = (tf.linalg.det(V) * tf.linalg.det(W)) < 0.0
    if d:
        S *= S_mul
    msd = (PSQ + QSQ) * iw - 2.0 * tf.reduce_sum(S)
    if msd < 0.0:
        msd = 0.0
    return tf.sqrt(msd)


def rmsd_tf(
    a: tf.Tensor,
    b: tf.Tensor,
    p: ADCParameters,
) -> tf.Tensor:
    out = []
    for i in range(p.batch_size):
        result = kabsch_tf(
            tf.gather_nd(a, [i]),
            tf.gather_nd(b, [i]),
        )
        out.append(result)
    return tf.stack(out)


################################################################################
# Metric Classes
################################################################################


@tf.keras.utils.register_keras_serializable()
class EncoderMapBaseMetric(tf.keras.metrics.Metric):
    """Base class for metrics in EncoderMap.

    Metrics are used to judge the performance of ML models. They are similar
    to loss functions as they can (but don't have to) be computed at every
    iteration of the training. Oftentimes, metrics implement more expensive
    calculations. Metrics are also automatically added to a model's training
    history, accessible via `history = emap.train()`.

     Examples:

         In this example, the `update` method always returns zero.

        >>> import encodermap as em
        >>> import numpy as np
        ...
        >>> class MyMetric(em.callbacks.EncoderMapBaseMetric):
        ...    def update(self, y_true, y_pred, sample_weight=None):
        ...        return 0.0
        ...
        >>> emap = em.EncoderMap()  # doctest: +ELLIPSIS
        Output...
        >>> emap.add_metric(MyMetric)
        >>> history = emap.train()  # doctest: +ELLIPSIS
        Saving...
        >>> np.mean(history.history["MyMetric Metric"])
        0.0
        >>> len(history.history["MyMetric Metric"]) == emap.p.n_steps
        True

    """

    custom_update_state = True

    def __init__(
        self,
        parameters: Optional[ADCParameters],
        name: str | None = None,
        current_training_step: Optional[int] = None,
        **kwargs,
    ) -> None:
        if name is None:
            name = f"{self.__class__.__name__} Metric"
        super(EncoderMapBaseMetric, self).__init__(name=name, **kwargs)
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters
        if current_training_step is None:
            self._my_training_step = K.variable(
                self.p.current_training_step, "int64", name="train_counter"
            )
        else:
            if parameters is not None:
                if current_training_step != parameters.current_training_step:
                    raise Exception(
                        f"Instantiation of {self.__class__.__name__} got different "
                        f"values for current training steps. In parameters, the "
                        f"training step is {parameters.current_training_step}, in "
                        f"the arguments, I got {current_training_step}"
                    )
            self._my_training_step = K.variable(
                current_training_step, "int64", name="train_counter"
            )
        self.custom_metric_scalar = self.add_weight(
            name=f"custom_metric_vector_{self.__class__.__name__}",
            initializer="zeros",
            dtype=tf.float32,
        )
        if not self.custom_update_state and not hasattr(self, "update"):
            raise Exception(
                f"Please implement an `update` method, that returns a scalar, when"
                f"sublcassing this metric."
            )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        custom_objects: Optional[dict[Any, Any]] = None,
    ):
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[str, Any]): A dictionary.
            custom_objects (Optional[dict[str, Any]]): Not needed here, but see
                https://keras.io/guides/serialization_and_saving/ for yourself.

        """
        parameters = config.pop("parameters")
        parameters = Parameters(**parameters)
        return cls(
            name=config.pop("name"),
            parameters=parameters,
            current_training_step=None,
            **config,
        )

    def get_config(self) -> dict[str, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[str, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config["parameters"] = self.p.to_dict()
        return config

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        """Updates this metric."""
        self.custom_metric_scalar.assign(self.update(y_true, y_pred))
        self._my_training_step.assign_add(1)

    def result(self):
        return self.custom_metric_scalar


@tf.keras.utils.register_keras_serializable()
class AngleDihedralCartesianEncoderMapBaseMetric(tf.keras.metrics.Metric):
    custom_update_state = False

    def __init__(
        self,
        parameters: Optional[ADCParameters],
        name: str | None = None,
        current_training_step: Optional[int] = None,
        **kwargs,
    ) -> None:
        if name is None:
            name = f"{self.__class__.__name__} Metric"
        super(AngleDihedralCartesianEncoderMapBaseMetric, self).__init__(
            name=name, **kwargs
        )
        if parameters is None:
            self.p = ADCParameters()
        else:
            self.p = parameters
        if current_training_step is None:
            self._my_training_step = K.variable(
                self.p.current_training_step, "int64", name="train_counter"
            )
        else:
            if parameters is not None:
                if current_training_step != parameters.current_training_step:
                    raise Exception(
                        f"Instantiation of {self.__class__.__name__} got different "
                        f"values for current training steps. In parameters, the "
                        f"training step is {parameters.current_training_step}, in "
                        f"the arguments, I got {current_training_step}"
                    )
            self._my_training_step = K.variable(
                current_training_step, "int64", name="train_counter"
            )
        if not self.custom_update_state and not hasattr(self, "update"):
            raise Exception(
                f"Please implement an `update` method, that returns a scalar, when"
                f"sublcassing this metric."
            )
        # self.custom_metric_vector = self.add_weight(
        #     name=f"custom_metric_vector_{self.__class__.__name__}",
        #     shape=self.p.batch_size,
        #     initializer="zeros",
        # )
        # self.custom_metric_scalar = self.add_weight(
        #     name=f"custom_metric_vector_{self.__class__.__name__}",
        #     initializer="zeros",
        #     dtype=tf.float32,
        # )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        custom_objects: Optional[dict[Any, Any]] = None,
    ):
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[str, Any]): A dictionary.
            custom_objects (Optional[dict[str, Any]]): Not needed here, but see
                https://keras.io/guides/serialization_and_saving/ for yourself.

        """
        parameters = config.pop("parameters")
        parameters = ADCParameters(**parameters)
        return cls(
            name=config.pop("name"),
            parameters=parameters,
            current_training_step=None,
            **config,
        )

    def get_config(self) -> dict[str, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[str, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config["parameters"] = self.p.to_dict()
        return config


class OmegaAngleBaseMetric(AngleDihedralCartesianEncoderMapBaseMetric):
    custom_update_state = True


class SidechainVsBackboneFrequencyBaseMetric(
    AngleDihedralCartesianEncoderMapBaseMetric
):
    custom_update_state = True


class ADCClashMetric(AngleDihedralCartesianEncoderMapBaseMetric):
    custom_update_state = True
    """Metric that computes clashes between atoms in the reconstructed backbone.

    Please choose the correct distance unit.

    """

    def __init__(
        self,
        distance_unit: Literal["nm", "ang"],
        name: str = "ADCClashMetric",
        parameters: Optional[ADCParameters] = None,
        **kwargs,
    ):
        super().__init__(name=name, parameters=parameters, **kwargs)
        self.clashes = self.add_weight(
            name="clashes",
            shape=(),
            initializer="zeros",
            dtype=tf.int64,
        )
        if distance_unit == "nm":
            self.clash_distance = 0.1
        elif distance_unit == "ang":
            self.clash_distance = 1
        else:
            raise Exception(
                f"Argument `distance_unit` must be either 'nm' or 'ang'. You "
                f"provided {distance_unit=}."
            )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        """Updates this metric.

        y_true (tf.Tensor): The train goal.
        y_pred (tf.Tensor): Current model output.

        """
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[2]
        clashes = tf.math.count_nonzero(
            pairwise_dist(y_pred, flat=True) < self.clash_distance,
            axis=1,
            dtype=tf.int64,
        )
        with tf.name_scope("Metrics"):
            tf.summary.scalar(
                "Mean Clashes", tf.reduce_mean(clashes), step=self._my_training_step
            )
            tf.summary.histogram("Batch Clashes", clashes, step=self._my_training_step)
        self.clashes.assign(tf.reduce_mean(clashes))
        self._my_training_step.assign_add(1)

    def result(self):
        return self.clashes


class ADCRMSDMetric(AngleDihedralCartesianEncoderMapBaseMetric):
    custom_update_state = True

    def __init__(
        self,
        name: str = "ADCRMSDMetric",
        parameters: Optional[Union[Parameters, ADCParameters]] = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate the RMSD metric. The RMSD of the output will be computed
        on the CA atoms of input vs output.

        Note:
            Output is in nm.

        Args:
            name (str): Name of the metric. Defaults to 'ADCRMSDMetric'.
            parameters (Optional[Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]]): An instance
                of a parameter class, which is used to define which atoms are CA
                atoms. Defaults to None.

        """
        super(ADCRMSDMetric, self).__init__(
            name=name,
            parameters=parameters,
            **kwargs,
        )
        if parameters is None:
            parameters = ADCParameters()
        self.p = parameters
        self.rmsd = self.add_weight(
            name="rmsd",
            shape=self.p.batch_size,
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[2]
        if isinstance(y_true, (list, tuple)):
            y_true = y_true[2]
        rmsd = rmsd_tf(y_true, y_pred, p=self.p)
        self.rmsd.assign(rmsd)
        with tf.name_scope("Metrics"):
            tf.summary.scalar("Mean RMSDs", tf.reduce_mean(self.rmsd))
            tf.summary.histogram("Batch RMSDs", self.rmsd)

    def result(self):
        return self.rmsd
