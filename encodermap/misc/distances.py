# -*- coding: utf-8 -*-
# encodermap/misc/distances.py
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
################################################################################
# Imports
################################################################################

from math import pi

import numpy as np
import tensorflow as tf

################################################################################
# Globals
################################################################################

__all__ = [
    "sigmoid",
    "periodic_distance",
    "periodic_distance_np",
    "pairwise_dist",
    "pairwise_dist_periodic",
]

################################################################################
# Functions
################################################################################


def sigmoid(sig, a, b):
    """Returns a sigmoid function with specified parameters.

    Args:
        sig (float): Sigma.
        a (float): a.
        b (float): b.

    Returns:
        function: A function that can be used to calculate the sigmoid with the
            specified parameters.
    """

    def func(r):
        return 1 - (1 + (2 ** (a / b) - 1) * (r / sig) ** a) ** (-b / a)

    return func


def periodic_distance_np(a, b, periodicity=2 * pi):
    """Calculates distance between two points and respects periodicity.

    If the provided dataset is periodic (i.e. angles and torsion angles), the returned
    distance is corrected.

    Args:
        a (np.ndarray): Coordinate of point a.
        b (np.ndarray): Coordinate of point b.
        periodicity (float, optional): The periodicity (i.e. the box length/ maximum angle)
            of your data. Defaults to 2*pi. Provide float('inf') for no periodicity.

    """
    d = np.abs(b - a)
    return np.minimum(d, periodicity - d)


def periodic_distance(a, b, periodicity=2 * pi):
    """Calculates distance between two points and respects periodicity.

    If the provided dataset is periodic (i.e. angles and torsion angles), the returned
    distance is corrected.

    Args:
        a (tf.Tensor): Coordinate of point a.
        b (tf.Tensor): Coordinate of point b.
        periodicity (float, optional): The periodicity (i.e. the box length/ maximum angle)
            of your data. Defaults to 2*pi. Provide float('inf') for no periodicity.

    Example:
        >>> import encodermap as em
        >>> x = tf.convert_to_tensor(np.array([[1.5], [1.5]]))
        >>> y = tf.convert_to_tensor(np.array([[-3.1], [-3.1]]))
        >>> r = em.misc.periodic_distance(x, y)
        >>> print(r.numpy())
        [[1.68318531]
         [1.68318531]]

    """
    d = tf.abs(b - a)
    return tf.minimum(d, periodicity - d)


def pairwise_dist_periodic(positions, periodicity):
    """Pairwise distances using periodicity.

    Args:
        positions (Union[np.ndarray, tf.Tensor]): The positions of the points.
            Currently only 2D arrays with positions.shape[0] == n_points
            and positions.shape[1] == 1 (rotational values) is supported.
        periodicity (float): The periodicity of the data. Most often
            you will use either 2*pi or 360.

    """
    assert len(positions.shape) == 2
    if not tf.debugging.is_numeric_tensor(positions):
        positions = tf.convert_to_tensor(positions)
    vecs = periodic_distance(
        tf.expand_dims(positions, axis=1),
        tf.expand_dims(positions, axis=0),
        periodicity,
    )
    mask = tf.cast(tf.equal(vecs, 0.0), "float32")
    vecs = vecs + mask * 1e-16  # gradient infinite for 0
    dists = tf.norm(vecs, axis=2)
    return dists


def pairwise_dist(positions, squared=False, flat=False):
    """Tensorflow implementation of `scipy.spatial.distances.cdist`.

    Returns a tensor with shape (positions.shape[1], positions.shape[1]).
    This tensor is the distance matrix of the provided positions. The
    matrix is hollow, i.e. the diagonal elements are zero.

    Args:
        positions (Union[np.ndarray, tf.Tensor]): Collection of
            n-dimensional points. positions.shape[0] are points.
            positions.shape[1] are dimensions.
        squared (bool, optional): Whether to return the pairwise squared
            euclidean distance matrix or normal euclidean distance matrix.
            Defaults to False.
        flat (bool, otpional): Whether to return only the lower triangle of
            the hollow matrix. Setting this to true mimics the behavior
            of `scipy.spatial.distance.pdist`. Defaults to False.

    """
    # thanks to https://omoindrot.github.io/triplet-loss

    if not tf.debugging.is_numeric_tensor(positions):
        positions = tf.convert_to_tensor(positions)
    if len(positions.get_shape()) == 2:
        positions = tf.expand_dims(positions, 0)

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(positions, tf.transpose(positions, [0, 2, 1]))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        tf.expand_dims(square_norm, 1)
        - 2.0 * dot_product
        + tf.expand_dims(square_norm, 2)
    )

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if flat:
        n = int(positions.shape[1])
        mask = np.ones((n, n), dtype=bool)
        mask[np.tril_indices(n)] = False
        distances = tf.boolean_mask(distances, mask, axis=1)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), np.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances