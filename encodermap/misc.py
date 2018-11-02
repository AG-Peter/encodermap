"""
EncoderMap
Copyright (C) 2018  Tobias Lemke

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import tensorflow as tf
from math import pi
import os
import numpy as np


def add_layer_summaries(layer):
    weights = layer.variables[0]
    biases = layer.variables[1]
    variable_summaries(layer.name + "_weights", weights)
    variable_summaries(layer.name + "_biases", biases)


def periodic_distance(a, b, periodicity=2*pi):
    d = tf.abs(b-a)
    return tf.minimum(d, periodicity-d)


def periodic_distance_np(a, b):
    d = np.abs(b-a)
    return np.minimum(d, 2*pi-d)


def variable_summaries(name, variables):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if not isinstance(variables, list):
        variables = [variables]

    for i, var in enumerate(variables):
        try:
            add_index = len(variables) > 1
        except TypeError:
            add_index = True
        if add_index:
            name = name + str(i)
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def sketchmap_cost(r_h, r_l, sig_h, a_h, b_h, sig_l, a_l, b_l):
    dist_h = pairwise_dist(r_h)
    dist_l = pairwise_dist(r_l)

    sig_h = sketchmap_sigmoid(dist_h, sig_h, a_h, b_h)
    sig_l = sketchmap_sigmoid(dist_l, sig_l, a_l, b_l)

    cost = tf.reduce_mean(tf.square(sig_h - sig_l))
    return cost


def sketchmap_sigmoid(r, sig, a, b):
    return 1 - (1 + (2**(a/b) - 1) * (r/sig)**a)**(-b/a)


def pairwise_dist(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    from https://omoindrot.github.io/triplet-loss

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def search_and_replace(file_path, search_pattern, replacement, out_path=None, backup=True):
    """
    Searches for a pattern in a text file and replaces it with the replacement
    Args:
        file_path: (str)
            path to the file to search
        search_pattern: (str)
            pattern to search for
        replacement: (str)
            string that replaces the search_pattern in the output file
        out_path: (str)
            path where to write the output file. If no path is given the original file will be replaced.
        backup: (bool)
            if backup is true the original file is renamed to filename.bak before it is overwritten

    Returns:

    """
    with open(file_path, "r") as f:
        file_data = f.read()

    file_data = file_data.replace(search_pattern, replacement)

    if not out_path:
        out_path = file_path
        if backup:
            os.rename(file_path, file_path+".bak")

    with open(out_path, "w") as file:
        file.write(file_data)


def run_path(path):
    """
    Creates a directory at "path/run{i}" where i the i corresponding to the smallest not yet existing path
    Args:
        path: (str)

    Returns: (str)
        path of the created folder

    """
    i = 0
    while True:
        current_path = os.path.join(path, "run{}".format(i))
        if not os.path.exists(current_path):
            os.makedirs(current_path)
            output_path = current_path
            break
        else:
            i += 1
    return output_path
