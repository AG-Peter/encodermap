import tensorflow as tf
from math import pi
import os
import numpy as np


def add_layer_summaries(layer):
    """

    :param layer:
    :return:
    """
    weights = layer.variables[0]
    biases = layer.variables[1]
    variable_summaries(layer.name + "_weights", weights)
    variable_summaries(layer.name + "_biases", biases)


def periodic_distance(a, b, periodicity=2*pi):
    """

    :param a:
    :param b:
    :param periodicity:
    :return:
    """
    d = tf.abs(b-a)
    return tf.minimum(d, periodicity-d)


def periodic_distance_np(a, b, periodicity=2*pi):
    """

    :param a:
    :param b:
    :param periodicity:
    :return:
    """
    d = np.abs(b-a)
    return np.minimum(d, periodicity-d)


def variable_summaries(name, variables):
    """
    Attach several summaries to a Tensor for TensorBoard visualization.

    :param name:
    :param variables:
    :return:
    """
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
    """

    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def distance_cost(r_h, r_l, sig_h, a_h, b_h, sig_l, a_l, b_l, periodicity):
    """

    :param r_h:
    :param r_l:
    :param sig_h:
    :param a_h:
    :param b_h:
    :param sig_l:
    :param a_l:
    :param b_l:
    :param periodicity:
    :return:
    """
    with tf.name_scope("distance_cost"):
        if periodicity == float("inf"):
            dist_h = pairwise_dist(r_h)
        else:
            dist_h = pairwise_dist_periodic(r_h, periodicity)
        dist_l = pairwise_dist(r_l)

        sig_h = sigmoid(dist_h, sig_h, a_h, b_h)
        sig_l = sigmoid(dist_l, sig_l, a_l, b_l)

        cost = tf.reduce_mean(tf.square(sig_h - sig_l))
        return cost


def sigmoid(r, sig, a, b):
    """

    :param r:
    :param sig:
    :param a:
    :param b:
    :return:
    """
    return 1 - (1 + (2**(a/b) - 1) * (r/sig)**a)**(-b/a)


def pairwise_dist_periodic(positions, periodicity):
    with tf.name_scope("pairwise_dist_periodic"):
        vecs = periodic_distance(tf.expand_dims(positions, axis=1), tf.expand_dims(positions, axis=0), periodicity)
        mask = tf.to_float(tf.equal(vecs, 0.0))
        vecs = vecs + mask * 1e-16  # gradient infinite for 0
        dists = tf.norm(vecs, axis=2)
        return dists


def pairwise_dist(positions, squared=False):
    """
    Compute the 2D matrix of distances between all the embeddings.

    :param positions: tensor of shape (batch_size, position_dim)
    :param squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
        If false, output is the pairwise euclidean distance matrix.

    :return: the pairwise_distances as a tensor of shape (batch_size, batch_size)
    """
    # thanks to https://omoindrot.github.io/triplet-loss

    with tf.name_scope("pairwise_dist"):
        # Get the dot product between all embeddings
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(positions, tf.transpose(positions))

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

    :param file_path: (str)
            path to the file to search
    :param search_pattern: (str)
            pattern to search for
    :param replacement: (str)
            string that replaces the search_pattern in the output file
    :param out_path: (str)
            path where to write the output file. If no path is given the original file will be replaced.
    :param backup: (bool)
            if backup is true the original file is renamed to filename.bak before it is overwritten

    :return:

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
    Creates a directory at "path/run{i}" where the i is corresponding to the smallest not yet existing path

    :param path: (str)
    :return: (str)
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


def random_on_cube_edges(n_points, sigma=0):
    x = y = z = 1
    r = np.random.uniform(size=n_points)

    coordinates = np.zeros((n_points, 3))

    ids = np.zeros(n_points)

    a = np.array([[0, 0, 0]]*3 +
                 [[x, y, 0]]*3 +
                 [[0, y, z]]*3 +
                 [[x, 0, z]]*3)

    b = np.array([[x, 0, 0],
                  [0, y, 0],
                  [0, 0, z],
                  [-x, 0, 0],
                  [0, -y, 0],
                  [0, 0, z],
                  [x, 0, 0],
                  [0, -y, 0],
                  [0, 0, -z],
                  [-x, 0, 0],
                  [0, y, 0],
                  [0, 0, -z],
                  ])

    for i in range(12):
        mask = (i/12 < r) & (r < (i+1)/12)
        coordinates[mask] += a[i] + (np.expand_dims(r[mask], axis=1) - i/12) * 12 * b[i]
        ids[mask] = i

    if sigma:
        coordinates += np.random.normal(scale=sigma, size=(n_points, 3))

    return coordinates, ids
