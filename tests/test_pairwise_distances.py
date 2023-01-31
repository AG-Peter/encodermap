# -*- coding: utf-8 -*-
# tests/test_pairwise_distances.py
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
import os
import unittest
from math import pi

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

from encodermap.misc import (
    pairwise_dist,
    pairwise_dist_periodic,
    periodic_distance,
    periodic_distance_np,
    sigmoid,
)
from encodermap.parameters.parameters import Parameters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Todo: This needs more tests

# If scipy was compiled against an older version of numpy these warnings are raised
# warnings in a testing environment are somewhat worrying
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def sigmoid_closure(sig, a, b):
    def func(i, j):
        x = 1 - (1 + (2 ** (a / b) - 1) * ((np.linalg.norm(j - i)) / sig) ** a) ** (
            -b / a
        )
        return x

    return func


def periodicity_closure(periodicity):
    def func(i, j):
        dx = np.linalg.norm(j - i)
        if dx > periodicity * 0.5:
            dx = dx - periodicity
        if dx <= -periodicity * 0.5:
            dx = dx + periodicity
        dx = np.abs(dx)
        return dx

    return func


class TestSigmoidDists(tf.test.TestCase):
    def test_sigmoid(self):
        sig_params = Parameters.defaults["dist_sig_parameters"]
        metric = sigmoid_closure(*sig_params[:3])

        seed = 1
        np.random.seed(1)
        periodicity = 360.0
        points = np.random.random((100, 2)).astype(np.float32) * periodicity

        sigmoid_dists_scipy = cdist(points, points, metric)
        distances_tensorflow = pairwise_dist(points).numpy().squeeze()
        sigmoid_dists_tensorflow = sigmoid(*sig_params[:3])(distances_tensorflow)
        self.assertAllClose(sigmoid_dists_scipy, sigmoid_dists_tensorflow, atol=1e-3)


class TestPeriodicDistances(tf.test.TestCase):
    def test_periodic_distance_np(self):
        a = np.array([0, 0, 0])
        b = np.array([pi / 2, pi, 3 / 2 * pi])
        distance = periodic_distance_np(a, b, periodicity=2 * pi)
        self.assertTrue(np.array_equal(distance, np.array([pi / 2, pi, pi / 2])))

    def test_periodic_distance_tf(self):
        a = np.array([0, 0, 0])
        b = np.array([pi / 2, pi, 3 / 2 * pi])
        distance = periodic_distance(a, b, periodicity=2 * pi)
        self.assertAllEqual(distance.numpy(), [pi / 2, pi, pi / 2])

    def test_periodic_many_points(self):
        periodicity = 360.0
        p1 = 20.0
        p2 = 40.0
        p3 = 340.0

        points = np.vstack([p1, p2, p3]).astype(np.float32)
        metric = periodicity_closure(periodicity)
        distances_scipy = cdist(
            points.astype(np.float32), points.astype(np.float32), metric
        )
        distances_encodermap = (
            pairwise_dist_periodic(points, periodicity).numpy().squeeze()
        )
        self.assertAllClose(distances_scipy, distances_encodermap, atol=1e-5)

        seed = 1
        np.random.seed(1)
        points = np.random.random((100, 1)).astype(np.float32) * periodicity
        distances_scipy = cdist(
            points.astype(np.float32), points.astype(np.float32), metric
        )
        distances_encodermap = (
            pairwise_dist_periodic(points, periodicity).numpy().squeeze()
        )
        self.assertAllClose(distances_scipy, distances_encodermap, atol=1e-5)


class TestPairwiseDistances(tf.test.TestCase):
    def test_pairwise_dist_periodic(self):
        points = np.array([[1 / 8, 1 / 2], [7 / 8, 1 / 2]], dtype=np.float32)
        pairwise_dists = pairwise_dist_periodic(points, 1)
        self.assertAllClose([[0, 1 / 4], [1 / 4, 0]], pairwise_dists.numpy())

    def test_pairwise_dist_periodic_not_periodic_case(self):
        points = np.array([[1 / 8, 1 / 2], [7 / 8, 1 / 2]], dtype=np.float32)
        pairwise_dists = pairwise_dist_periodic(points, float("inf"))
        self.assertAllClose([[0, 6 / 8], [6 / 8, 0]], pairwise_dists.numpy())

    def test_pairwise_dist(self):
        points = [[1 / 8, 1 / 2], [7 / 8, 1 / 2]]
        pairwise_dists = pairwise_dist(points)
        self.assertAllClose([[[0, 6 / 8], [6 / 8, 0]]], pairwise_dists.numpy())

    def test_compare_pairwise_dist_and_periodic_pairwise_dist(self):
        points = np.random.normal(size=(10, 3))
        points = points.astype(np.float32)
        pairwise_dists = pairwise_dist(points)
        periodic_pwd = pairwise_dist_periodic(points, 10000)
        self.assertAllClose(
            periodic_pwd.numpy().reshape(-1), pairwise_dists.numpy().reshape(-1)
        )

    def test_pairwise_dist_flat(self):
        points = np.array([[0, 0], [1, 0], [0, 1]])
        points = points.astype(np.float32)
        pairwise_dists = pairwise_dist(points, flat=True)
        self.assertAllClose(pairwise_dists.numpy(), [[1, 1, 2 ** (1 / 2)]])


class TestDistancesEm1Em2(tf.test.TestCase):
    def test_periodic_distance(self):
        from encodermap.encodermap_tf1.misc import (
            periodic_distance as em1_periodic_distance,
        )
        from encodermap.misc.distances import periodic_distance as em2_periodic_distance

        x = tf.convert_to_tensor(np.array([[1.5], [1.5]]))
        y = tf.convert_to_tensor(np.array([[-3.1], [-3.1]]))
        r1 = em1_periodic_distance(x, y)
        r2 = em2_periodic_distance(x, y)
        self.assertAllEqual(r1, r2)

    def test_sigmoid(self):
        from encodermap.encodermap_tf1.misc import sigmoid as em1_sigmoid
        from encodermap.misc.distances import sigmoid as em2_sigmoid

        r1 = em1_sigmoid(1.5, 5, 12, 2)
        r2 = em2_sigmoid(5, 12, 2)(1.5)
        self.assertEqual(r1, r2)

    def test_pairwise_dist_periodic(self):
        from encodermap.encodermap_tf1.misc import (
            pairwise_dist_periodic as em1_pairwise_dist_periodic,
        )
        from encodermap.misc.distances import (
            pairwise_dist_periodic as em2_pairwise_dist_periodic,
        )

        x = tf.convert_to_tensor(
            np.array([[1.5, 1.3, -3.1, -3.0], [1.5, 1.6, -3.1, -3.2]]).astype("float32")
        )
        r1 = em1_pairwise_dist_periodic(x, periodicity=2 * np.pi)
        r2 = em2_pairwise_dist_periodic(x, periodicity=2 * np.pi)
        self.assertAllEqual(r1, r2)

    def test_pairwise_dist(self):
        from encodermap.encodermap_tf1.misc import pairwise_dist as em1_pairwise_dist
        from encodermap.misc.distances import pairwise_dist as em2_pairwise_dist

        seed = 1
        x = tf.convert_to_tensor(np.random.random((100, 2)).astype("float32"))
        r1 = em1_pairwise_dist(x)
        r2 = em2_pairwise_dist(x)
        self.assertAllEqual(r1, r2)


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (
    TestSigmoidDists,
    TestPeriodicDistances,
    TestPairwiseDistances,
    TestDistancesEm1Em2,
)

# doctests
import doctest

import encodermap.misc.distances as distances

doc_tests = (doctest.DocTestSuite(distances),)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    suite.addTests(doc_tests)
    return suite


# unittest.TextTestRunner(verbosity = 2).run(testSuite)

# if __name__ == '__main__':
#     print(unittest.__file__)
#     unittest.main()
