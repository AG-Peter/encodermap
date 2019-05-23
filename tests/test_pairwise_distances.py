import encodermap as em
import numpy as np
from math import pi
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class TestPeriodicDistances(tf.test.TestCase):
    def test_periodic_distance_np(self):
        a = np.array([0, 0, 0])
        b = np.array([pi/2, pi, 3/2*pi])
        distance = em.misc.periodic_distance_np(a, b, periodicity=2*pi)
        self.assertTrue(np.array_equal(distance, np.array([pi/2, pi, pi/2])))

    def test_periodic_distance_tf(self):
        a = np.array([0, 0, 0])
        b = np.array([pi / 2, pi, 3 / 2 * pi])
        distance = em.misc.periodic_distance(a, b, periodicity=2 * pi)
        with self.test_session():
            self.assertAllEqual(distance.eval(), [pi / 2, pi, pi / 2])


class TestPairwiseDistances(tf.test.TestCase):
    def test_pairwise_dist_periodic(self):
        points = [[1/8, 1/2], [7/8, 1/2]]
        pairwise_dists = em.misc.pairwise_dist_periodic(points, 1)
        with self.test_session():
            self.assertAllClose([[0, 1/4], [1/4, 0]], pairwise_dists.eval())

    def test_pairwise_dist_periodic_not_periodic_case(self):
        points = [[1/8, 1/2], [7/8, 1/2]]
        pairwise_dists = em.misc.pairwise_dist_periodic(points, float("inf"))
        with self.test_session():
            self.assertAllClose([[0, 6/8], [6/8, 0]], pairwise_dists.eval())

    def test_pairwise_dist(self):
        points = [[1/8, 1/2], [7/8, 1/2]]
        pairwise_dists = em.misc.pairwise_dist(points)
        with self.test_session():
            self.assertAllClose([[[0, 6/8], [6/8, 0]]], pairwise_dists.eval())

    def test_compare_pairwise_dist_and_periodic_pairwise_dist(self):
        points = np.random.normal(size=(10, 3))
        points = points.astype(np.float32)
        pairwise_dists = em.misc.pairwise_dist(points)
        periodic_pwd = em.misc.pairwise_dist_periodic(points, 10000)
        with self.test_session():
            self.assertAllClose(periodic_pwd.eval().reshape(-1), pairwise_dists.eval().reshape(-1))

    def test_pairwise_dist_flat(self):
        points = np.array([[0, 0],
                           [1, 0],
                           [0, 1]])
        points = points.astype(np.float32)
        pairwise_dists = em.misc.pairwise_dist(points, flat=True)
        with self.test_session():
            self.assertAllClose(pairwise_dists.eval(), [[1, 1, 2**(1/2)]])
