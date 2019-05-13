import encodermap as em
import MDAnalysis as md
import numpy as np
from math import pi
import tensorflow as tf
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class TestNonBackboneAtoms(tf.test.TestCase):

    def test_guess_amide_H(self):
        cartesians_non_tf = np.array([[[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]],
                                      [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]]])
        cartesians = tf.constant(cartesians_non_tf, dtype=tf.float32)
        atom_names = ["N", "CA", "C", "N", "CA", "C"]

        with self.test_session() as sess:
            H_cartesians = sess.run(em.backmapping.guess_amide_H(cartesians, atom_names))
        fig, axe = plt.subplots()
        axe.plot(cartesians_non_tf[0, :, 0], cartesians_non_tf[0, :, 1], linestyle="", marker="o")
        axe.plot(H_cartesians[0, :, 0], H_cartesians[0, :, 1], linestyle="", marker="o")
        axe.axis("equal")
        plt.show()
    
    def test_guess_amide_O(self):
        cartesians_non_tf = np.array([[[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]],
                                      [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]]])
        cartesians = tf.constant(cartesians_non_tf, dtype=tf.float32)
        atom_names = ["CA", "N", "C", "CA", "N", "C"]

        with self.test_session() as sess:
            O_cartesians = sess.run(em.backmapping.guess_amide_O(cartesians, atom_names))
        fig, axe = plt.subplots()
        axe.plot(cartesians_non_tf[0, :, 0], cartesians_non_tf[0, :, 1], linestyle="", marker="o")
        axe.plot(O_cartesians[0, :, 0], O_cartesians[0, :, 1], linestyle="", marker="o")
        axe.axis("equal")
        plt.show()
