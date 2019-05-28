from .autoencoder import Autoencoder
import tensorflow as tf
import numpy as np
from .misc import periodic_distance, variable_summaries, potential_energy, distance_cost, pairwise_dist
from .backmapping import dihedrals_to_cartesian_tf, chain_in_plane, guess_amide_H, guess_amide_O, merge_cartesians
from .moldata import MolData
from .parameters import ADCParameters
from math import pi


class AngleDihedralCartesianEncoder(Autoencoder):

    def __init__(self, *args, **kwargs):
        super(AngleDihedralCartesianEncoder, self).__init__(*args, **kwargs)
        assert isinstance(self.p, ADCParameters)
        assert isinstance(self.train_moldata, MolData)

    def _prepare_data(self):
        self.train_moldata = self.train_data
        assert self.train_moldata.lengths.shape[1] == self.train_moldata.central_cartesians.shape[1] - 1
        assert self.train_moldata.angles.shape[1] == self.train_moldata.central_cartesians.shape[1] - 2
        assert self.train_moldata.dihedrals.shape[1] == self.train_moldata.central_cartesians.shape[1] - 3
        self.train_data = (self.train_moldata.angles,
                           self.train_moldata.dihedrals,
                           self.train_moldata.central_cartesians)

        if self.validation_data is not None:
            raise ValueError("validation data not supported yet")  # Todo: add support

    def _setup_network(self):
        self.inputs = self.data_iterator.get_next()
        if self.p.use_backbone_angles:
            self.main_inputs = tf.concat([self.inputs[0], self.inputs[1]], axis=1)
        else:
            self.main_inputs = self.inputs[1]
        self.main_inputs = tf.placeholder_with_default(self.main_inputs, self.main_inputs.shape)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2_reg_constant)
        encoded = self._encode(self.main_inputs)
        self.latent = tf.placeholder_with_default(encoded, encoded.shape)
        variable_summaries("latent", self.latent)
        self.generated = self._generate(self.latent)

        if self.p.use_backbone_angles:
            self.generated_angles = self.generated[:, :self.inputs[0].shape[1]]
            self.generated_dihedrals = self.generated[:, self.inputs[0].shape[1]:]
        else:
            self.generated_dihedrals = self.generated
            self.generated_angles = tf.tile(np.expand_dims(np.mean(self.train_moldata.angles, axis=0), axis=0),
                                            [tf.shape(self.generated_dihedrals)[0], 1])

        mean_lengths = np.expand_dims(np.mean(self.train_moldata.lengths, axis=0), axis=0)
        self.chain_in_plane = chain_in_plane(mean_lengths, self.generated_angles)
        self.cartesian = dihedrals_to_cartesian_tf(self.generated_dihedrals + pi,
                                                   self.chain_in_plane)

        self.amide_H_cartesian = guess_amide_H(self.cartesian, self.train_moldata.central_atoms.names)
        self.amide_O_cartesian = guess_amide_O(self.cartesian, self.train_moldata.central_atoms.names)

        self.cartesian_with_guessed_atoms = merge_cartesians(self.cartesian, self.train_moldata.central_atoms.names,
                                                             self.amide_H_cartesian, self.amide_O_cartesian)

        self.input_cartesian_pairwise_dist = pairwise_dist(self.inputs[2][:, self.p.cartesian_pwd_start:
                                                                      self.p.cartesian_pwd_stop:
                                                                      self.p.cartesian_pwd_step], flat=True)

        self.gen_cartesian_pairwise_dist = pairwise_dist(self.cartesian[:, self.p.cartesian_pwd_start:
                                                                          self.p.cartesian_pwd_stop:
                                                                          self.p.cartesian_pwd_step], flat=True)

        self.clashes = tf.count_nonzero(pairwise_dist(self.cartesian, flat=True) < 1, axis=1, dtype=tf.float32)
        tf.summary.scalar("clashes", tf.reduce_mean(self.clashes))

    def _setup_cost(self):
        self._dihedral_cost()
        self._angle_cost()
        self._cartesian_cost()

        self._distance_cost()
        self._cartesian_distance_cost()
        self._center_cost()
        self._l2_reg_cost()

    def _dihedral_cost(self):
        if self.p.dihedral_cost_scale is not None:
            if self.p.dihedral_cost_variant == "mean_square":
                dihedral_cost = tf.reduce_mean(
                    tf.square(periodic_distance(self.inputs[1], self.generated_dihedrals, self.p.periodicity)))
            elif self.p.dihedral_cost_variant == "mean_abs":
                dihedral_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(self.inputs[1], self.generated_dihedrals, self.p.periodicity)))
            elif self.p.dihedral_cost_variant == "mean_norm":
                dihedral_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(self.inputs[1], self.generated_dihedrals, self.p.periodicity),
                            axis=1))
            else:
                raise ValueError("dihedral_cost_variant {} not available".format(self.p.auto_cost_variant))
            tf.summary.scalar("dihedral_cost", dihedral_cost)
            if self.p.dihedral_cost_scale != 0:
                self.cost += self.p.dihedral_cost_scale * dihedral_cost

    def _angle_cost(self):
        if self.p.angle_cost_scale is not None:
            if self.p.angle_cost_variant == "mean_square":
                angle_cost = tf.reduce_mean(
                    tf.square(periodic_distance(self.inputs[0], self.generated_angles, self.p.periodicity)))
            elif self.p.angle_cost_variant == "mean_abs":
                angle_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(self.inputs[0], self.generated_angles, self.p.periodicity)))
            elif self.p.angle_cost_variant == "mean_norm":
                angle_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(self.inputs[0], self.generated_angles, self.p.periodicity),
                            axis=1))
            else:
                raise ValueError("angle_cost_variant {} not available".format(self.p.auto_cost_variant))
            tf.summary.scalar("angle_cost", angle_cost)
            if self.p.angle_cost_scale != 0:
                self.cost += self.p.angle_cost_scale * angle_cost

    def _distance_cost(self):
        if self.p.distance_cost_scale is not None:
            dist_cost = distance_cost(self.main_inputs, self.latent, *self.p.dist_sig_parameters, self.p.periodicity)
            tf.summary.scalar("distance_cost", dist_cost)
            if self.p.distance_cost_scale != 0:
                self.cost += self.p.distance_cost_scale * dist_cost

    def _cartesian_distance_cost(self):
        if self.p.cartesian_distance_cost_scale is not None:
            dist_cost = distance_cost(self.input_cartesian_pairwise_dist, self.latent,
                                      *self.p.cartesian_dist_sig_parameters,
                                      float("inf"))
            tf.summary.scalar("cartesian_distance_cost", dist_cost)
            if self.p.cartesian_distance_cost_scale != 0:
                self.cost += self.p.cartesian_distance_cost_scale * dist_cost

    def _cartesian_cost(self):
        if self.p.cartesian_cost_scale is not None:
            if self.p.cartesian_cost_variant == "mean_square":
                cartesian_cost = tf.reduce_mean(tf.square(
                    self.input_cartesian_pairwise_dist - self.gen_cartesian_pairwise_dist))
            elif self.p.cartesian_cost_variant == "mean_abs":
                cartesian_cost = tf.reduce_mean(tf.abs(
                    self.input_cartesian_pairwise_dist - self.gen_cartesian_pairwise_dist))
            elif self.p.cartesian_cost_variant == "mean_norm":
                cartesian_cost = tf.reduce_mean(tf.norm(
                    self.input_cartesian_pairwise_dist - self.gen_cartesian_pairwise_dist, axis=1))
            else:
                raise ValueError("cartesian_cost_variant {} not available".
                                 format(self.p.dihedral_to_cartesian_cost_variant))

            tf.summary.scalar("cartesian_cost", cartesian_cost)
            if self.p.cartesian_cost_scale != 0:
                self.cost += self.p.cartesian_cost_scale * cartesian_cost

    def generate(self, latent, quantity=None):
        if quantity is None:
            all_dihedrals = []
            all_cartesians = []
            all_angles = []
            batches = np.array_split(latent, max(1, int(len(latent) / 2048)))
            for batch in batches:
                angles, dihedrals, cartesians = self.sess.run((self.generated_angles,
                                                               self.generated_dihedrals,
                                                               self.cartesian_with_guessed_atoms),
                                                              feed_dict={self.latent: batch})
                all_dihedrals.append(dihedrals)
                all_cartesians.append(cartesians)
                all_angles.append(angles)
            all_dihedrals = np.concatenate(all_dihedrals, axis=0)
            all_cartesians = np.concatenate(all_cartesians, axis=0)
            all_angles = np.concatenate(all_angles, axis=0)
            return all_angles, all_dihedrals, all_cartesians

        else:
            results = []
            batches = np.array_split(latent, max(1, int(len(latent) / 2048)))
            for batch in batches:
                results.append(self.sess.run(quantity, feed_dict={self.latent: batch}))
            return np.concatenate(results, axis=0)
