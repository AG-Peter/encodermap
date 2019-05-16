from .autoencoder import Autoencoder
import tensorflow as tf
import numpy as np
from .misc import periodic_distance, variable_summaries, add_layer_summaries, distance_cost, pairwise_dist
from .backmapping import dihedrals_to_cartesian_tf, chain_in_plane, guess_amide_H, guess_amide_O, merge_cartesians
import os
from .parameters import Parameters
from math import pi


class AngleDihedralCartesianEncoder(Autoencoder):
    def __init__(self, parameters, moldata, validation_data=None, checkpoint_path=None, trainable=True):
        """
        :param parameters: Parameters object as defined in :py:class:`encodermap.parameters.Parameters`

        :param train_data: 2d numpy array where each row is treated as a training point

        :param validation_data: A 2d numpy array. This data will only be used to calculate a validation error during
                                training. It will not be used for training.

        :param checkpoint_path: If a checkpoint path is given values like neural network weights stored in this
                                checkpoint will be restored.

        :param n_inputs: If no train_data is given, for example when an already trained network is restored from a
                         checkpoint, the number of of inputs needs to be given. This should be equal to the number of
                         columns of the train_data the network was originally trained with.
        """
        # Parameters:
        self.p = parameters  # type: Parameters
        self.p.save()
        print("Output files are saved to {}".format(self.p.main_path),
              "as defined in 'main_path' in the parameters.")

        self.graph = tf.Graph()
        with self.graph.as_default():

            if validation_data is None:
                self.validation_data = None
            else:
                self.validation_data = validation_data.astype(np.float32)
                # Todo: allow lists of validation data
            self.moldata = moldata
            assert moldata.lengths.shape[1] == moldata.central_cartesians.shape[1] - 1
            assert moldata.angles.shape[1] == moldata.central_cartesians.shape[1] - 2
            assert moldata.dihedrals.shape[1] == moldata.central_cartesians.shape[1] - 3
            self.train_data = (moldata.angles, moldata.dihedrals, moldata.central_cartesians)

            self.data_placeholders = tuple(tf.placeholder(dat.dtype, dat.shape) for dat in self.train_data)
            self.data_set = tf.data.Dataset.from_tensor_slices(self.data_placeholders)
            self.data_set = self.data_set.shuffle(buffer_size=len(self.train_data[0]))  # Todo: make parameter
            # self.data_set = self.data_set.shuffle(buffer_size=10)
            self.data_set = self.data_set.repeat()
            self.data_set = self.data_set.batch(self.p.batch_size)
            self.data_iterator = self.data_set.make_initializable_iterator()

            # Setup Network:
            self.inputs = self.data_iterator.get_next()
            self.main_inputs = tf.concat([self.inputs[0], self.inputs[1]], axis=1)
            self.main_inputs = tf.placeholder_with_default(self.main_inputs, self.main_inputs.shape)
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2_reg_constant)
            encoded = self._encode(self.main_inputs)
            self.latent = tf.placeholder_with_default(encoded, encoded.shape)
            variable_summaries("latent", self.latent)
            self.generated = self._generate(self.latent)

            self.generated_angles = self.generated[:, :self.inputs[0].shape[1]]
            self.generated_dihedrals = self.generated[:, self.inputs[0].shape[1]:]

            mean_lengths = np.expand_dims(np.mean(moldata.lengths, axis=0), axis=0)
            self.chain_in_plane = chain_in_plane(mean_lengths, self.generated_angles)
            self.cartesian = dihedrals_to_cartesian_tf(self.generated_dihedrals + pi,
                                                       self.chain_in_plane)

            self.amide_H_cartesian = guess_amide_H(self.cartesian, moldata.central_atoms.names)
            self.amide_O_cartesian = guess_amide_O(self.cartesian, moldata.central_atoms.names)

            self.cartesian_with_guessed_atoms = merge_cartesians(self.cartesian, moldata.central_atoms.names,
                                                                 self.amide_H_cartesian, self.amide_O_cartesian)

            if trainable:
                # Define Cost function:
                self.cost = self._cost()

                # Setup Optimizer:
                self.optimizer = tf.train.AdamOptimizer(self.p.learning_rate)
                gradients = self.optimizer.compute_gradients(self.cost)
                variable_summaries("gradients_last", gradients[-1])
                self.global_step = tf.train.create_global_step()
                self.optimize = self.optimizer.apply_gradients(gradients, global_step=self.global_step)

                self.merged_summaries = tf.summary.merge_all()

            # Setup Session
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.p.gpu_memory_fraction)
            gpu_options = tf.GPUOptions(allow_growth = True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.data_iterator.initializer,
                          feed_dict={p: d for p, d in zip(self.data_placeholders, self.train_data)})
            if trainable:
                self.train_writer = tf.summary.FileWriter(os.path.join(self.p.main_path, "train"), self.sess.graph)
                if self.validation_data is not None:
                    self.validation_writer = tf.summary.FileWriter(os.path.join(self.p.main_path, "validation"), self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=100)

            # load Checkpoint
            if checkpoint_path:
                self.saver.restore(self.sess, checkpoint_path)

    def _cost(self):
        with tf.name_scope("cost"):
            cost = 0

            cartesian_pairwise_dist = pairwise_dist(self.inputs[2])
            # if self.p.auto_cost_scale != 0:
            if True:
                auto_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(self.main_inputs, self.generated, self.p.periodicity)))
                tf.summary.scalar("auto_cost", auto_cost)
                cost += self.p.auto_cost_scale * auto_cost

            if self.p.distance_cost_scale != 0:
                dist_cost = distance_cost(self.main_inputs, self.latent, *self.p.dist_sig_parameters, self.p.periodicity)
                tf.summary.scalar("distance_cost", dist_cost)
                cost += self.p.distance_cost_scale * dist_cost

            # if self.p.cartesian_distance_cost_scale:
            #     cpd_shape = cartesian_pairwise_dist.shape
            #     new_shape = (-1, cpd_shape[1]*cpd_shape[2])
            #     dist_cost = distance_cost(tf.reshape(cartesian_pairwise_dist, shape=new_shape), self.latent, *self.p.cartesian_dist_sig_parameters,
            #                               float("inf"))
            #     tf.summary.scalar("cartesian_distance_cost", dist_cost)
            #     cost += self.p.cartesian_distance_cost_scale * dist_cost

            if self.p.center_cost_scale != 0:
                center_cost = tf.reduce_mean(tf.square(self.latent))
                tf.summary.scalar("center_cost", center_cost)
                cost += self.p.center_cost_scale * center_cost

            if self.p.l2_reg_constant != 0:
                reg_cost = tf.losses.get_regularization_loss()
                tf.summary.scalar("reg_cost", reg_cost)
                cost += reg_cost

            gen_cartesian_pairwise_dist = pairwise_dist(self.cartesian)
            tf.summary.scalar("clashes", tf.reduce_mean(tf.count_nonzero(gen_cartesian_pairwise_dist < 1, axis=(1, 2), dtype=tf.float32)
                                                        - int(self.cartesian.shape[1])) / 2)
            dihedrals_to_cartesian_cost = tf.reduce_mean(tf.abs(
                self.transform_pairwise_dists(cartesian_pairwise_dist)
                - self.transform_pairwise_dists(gen_cartesian_pairwise_dist)))
            if self.p.dihedral_to_cartesian_cost_scale != 0:
                cost += self.p.dihedral_to_cartesian_cost_scale * dihedrals_to_cartesian_cost
            tf.summary.scalar("dihedrals_to_cartesian_cost", dihedrals_to_cartesian_cost)

        tf.summary.scalar("cost", cost)
        return cost

    # @staticmethod
    # def transform_pairwise_dists(pwd):
    #     return 1/(pwd + 1e-16)

    @staticmethod
    def transform_pairwise_dists(pwd):
        return pwd

    def generate(self, latent):
        """
        Generates new high-dimensional points based on given low-dimensional points using the decoder part of the
        autoencoder.

        :param latent: 2d numpy array containing points in the low-dimensional space. The number of columns must be
                       equal to the number of neurons in the bottleneck layer of the autoencoder.
        :return:
        """
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
