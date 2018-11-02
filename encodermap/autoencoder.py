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
import numpy as np
from .misc import periodic_distance, variable_summaries, add_layer_summaries, sketchmap_cost
import os
from .parameters import Parameters
from tqdm import tqdm


class Autoencoder:

    def __init__(self, parameters, train_data=None, validation_data=None, checkpoint_path=None, n_inputs=None):

        # Parameters:
        self.p = parameters  # type: Parameters
        self.p.save()

        # Load Data:
        if train_data is None:
            assert n_inputs is not None, "If no train_data is given, n_inputs needs to be given"
            self.train_data = np.zeros((3, n_inputs), dtype=np.float32)
        else:
            self.train_data = train_data.astype(np.float32)

        if validation_data is None:
            self.validation_data = None
        else:
            self.validation_data = validation_data.astype(np.float32)

        self.data_placeholder = tf.placeholder(self.train_data.dtype, self.train_data.shape)
        self.data_set = tf.data.Dataset.from_tensor_slices(self.data_placeholder)
        self.data_set = self.data_set.shuffle(buffer_size=len(self.train_data))
        self.data_set = self.data_set.repeat()
        self.data_set = self.data_set.batch(self.p.batch_size)
        self.data_iterator = self.data_set.make_initializable_iterator()

        # Setup Network:
        self.inputs = self.data_iterator.get_next()
        self.inputs = tf.placeholder_with_default(self.inputs, self.inputs.shape)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2_reg_constant)
        encoded = self._encode(self.inputs)
        self.latent = tf.placeholder_with_default(encoded, encoded.shape)
        variable_summaries("latent", self.latent)
        self.generated = self._generate(self.latent)

        # Define Cost function:
        self.cost = self._cost()

        # Setup Optimizer:
        self.optimizer = tf.train.AdamOptimizer(self.p.learning_rate)
        gradients = self.optimizer.compute_gradients(self.cost)
        # gradients = [(tf.Print(grad, [grad.shape, grad], "gradients", summarize=50), var) for grad, var in gradients]
        self.optimize = self.optimizer.apply_gradients(gradients)

        self.merged_summaries = tf.summary.merge_all()

        # Setup Session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.p.gpu_memory_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.data_iterator.initializer, feed_dict={self.data_placeholder: self.train_data})
        self.train_writer = tf.summary.FileWriter(os.path.join(self.p.main_path, "train"), self.sess.graph)
        if self.validation_data is not None:
            self.validation_writer = tf.summary.FileWriter(os.path.join(self.p.main_path, "validation"), self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=100)

        # load Checkpoint
        if checkpoint_path:
            self.saver.restore(self.sess, checkpoint_path)

    def _encode(self, inputs):
        with tf.name_scope("encoder"):
            if self.p.periodicity < float("inf"):
                current_layer = tf.concat([tf.sin(inputs), tf.cos(inputs)], 1)
            else:
                current_layer = inputs
            for i, (n_neurons, act_fun) in enumerate(zip(self.p.n_neurons, self.p.activation_functions[1:])):
                if act_fun:
                    act_fun = getattr(tf.nn, act_fun)
                else:
                    act_fun = None
                variable_summaries("activation{}".format(i), current_layer)
                dense = tf.layers.Dense(n_neurons, activation=act_fun,
                                        kernel_initializer=tf.variance_scaling_initializer,
                                        kernel_regularizer=self.regularizer,
                                        bias_initializer=tf.random_normal_initializer(0.1, 0.05))
                current_layer = dense(current_layer)
                add_layer_summaries(dense)
            return current_layer

    def _generate(self, inputs):
        with tf.name_scope("generator"):
            current_layer = inputs
            if self.p.periodicity < float("inf"):
                n_neurons_with_inputs = [self.train_data.shape[1] * 2] + self.p.n_neurons
            else:
                n_neurons_with_inputs = [self.train_data.shape[1]] + self.p.n_neurons
            for n_neurons, act_fun in zip(n_neurons_with_inputs[-2::-1], self.p.activation_functions[-2::-1]):
                if act_fun:
                    act_fun = getattr(tf.nn, act_fun)
                else:
                    act_fun = None
                current_layer = tf.layers.dense(current_layer, n_neurons,
                                                activation=act_fun,
                                                kernel_initializer=tf.variance_scaling_initializer,
                                                kernel_regularizer=self.regularizer,
                                                bias_initializer=tf.random_normal_initializer(0.1, 0.05))
            if self.p.periodicity < float("inf"):
                split = self.train_data.shape[1]
                current_layer = tf.atan2(current_layer[:, :split], current_layer[:, split:])
            return current_layer

    def _cost(self):
        self.auto_cost = tf.reduce_mean(
            tf.norm(periodic_distance(self.inputs, self.generated, self.p.periodicity), axis=1))
        self.sketch_cost = sketchmap_cost(self.inputs, self.latent, *self.p.sketch_parameters)
        self.mean_cost = tf.reduce_mean(tf.square(tf.norm(self.latent, axis=1)))
        # Todo: square of square root is ineffcient
        self.reg_cost = tf.losses.get_regularization_loss()
        cost = self.p.sketch_cost_scale * self.sketch_cost \
                    + self.p.auto_cost_scale * self.auto_cost \
                    + self.p.mean_cost_scale * self.mean_cost \
                    + self.reg_cost

        tf.summary.scalar("cost", cost)
        tf.summary.scalar("auto_cost", self.auto_cost)
        tf.summary.scalar("sketch_cost", self.sketch_cost)
        tf.summary.scalar("mean_cost", self.mean_cost)
        tf.summary.scalar("reg_cost", self.reg_cost)
        return cost

    def encode(self, inputs):
        latents = []
        batches = np.array_split(inputs, max(1, int(len(inputs) / 2048)))
        for batch in batches:
            latent = self.sess.run(self.latent, feed_dict={self.inputs: batch})
            latents.append(latent)
        latents = np.concatenate(latents, axis=0)
        return latents

    def generate(self, latent):
        generateds = []
        batches = np.array_split(latent, max(1, int(len(latent) / 2048)))
        for batch in batches:
            generated = self.sess.run(self.generated, feed_dict={self.latent: batch})
            generateds.append(generated)
        generated = np.concatenate(generateds, axis=0)
        return generated

    def random_batch(self, data, batch_size=None):
        batch_size = batch_size or self.p.batch_size
        batch = data[np.random.choice(len(data), batch_size, replace=False)]
        return batch

    def train(self):
        step = 0
        for step in tqdm(range(self.p.n_steps)):
            # feed_dict = {self.inputs: self.random_batch()}

            if step % self.p.summary_step == 0:
                _, summary_values = self.sess.run((self.optimize, self.merged_summaries))
                self.train_writer.add_summary(summary_values, step)
                if self.validation_data is not None:
                    summary_values = self.sess.run(self.merged_summaries,
                                                   feed_dict={self.inputs: self.random_batch(self.validation_data)})
                    self.validation_writer.add_summary(summary_values, step)
            else:
                self.sess.run(self.optimize)

            if step % self.p.checkpoint_step == 0 and step != 0:
                self.saver.save(self.sess, os.path.join(self.p.main_path, "checkpoints", "step{}.ckpt".format(step)))
        else:
            self.saver.save(self.sess, os.path.join(self.p.main_path, "checkpoints", "step{}.ckpt".format(step)))
