import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
import numpy as np
from .misc import periodic_distance, variable_summaries, add_layer_summaries, distance_cost, pairwise_dist
from .backmapping import dihedrals_to_cartesian_tf
import os
from .parameters import Parameters
from tqdm import tqdm
from tensorflow.python.client import timeline
from math import pi


class Autoencoder:

    def __init__(self, parameters, train_data=None, validation_data=None, checkpoint_path=None, n_inputs=None,
                 read_only=False):
        """
        :param parameters: Parameters object as defined in :class:`.Parameters`

        :param train_data: 2d numpy array where each row is treated as a training point

        :param validation_data: A 2d numpy array. This data will only be used to calculate a validation error during
                                training. It will not be used for training.

        :param checkpoint_path: If a checkpoint path is given, values like neural network weights stored in this
                                checkpoint will be restored.

        :param n_inputs: If no train_data is given, for example when an already trained network is restored from a
                         checkpoint, the number of of inputs needs to be given. This should be equal to the number of
                         columns of the train_data the network was originally trained with.
        :param read_only: if True, no output is writen
        """
        # Parameters:
        self.p = parameters
        self.n_inputs = n_inputs
        if not read_only:
            self.p.save()
        print("Output files are saved to {}".format(self.p.main_path),
              "as defined in 'main_path' in the parameters.")

        self.train_data = train_data
        self.validation_data = validation_data

        self._prepare_data()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.train.create_global_step()

            self._setup_data_iterator()

            self._setup_network()

            with tf.name_scope("cost"):
                self.cost = 0
                self._setup_cost()
                tf.summary.scalar("combined_cost", self.cost)

            # Setup Optimizer:
            self.optimizer = tf.train.AdamOptimizer(self.p.learning_rate)
            gradients = self.optimizer.compute_gradients(self.cost)
            self.optimize = self.optimizer.apply_gradients(gradients, global_step=self.global_step)

            self.merged_summaries = tf.summary.merge_all()

            # Setup Session
            if self.p.gpu_memory_fraction == 0:
                gpu_options = tf.GPUOptions(allow_growth=True)
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.p.gpu_memory_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.data_iterator.initializer,
                          feed_dict={p: d for p, d in zip(self.data_placeholders, self.train_data)})
            if not read_only:
                self.train_writer = tf.summary.FileWriter(os.path.join(self.p.main_path, "train"), self.sess.graph)
                if self.validation_data is not None:
                    self.validation_writer = tf.summary.FileWriter(os.path.join(self.p.main_path, "validation"), self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=100)

            # load Checkpoint
            if checkpoint_path:
                self.saver.restore(self.sess, checkpoint_path)

    def _prepare_data(self):
        if self.train_data is None:
            assert self.n_inputs is not None, "If no train_data is given, n_inputs needs to be given"
            self.train_data = [np.zeros((3, self.n_inputs), dtype=np.float32)]
        elif isinstance(self.train_data, np.ndarray):
            self.train_data = [self.train_data.astype(np.float32)]
        elif isinstance(self.train_data, (list, tuple)):
            self.train_data = [dat.astype(np.float32) for dat in self.train_data]
        else:
            raise ValueError("{} is not supported as input type for train_data".format(type(train_data)))

        if self.validation_data is not None:
            self.validation_data = self.validation_data.astype(np.float32)
            # Todo: allow lists of validation data

    def _setup_network(self):
        self.inputs = self.data_iterator.get_next()
        self.main_inputs = self.inputs[0]
        self.main_inputs = tf.placeholder_with_default(self.main_inputs, self.main_inputs.shape)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2_reg_constant)
        encoded = self._encode(self.main_inputs)
        self.latent = tf.placeholder_with_default(encoded, encoded.shape)
        variable_summaries("latent", self.latent)
        self.generated = self._generate(self.latent)

    def _setup_data_iterator(self):
        self.data_placeholders = tuple(tf.placeholder(dat.dtype, dat.shape) for dat in self.train_data)
        self.data_set = tf.data.Dataset.from_tensor_slices(self.data_placeholders)
        self.data_set = self.data_set.shuffle(buffer_size=len(self.train_data[0]))
        self.data_set = self.data_set.repeat()
        self.data_set = self.data_set.batch(self.p.batch_size)
        self.data_iterator = self.data_set.make_initializable_iterator()

    def _encode(self, inputs):
        with tf.name_scope("encoder"):
            if self.p.periodicity < float("inf"):
                if self.p.periodicity != 2 * pi:
                    inputs = inputs / self.p.periodicity * 2 * pi
                self.unit_circle_inputs = tf.concat([tf.sin(inputs), tf.cos(inputs)], 1)
                current_layer = self.unit_circle_inputs
            else:
                current_layer = inputs

            assert len(self.p.n_neurons) == len(self.p.activation_functions) - 1, \
                "you need one activation function more then layers given in n_neurons"
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
                n_neurons_with_inputs = [self.main_inputs.shape[1] * 2] + self.p.n_neurons
            else:
                n_neurons_with_inputs = [self.main_inputs.shape[1]] + self.p.n_neurons
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
                split = self.main_inputs.shape[1]
                current_layer = tf.atan2(current_layer[:, :split], current_layer[:, split:])
                if self.p.periodicity != 2 * pi:
                    current_layer = current_layer / (2*pi) * self.p.periodicity
            return current_layer

    def _setup_cost(self):
        self._auto_cost()
        self._center_cost()
        self._l2_reg_cost()

    def _auto_cost(self):
        if self.p.auto_cost_scale is not None:
            if self.p.auto_cost_variant == "mean_square":
                auto_cost = tf.reduce_mean(
                    tf.square(periodic_distance(self.main_inputs, self.generated, self.p.periodicity)))
            elif self.p.auto_cost_variant == "mean_abs":
                auto_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(self.main_inputs, self.generated, self.p.periodicity)))
            elif self.p.auto_cost_variant == "mean_norm":
                auto_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(self.main_inputs, self.generated, self.p.periodicity), axis=1))
            else:
                raise ValueError("auto_cost_variant {} not available".format(self.p.auto_cost_variant))
            tf.summary.scalar("auto_cost", auto_cost)
            if self.p.auto_cost_scale != 0:
                self.cost += self.p.auto_cost_scale * auto_cost

    def _l2_reg_cost(self):
        if self.p.l2_reg_constant is not None:
            reg_cost = tf.losses.get_regularization_loss()
            tf.summary.scalar("reg_cost", reg_cost)
            if self.p.l2_reg_constant != 0:
                self.cost += reg_cost

    def _center_cost(self):
        if self.p.center_cost_scale is not None:
            center_cost = tf.reduce_mean(tf.square(self.latent))
            tf.summary.scalar("center_cost", center_cost)
            if self.p.center_cost_scale != 0:
                self.cost += self.p.center_cost_scale * center_cost

    def encode(self, inputs):
        """
        Projects high dimensional data to a low dimensional space using the encoder part of the autoencoder.

        :param inputs: 2d numpy array with the same number of columns as the used train_data
        :return: 2d numpy array with the point projected the the low dimensional space. The number of columns is equal
                 to the number of neurons in the bottleneck layer of the autoencoder.
        """
        latents = []
        batches = np.array_split(inputs, max(1, int(len(inputs) / 2048)))
        for batch in batches:
            latent = self.sess.run(self.latent, feed_dict={self.main_inputs: batch})
            latents.append(latent)
        latents = np.concatenate(latents, axis=0)
        return latents

    def generate(self, latent):
        """
        Generates new high-dimensional points based on given low-dimensional points using the decoder part of the
        autoencoder.

        :param latent: 2d numpy array containing points in the low-dimensional space. The number of columns must be
                       equal to the number of neurons in the bottleneck layer of the autoencoder.
        :return: 2d numpy array containing points in the high-dimensional space.
        """
        generateds = []
        batches = np.array_split(latent, max(1, int(len(latent) / 2048)))
        for batch in batches:
            generated = self.sess.run(self.generated, feed_dict={self.latent: batch})
            generateds.append(generated)
        generated = np.concatenate(generateds, axis=0)
        return generated

    def _random_batch(self, data, batch_size=None):
        batch_size = batch_size or self.p.batch_size
        batch = data[np.random.choice(len(data), batch_size, replace=False)]
        return batch

    def train(self):
        """
        Train the autoencoder as specified in the parameters object.
        """
        for i in tqdm(range(self.p.n_steps)):
            if (i+1) % self.p.summary_step == 0:
                # _, summary_values = self.sess.run((self.optimize, self.merged_summaries))
                _, summary_values = self.sess.run((self.optimize, self.merged_summaries))
                self.train_writer.add_summary(summary_values, self._step())
                if self.validation_data is not None:
                    summary_values = self.sess.run(self.merged_summaries,
                                                   feed_dict={self.main_inputs: self._random_batch(self.validation_data)})
                    self.validation_writer.add_summary(summary_values, self._step())
            else:
                self.sess.run(self.optimize)

            if (self._step()) % self.p.checkpoint_step == 0:
                self.saver.save(self.sess, os.path.join(self.p.main_path, "checkpoints",
                                                        "step{}.ckpt".format(self._step())))
        else:
            self.saver.save(self.sess, os.path.join(self.p.main_path, "checkpoints", "step{}.ckpt".format(self._step())))
            self.train_writer.flush()

    def _step(self):
        return tf.train.global_step(self.sess, self.global_step)

    def profile(self):
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range(5):
            self.sess.run(self.optimize, options=options, run_metadata=run_metadata)

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(self.p.main_path, 'timeline{}.json'.format(i)), 'w') as f:
                f.write(chrome_trace)

    def close(self):
        """
        Close tensorflow session to free resources.
        :return:
        """
        self.sess.close()
        try:
            tf_ops.dismantle_graph(self.graph)  # not implemented in older versions of tensorflow
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def __del__(self):
        self.close()
