# -*- coding: utf-8 -*-
# encodermap/autoencoder/autoencoder.py
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
"""Forward facing Autoencoder classes. Contains four classes:

* Autoencoder: Simple NN dense, fully connected AE architecture. Reg loss, auto loss and center loss
* EncoderMap: Uses the same architecture as `Autoencoder`, but adds another loss function.
* DihedralEncoderMap: Basically the same as `EncoderMap`, but rewrites the `generate` method to use
    an atomistic topology to rebuild a trajectory.
* AngleDihedralCartesianEncoderMap: Uses more loss functions and tries to learn a full all atom conformation.

"""


################################################################################
# Imports
################################################################################


from __future__ import annotations

import typing
import warnings
from copy import deepcopy
from typing import Literal, Optional, Union

import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow.keras

import encodermap

from ..callbacks.callbacks import (
    CheckpointSaver,
    ImageCallback,
    IncreaseCartesianCost,
    ProgressBar,
    TensorboardWriteBool,
)
from ..loss_functions.loss_functions import (
    angle_loss,
    auto_loss,
    cartesian_distance_loss,
    cartesian_loss,
    center_loss,
    dihedral_loss,
    distance_loss,
    reconstruction_loss,
    regularization_loss,
    side_dihedral_loss,
)
from ..misc.backmapping import dihedral_backmapping, mdtraj_backmapping
from ..misc.misc import BadError, create_n_cube, plot_model
from ..misc.saving_loading_models import load_model, save_model
from ..models.models import gen_functional_model, gen_sequential_model
from ..parameters.parameters import ADCParameters, Parameters
from ..trajinfo.info_all import Capturing, TrajEnsemble

################################################################################
# Typing
################################################################################


if typing.TYPE_CHECKING:
    import MDAnalysis as mda
    import mdtraj as md


################################################################################
# Globals
################################################################################


__all__ = [
    "Autoencoder",
    "EncoderMap",
    "AngleDihedralCartesianEncoderMap",
    "DihedralEncoderMap",
]


##############################################################################
# Function definition which allows self.p.tensorboard to be passed
# @function(self.p.tensorboard)
# def train(self):
#     # some training
##############################################################################


def function(f, tensorboard=False):
    """Compiles functions with `tensorflow.function` based on a `tensorboard`
    parameter.


    To understand the neccessity of this function, we need to have a look how
    tensorflow executes computations. There are two modes of execution:
    * eager mode: In eager mode, the computations are handles by python.
        The input types are python objects, and the output is a python object.
        This eager execution allows you to directly execute a calculation of
        two tensors (e.g. multiplication).
    * graph mode: In graph mode, computations are done inside tensorflow graphs,
        which are a collection of operations and tensors (i.e. data), that flow
        through the operations of the graph. These graphs, make tensorflow
        computations portable and significantly increase the performance of
        similar computations.
    Normally, you would accelerate a simple python function in tensorflow,
    by compiling it like so:

    ```python
    import tensorflow as tf

    @tf.function
    def multiply(a, b):
        return a * b

    multiply(tf.constant(2), tf.constant(3)).numpy()
    # 6
    ```

    However, the basic paradigm of accelerating the computation interferes with
    `encodermap.Parameters` `tensorboard=True` argument, as it writes a lot of
    additional information to tensorboard. Thus, a compilation with tf.function
    does not make sense here. That's why encodermap's `function` decorator
    takes an additional argument:

    """

    def wrapper(*args, **kwargs):
        tensorboard = kwargs.pop("tensorboard", False)
        """Wrapper of `encodermap.function`."""
        if tensorboard:
            warnings.warn(
                "Running in tensorboard mode writes a lot of stuff to tensorboard. For speed boost deactivate tensorboard mode."
            )
            result = f(*args, **kwargs)
        else:
            compiled = tf.function(f)
            result = compiled(*args, **kwargs)
        return result

    return wrapper


##############################################################################
# Public Classes
##############################################################################


class Autoencoder:
    """Main Autoencoder class preparing data, setting up the neural network and implementing training.

    This is the main class for neural networks inside EncoderMap. The class prepares the data
    (batching and shuffling), creates a `tf.keras.Model` of layers specified by the attributes of
    the `encodermap.Parameters` class. Depending on what Parent/Child-Class is instantiated
    a combination of cost functions is set up. Callbacks to Tensorboard are also set up.

    Attributes:
        train_data (np.ndarray): The numpy array of the train data passed at init.
        p (encodermap.Parameters): An `encodermap.Parameters()` class containing all info needed to set
            up the network.
        dataset (tensorflow.data.Dataset): The dataset that is actually used in training the keras model. The dataset
            is a batched, shuffled, infinitely-repeating dataset.
        read_only (bool): Variable telling the class whether it is allowed to write to disk (False) or not (True).
        optimizer (tf.keras.optimizers.Adam): Instance of the Adam optimizer with learning rate specified by
            the Parameters class.
        metrics (list): A list of metrics passed to the model when it is compiled.
        callbacks (list): A list of tf.keras.callbacks.Callback Sub-classes changing the behavior of the model during
            training. Some standard callbacks are always present like:
                * encodermap.callbacks.callbacks.ProgressBar:
                    A progress bar callback using tqdm giving the current progress of training and the
                    current loss.
                * CheckPointSaver:
                    A callback that saves the model every parameters.checkpoint_step steps into
                    the main directory. This callback will only be used, when `read_only` is False.
                * TensorboardWriteBool:
                    A callback that contains a boolean Tensor that will be True or False,
                    depending on the current training step and the summary_step in the parameters class. The loss
                    functions use this callback to decide whether they should write to Tensorboard. This callback
                    will only be present, when `read_only` is False and `parameters.tensorboard` is True.
            You can append your own callbacks to this list before executing Autoencoder.train().
        encoder (tf.keras.models.Model): The encoder (sub)model of `model`.
        decoder (tf.keras.models.Model): The decoder (sub)model of `model`.

    Methods:
        from_checkpoint: Rebuild the model from a checkpoint.
        add_images_to_tensorboard: Make tensorboard plot images.
        train: Starts the training of the tf.keras.models.Model.
        plot_network: Tries to plot the network. For this method to work graphviz, pydot and pydotplus needs to be installed.
        encode: Takes high-dimensional data and sends it through the encoder.
        decode: Takes low-dimensional data and sends it through the encoder.
        generate: Same as decode. For AngleDihedralCartesianAutoencoder classes this will build a protein strutcure.

    Note:
        Performance of tensorflow is not only dependant on your system's hardware and how the data is presented to
        the network (for this check out https://www.tensorflow.org/guide/data_performance), but also how you compiled
        tensorflow. Normal tensorflow (pip install tensorflow) is build without CPU extensions to work on many CPUs.
        However, Tensorflow can greatly benefit from using CPU instructions like AVX2, AVX512 that bring a speed-up
        in linear algebra computations of 300%. By building tensorflow from source you can activate these extensions.
        However, the CPU speed-up is dwarfed by the speed-up when you allow tensorflow to run on your GPU (grapohics
        card). To check whether a GPU is available run:
        `print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))`.
        Refer to these pages to install tensorflow for best performance:
        https://www.tensorflow.org/install/pip, https://www.tensorflow.org/install/gpu

    Examples:
        >>> import encodermap as em
        >>> # without providing any data, default parameters and a 4D hypercube as input data will be used.
        >>> e_map = em.EncoderMap(read_only=True)
        >>> print(e_map.train_data.shape)
        (16000, 4)
        >>> print(e_map.dataset)
        <BatchDataset element_spec=(TensorSpec(shape=(None, 4), dtype=tf.float32, name=None), TensorSpec(shape=(None, 4), dtype=tf.float32, name=None))>
        >>> print(e_map.encode(e_map.train_data).shape)
        (16000, 2)

    """

    def __init__(
        self,
        parameters=None,
        train_data: Optional[Union[np.ndarray, tf.Dataset]] = None,
        model=None,
        read_only=False,
        sparse=False,
    ):
        """Instantiate the Autoencoder class.

        Args:
            parameters (Union[encodermap.Parameters, None], optional): The parameters to be used. If None is
                provided default values (check them with print(em.Parameters.defaults_description()))
                are used. Defaults to None.
            train_data (Union[np.ndarray, tf.data.Dataset, None], optional): The train data. Can be one of the following:
                * None: If None is provided points on the edges of a 4-dimensional hypercube will be used as train data.
                * np.ndarray: If a numpy array is provided, it will be transformed into a batched tf.data.Dataset by
                    first making it an infinitely repeating dataset, shuffling it and the batching it with a batch
                    size specified by parameters.batch_size.
                * tf.data.Dataset: If a dataset is provided it will be used without making any adjustments. Make
                    sure, that the dataset uses `float32` as its type.
                Defaults to None.
            model (Union[tf.keras.models.Model, None], optional): Providing a keras model to this argument will make
                the Autoencoder/EncoderMap class use this model instead of the predefined ones. Make sure the model
                can accept EncoderMap's loss functions. If None is provided the model will be built using
                the specifications in parameters. Defaults to None.
            read_only (bool, optional): Whether the class is allowed to write to disk (False) or not (True). Defaults
                to False and will allow the class to write to disk.

        Raises:
            BadError: When read_only is `True` and `parameters.tensorboard` is `True`, this Exception will be raised,
                because they are mutually exclusive.

        """
        # parameters
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters

        if self.p.seed is not None:
            tf.random.set_seed(self.p.seed)
        self.read_only = read_only

        if not self.read_only:
            self.p.save()
            print(
                "Output files are saved to {}".format(self.p.main_path),
                "as defined in 'main_path' in the parameters.",
            )

        # check whether Tensorboard and Read-Only makes Sense
        if self.read_only and self.p.tensorboard:
            raise BadError(
                "Setting tensorboard and read_only True is not possible. Tensorboard will always write to disk."
                " If you received this Error while loading a trained model, pass read_only=False as an argument"
                f" or set overwrite_tensorboard_bool True to overwrite the tensorboard parameter."
            )

        # clear old sessions
        tf.keras.backend.clear_session()
        self.sparse = sparse

        # set up train_data
        if train_data is None:
            self.train_data = create_n_cube(4, seed=self.p.seed)[0].astype("float32")
            self.p.periodicity = float("inf")
        elif isinstance(train_data, np.ndarray):
            if np.any(np.isnan(train_data)):
                self.sparse = True
                print("Input contains nans. Using sparse network.")
                indices = np.stack(np.where(~np.isnan(train_data))).T.astype("int64")
                dense_shape = train_data.shape
                values = train_data[~np.isnan(train_data)].flatten().astype("float32")
                sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
                self.train_data = sparse_tensor
            else:
                self.train_data = train_data.astype("float32")
        elif isinstance(train_data, tf.data.Dataset):
            self.dataset = train_data
            try:
                for _, __ in self.dataset:
                    break
            except ValueError:
                if self.p.training == "auto":
                    print(
                        f"It seems like your dataset only yields tensors and not "
                        f"tuples of tensors. Tensorlfow is optimized for classification "
                        f"tasks, where datasets yield tuples of (data, classes). EncoderMap,"
                        f"however is a regression task, but uses the same code as the "
                        f"classification tasks. I will transform your dataset using "
                        f"the `tf.data.Dataset.zip()` function of `tf.data`. You can "
                        f"set the `training` parameter in the parameter class to "
                        f"'custom' to not alter your dataset."
                    )
                    self.dataset = tf.data.Dataset.zip((self.dataset, self.dataset))
                    for _, __ in self.dataset:
                        break
                else:
                    for _ in self.dataset:
                        break
            self.train_data = _
        else:
            raise TypeError(
                f"train_data must be `None`, `np.ndarray` or `tf.data.Dataset`. You supplied {type(train_data)}."
            )

        # check data and periodicity
        if not self.sparse and not train_data is None:
            if np.any(train_data > self.p.periodicity):
                raise Exception(
                    "There seems to be an error regarding the periodicity "
                    f"of your data. The chosen periodicity is {self.p.periodicity}, "
                    f"but there are datapoints outwards of this range: {train_data.max()}"
                )

        # prepare the data
        if isinstance(self.train_data, (np.ndarray, tf.sparse.SparseTensor)):
            if self.p.training == "auto":
                dataset = tf.data.Dataset.from_tensor_slices(
                    (self.train_data, self.train_data)
                )
            else:
                dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
            dataset = dataset.shuffle(
                buffer_size=self.train_data.shape[0], reshuffle_each_iteration=True
            )
            dataset = dataset.repeat()
            self.dataset = dataset.batch(self.p.batch_size)
        else:
            pass

        # ToDo: Make training faster with Autotune, XLA (jit) compilation, DataRecords
        # self.dataset = self.dataset.prefetch(self.p.batch_size * 4)
        # self.dataset = self.dataset.interleave(num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # num_batches = 20
        # num_threads = 8
        # if self.p.training == 'auto':
        #     ds = tf.data.Dataset.from_tensors((self.train_data, self.train_data)).repeat(self.p.batch_size * num_batches)
        # else:
        #     ds = tf.data.Dataset.from_tensors(self.train_data).repeat(self.p.batch_size * num_batches)
        # # ds = tf.data.Dataset.from_tensors(self.train_data).repeat(self.p.batch_size * num_batches)
        # ds = ds.batch(self.p.batch_size)
        # self.dataset = ds.interleave(lambda *args:tf.data.Dataset.from_tensor_slices(args), num_threads, 1, num_threads)

        # create model based on user input
        if model is None:
            self.model = self.p.model_api
        else:
            self._model = model

        # setup callbacks for nice progress bars and saving every now and then
        self._setup_callbacks()

        # create loss based on user input
        self.loss = self.p.loss

        # choose optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)

        # compile model
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

        # do this if tensorboard is true.
        if self.p.tensorboard:
            self._log_images = False
            # get the output from model summary.
            with Capturing() as output:
                self.model.summary()
            with open(self.p.main_path + "/model_summary.txt", "w") as f:
                f.write("\n".join(output))
            tf.keras.utils.plot_model(
                self.model,
                to_file=self.p.main_path + "/model_summary.png",
                show_shapes=True,
                rankdir="LR",
                expand_nested=True,
            )
            print(
                f"Saved a text-summary of the model and an image in {self.p.main_path},",
                "as specified in 'main_path' in the parameters.",
            )

            # sets up the tb callback to plot the model
            self.tb_callback = tf.keras.callbacks.TensorBoard(
                self.p.main_path, write_graph=True
            )
            self.tb_callback.set_model(self.model)

    def _setup_callbacks(self):
        """Sets up a list with callbacks to be passed to self.model.fit()"""
        self.metrics = []
        self.callbacks = []
        self.callbacks.append(ProgressBar(parameters=self.p))
        if not self.read_only:
            self.callbacks.append(CheckpointSaver(self.p))
        if self.p.tensorboard:
            self.tensorboard_write_bool = TensorboardWriteBool(self.p)
            self.callbacks.append(self.tensorboard_write_bool)
            file_writer = tf.summary.create_file_writer(self.p.main_path + "/train")
            file_writer.set_as_default()
            tf.summary.text(
                name=f"Parameters Summary for {self.p.main_path}",
                data=self.p.parameters,
                step=0,
            )
            # callbacks.append(self.tb_callback)
        else:
            self.tensorboard_write_bool = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        read_only=True,
        overwrite_tensorboard_bool=False,
        sparse=False,
    ):
        """Reconstructs the class from a checkpoint.

        Args:
            Checkpoint path (str): The path to the checkpoint. Most models are saved in parts (encoder, decoder)
                and thus the provided path often needs a wildcard (*). The `save()` method of this class prints
                a string with which the model can be reloaded.
            read_only (bool, optional): Whether to reload the model in read_only mode (True) or allow the `Autoencoder`
                class to write to disk (False). This option might collide with the tensorboard Parameter in the
                respective parameters.json file in the maith_path. Defaults to True.
            overwrite_tensorboard_bool (bool, optional): Whether to overwrite the tensorboard Parameter while reloading
                the class. This can be set to True to set the tensorboard parameter False and allow read_only.
                Defaults to False.

        Raises:
            BadError: When read_only is True, overwrite_tensorboard_bool is False and the reloaded parameters
                have tensorboard set to True.

        Returns:
            Autoencoder: Encodermap `Autoencoder` class.

        """
        return load_model(
            cls, checkpoint_path, read_only, overwrite_tensorboard_bool, sparse=sparse
        )

    @property
    def model(self):
        """tf.keras.models.Model: The tf.keras.Model model used for training."""
        return self._model

    @model.setter
    def model(self, model):
        """sets self.model according to `model_api` argument in self.parameters."""
        if model == "functional":
            for d in self.dataset:
                break
            if any([isinstance(_, tf.sparse.SparseTensor) for _ in d]):
                self.sparse = True
            self._model = gen_functional_model(self.dataset, self.p, sparse=self.sparse)
        elif model == "sequential":
            if isinstance(self.train_data, tf.sparse.SparseTensor):
                self.sparse = True
            self._model = gen_sequential_model(
                self.train_data.shape[1], self.p, sparse=self.sparse
            )
        elif model == "custom":
            raise NotImplementedError("No custom API currently supported")
        else:
            raise ValueError(
                f"API argument needs to be one of `functional`, `sequential`, `custom`. You provided '{model}'."
            )

    @property
    def encoder(self):
        """tf.keras.models.Model: Encoder part of the model."""
        return self._model.encoder_model

    @property
    def decoder(self):
        """tf.keras.models.Model: Decoder part of the model."""
        return self._model.decoder_model

    @property
    def loss(self):
        """(Union[list, string, function]): A list of loss functions passed to the model when it is compiled.
        When the main Autoencoder class is used and parameters.loss is 'emap_cost' this list is comprised of
        center_cost, regularization_cost, auto_cost. When the EncoderMap sub-class is used and parameters.loss is
        'emap_cost' distance_cost is added to the list. When parameters.loss is not 'emap_cost', the loss can either
        be a string ('mse'), or a function, that both are acceptable arguments for loss, when a keras model
        is compiled.

        """
        return self._loss

    @loss.setter
    def loss(self, loss):
        """sets self.loss according to `loss` in self.parameters."""
        if loss == "reconstruction_loss":
            self._loss = reconstruction_loss(self.model)
        elif loss == "emap_cost":
            self.auto_loss = auto_loss(self.model, self.p, self.tensorboard_write_bool)
            self.regularization_loss = regularization_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.center_loss = center_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self._loss = [self.auto_loss, self.regularization_loss, self.center_loss]
        elif loss == "mse":
            self._loss = "mse"
        else:
            raise ValueError(
                f"loss argument needs to be `reconstruction_loss`, `mse` or `emap_cost`. You provided '{loss}'."
            )

    def train(self):
        """Starts the training of the model."""
        if self.p.training == "custom" and self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "custom" and not self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "auto":
            if self.p.tensorboard and self._log_images:
                # get the old backend because the Tensorboard Images callback will set 'Agg'
                old_backend = matplotlib.get_backend()
            # start_time = time.perf_counter()
            self.history = self.model.fit(
                self.dataset,
                batch_size=self.p.batch_size,
                epochs=self.p.n_steps,
                steps_per_epoch=1,
                verbose=0,
                callbacks=self.callbacks,
            )
            # print("Execution time:", time.perf_counter() - start_time)
        else:
            raise ValueError(
                f"training argument needs to be `auto` or `custom`. You provided '{self.training}'."
            )
        self.save(step=self.p.n_steps)
        # reset the backend.
        if self.p.tensorboard and self._log_images:
            matplotlib.use(old_backend)

    def add_images_to_tensorboard(
        self,
        data=None,
        image_step=None,
        scatter_kws={"s": 20},
        hist_kws={"bins": 50},
        additional_fns=None,
        when="epoch",
    ):
        """Adds images to Tensorboard using the data in data and the ids in ids.

        Args:
            data (Union[np.ndarray, list, None], optional): The input-data will be passed through the encoder
                part of the autoencoder. If None is provided a set of 10000 points from the provided
                train data will be taken. A list is needed for the functional API of the ADCAutoencoder, that takes
                a list of [angles, dihedrals, side_dihedrals]. Defaults to None.
            image_step (Union[int, None], optional): The interval in which to plot images to tensorboard.
                If None is provided, the update step will be the same as parameters.summary_step. Defaults to None.
            scatter_kws (dict, optional): A dict with items that matplotlib.pyplot.scatter() will accept. Defaults to
                {'s': 20}, which sets an appropriate size of scatter points for the size of datasets encodermap is
                usually used for.
            hist_kws (dict, optional): A dict with items that matplotlib.pyplot.scatter() will accept. You can
                choose a colorbar here. Defaults to {'bins': 50} which sets an appropriate bin count  for the
                size of datasets encodermap is usually used for.
            additional_fns (Union[list, None], optional): A list of functions that will accept the low-dimensional
                output of the autoencoder's latent/bottleneck layer and return a tf.Tensor that can be logged
                by `tf.summary.image()`. See the notebook 'writing_custom_images_to_tensorboard.ipynb' in
                tutorials/notebooks_customization for more info. If None is provided no additional functions will be
                used to plot to tensorboard. Defaults to None.
            when (str, optional): When to log the images can be either 'batch', then the images will be logged after
                every step during training, or 'epoch', then only after every image_step epoch the images will be
                written. Defaults to 'epoch'.

        """
        if not self.p.tensorboard:
            print(
                "Nothing is written to Tensorboard for this Model. Please change parameters.tensorboard to True."
            )
            return
        if image_step is None:
            image_step = self.p.summary_step

        self._log_images = True

        # make a dataset for images
        if data is None:
            if isinstance(self.train_data, np.ndarray):
                data = self.train_data
            elif isinstance(self.train_data, list) or self.sparse:
                data = self.train_data
            else:
                data = list(self.dataset.take(int(10000 / self.p.batch_size)))
                data = np.stack(data)[:, 0, :].reshape(-1, self.train_data.shape[1])
        else:
            if type(data) != type(self.train_data):
                raise Exception(
                    f"Provided data has wrong type. Train data in this class is {type(self.train_data)}, provided data is {type(data)}"
                )

        self.callbacks.append(
            ImageCallback(
                data,
                image_step,
                scatter_kws=scatter_kws,
                hist_kws=hist_kws,
                additional_fns=additional_fns,
                when=when,
            )
        )
        if isinstance(data, (np.ndarray, tf.sparse.SparseTensor)):
            print(
                f"Logging images with {data.shape}-shaped data every {image_step} epochs to Tensorboard at {self.p.main_path}"
            )
        else:
            print(
                f"Logging images with {[i.shape for i in data]}-shaped data every {image_step} epochs to Tensorboard at {self.p.main_path}"
            )

    def plot_network(self):
        """Tries to plot the network using pydot, pydotplus and graphviz. Doesn't raise an exception if plotting is
        not possible.

        Note:
            Refer to this guide to install these programs:
            https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py

        """
        try:
            plot_model(self.model, self.train_data.shape[1])
        except:
            pass

    def encode(self, data=None):
        """Calls encoder part of model.

        Args:
            data (Union[np.ndarray, None], optional): The data to be passed top the encoder part.
                Can be either numpy ndarray or None. If None is provided a set of 10000 points from the provided
                train data will be taken. Defaults to None.

        Returns:
            np.ndarray: The output from the bottlenack/latent layer.

        """
        if data is None:
            data = self.train_data
        if hasattr(self.model, "encoder"):
            out = self.model.encoder(data)
        elif hasattr(self.model, "encoder_model"):
            out = self.model.encoder_model(data)
        if isinstance(out, list):
            out = [o.numpy() for o in out]
        else:
            out = out.numpy()
        return out

    def generate(self, data):
        """Duplication of decode.

        In Autoencoder and EncoderMap this method is equivalent to `decode()`. In AngleDihedralCartesianAutoencoder
        this method will be overwritten to produce output molecular conformations.

        Args:
            data (np.ndarray): The data to be passed to the decoder part of the model. Make sure that the
                shape of the data matches the number of neurons in the latent space.

        Returns:
            np.ndarray: Oue output from the decoder part.

        """
        return self.model.decoder(data)

    def decode(self, data):
        """Calls the decoder part of the model.

        AngleDihedralCartesianAutoencoder will, like the other two classes' output a tuple of data.

        Args:
            data (np.ndarray):  The data to be passed to the decoder part of the model. Make sure that the
                shape of the data matches the number of neurons in the latent space.

        Returns:
            np.ndarray: Oue output from the decoder part.
        """
        out = self.decoder(data)
        if isinstance(out, list):
            out = [o.numpy() for o in out]
        else:
            out = out.numpy()
        return out

    def save(self, step=None):
        """Saves the model to the current path defined in `parameters.main_path`.

        Args:
            step (Union[int, None], optional): Does not actually save the model at the given training step, but rather
                changes the string used for saving the model from an datetime format to another.

        """
        if not self.read_only:
            save_model(self.model, self.p.main_path, self.__class__.__name__, step=step)

    def close(self):
        """Clears the current keras backend and frees up resources."""
        # clear old sessions
        tf.keras.backend.clear_session()


class EncoderMap(Autoencoder):
    """Complete copy of Autoencoder class but uses additional distance cost
    scaled by the SketchMap sigmoid params"""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        read_only=True,
        overwrite_tensorboard_bool=False,
        sparse=False,
    ):
        """Reconstructs the model from a checkpoint."""
        # Is this classmethod necessary? We need to make sure the class knows all losses.
        # And I don't know if the parent class calls the correct loss.setter
        return load_model(
            cls, checkpoint_path, read_only, overwrite_tensorboard_bool, sparse=sparse
        )

    @Autoencoder.loss.setter
    def loss(self, loss):
        if loss == "reconstruction_loss":
            self._loss = reconstruction_loss(self.model)
        elif loss == "emap_cost":
            self.auto_loss = auto_loss(self.model, self.p, self.tensorboard_write_bool)
            self.regularization_loss = regularization_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.center_loss = center_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            # this cost is new
            self.distance_loss = distance_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self._loss = [
                self.auto_loss,
                self.regularization_loss,
                self.center_loss,
                self.distance_loss,
            ]
        elif loss == "mse":
            self._loss = "mse"
        else:
            raise ValueError(
                f"loss argument needs to be `reconstruction_loss`, `mse` or `emap_cost`. You provided '{loss}'."
            )


class DihedralEncoderMap(EncoderMap):
    """Similar to the `EncoderMap` class, but overwrites the `generate` method.

    Using this class, instead of tbe `EncoderMap` class, the `generate` method,
    needs an additional argument: `top`, which should be a topology file. This
    topology will be used as a base on which the dihedrals of the `decode`
    method are applied.

    """

    def generate(self, data: np.ndarray, top: str) -> MDAnalysis.Universe:
        """Overwrites `EncoderMap`'s generate method and actually does backmapping if a list of dihedrals is
        provided.

        Args:
            data (np.ndarray): The low-dimensional/latent/bottleneck data. A ndim==2 numpy array with xy coordinates
                of points in latent space.
            top (str): Topology file for this run of EncoderMap (can be .pdb, .gro, .. etc.).

        Returns:
            MDAnalysis.Universe: The topology with the provided backbone torsions.

        Examples:
            >>> # get some time-resolved pdb files
            >>> import requests
            >>> import numpy as np
            >>> pdb_link = 'https://files.rcsb.org/view/1YUF.pdb'
            >>> contents = requests.get(pdb_link).text
            >>> print(contents.splitlines()[0]) # doctest: +SKIP
            HEADER    GROWTH FACTOR                           01-APR-96   1YUF
            >>> # fake a file with stringio
            >>> from io import StringIO
            >>> import MDAnalysis as mda
            >>> import numpy as np
            >>> file = StringIO(contents)
            >>> # pass it to MDAnalysis
            >>> u = mda.Universe(file, format='PDB')
            >>> print(u)
            <Universe with 720 atoms>
            >>> # select the atomgroups
            >>> ags = [*[res.psi_selection() for res in u.residues],
            ...        *[res.omega_selection() for res in u.residues],
            ...        *[res.phi_selection() for res in u.residues]
            ...        ]
            >>> # filter Nones
            >>> ags = list(filter(lambda x: False if x is None else True, ags))
            >>> print(ags[0][0]) # doctest: +SKIP
            <Atom 3: C of type C of resname VAL, resid 1 and segid A and altLoc >
            >>> # Run dihedral Angles
            >>> from MDAnalysis.analysis.dihedrals import Dihedral
            >>> R = np.deg2rad(Dihedral(ags).run().results.angles)
            >>> print(R.shape)
            (16, 147)
            >>> # import EncoderMap and define parameters
            >>> from encodermap.autoencoder import DihedralEncoderMap
            >>> import encodermap as em
            >>> parameters = em.Parameters(
            ... dist_sig_parameters = (4.5, 12, 6, 1, 2, 6),
            ... periodicity = 2*np.pi,
            ... l2_reg_constant = 10.0,
            ... summary_step = 5,
            ... tensorboard = False,
            ... )
            >>> e_map = DihedralEncoderMap(parameters, R, read_only=True)
            >>> print(e_map.__class__.__name__)
            DihedralEncoderMap
            >>> # get some low-dimensional data
            >>> lowd = np.random.random((100, 2))
            >>> # use the generate method to get a new MDAnalysis universe
            >>> # but first remove the time resolution
            >>> file = StringIO(contents.split('MODEL        2')[0])
            >>> new = e_map.generate(lowd, file)
            >>> print(new.trajectory.coordinate_array.shape)
            (100, 720, 3)
            >>> # check whether frame 0 of u and new_u are different
            >>> for ts in u.trajectory:
            ...     a1 = ts.positions
            ...     break
            >>> print(np.array_equal(a1, new.trajectory.coordinate_array[0]))
            False

        """
        assert np.any(data)
        dihedrals = self.decode(data)
        assert np.any(dihedrals)
        uni = dihedral_backmapping(top, dihedrals)
        return uni


class AngleDihedralCartesianEncoderMap(Autoencoder):
    """Different `__init__` method, than Autoencoder Class. Uses callbacks to tune-in cartesian cost.

    Overwritten methods: `_set_up_callbacks` and `generate`.

    Examples:
        >>> import encodermap as em
        >>> # Load two trajectories
        >>> xtcs = ["tests/data/1am7_corrected_part1.xtc", "tests/data/1am7_corrected_part2.xtc"]
        >>> tops = ["tests/data/1am7_protein.pdb", "tests/data/1am7_protein.pdb"]
        >>> trajs = em.load(xtcs, tops)
        >>> print(trajs)
        encodermap.TrajEnsemble object. Current backend is no_load. Containing 2 trajs. Not containing any CVs.
        >>> # load CVs
        >>> # This step can be omitted. The AngleDihedralCartesianEncoderMap class automatically loads CVs
        >>> trajs.load_CVs('all')
        >>> print(trajs.CVs['central_cartesians'].shape)
        (51, 474, 3)
        >>> print(trajs.CVs['central_dihedrals'].shape)
        (51, 471)
        >>> # create some parameters
        >>> p = em.ADCParameters(periodicity=360, use_backbone_angles=True, use_sidechains=True,
        ...                      cartesian_cost_scale_soft_start=(6, 12))
        >>> # Standard is functional model, as it offers more flexibility
        >>> print(p.model_api)
        functional
        >>> print(p.distance_cost_scale)
        None
        >>> # Instantiate the class
        >>> e_map = em.AngleDihedralCartesianEncoderMap(trajs, p, read_only=True)
        >>> # dataset contains these inputs:
        >>> # central_angles, central_dihedrals, central_cartesians, central_distances, sidechain_dihedrals
        >>> print(e_map.dataset)
        <BatchDataset element_spec=(TensorSpec(shape=(None, 472), dtype=tf.float32, name=None), TensorSpec(shape=(None, 471), dtype=tf.float32, name=None), TensorSpec(shape=(None, 474, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 473), dtype=tf.float32, name=None), TensorSpec(shape=(None, 316), dtype=tf.float32, name=None))>
        >>> # output from the model contains the following data:
        >>> # out_angles, out_dihedrals, back_cartesians, pairwise_distances of inp cartesians, pairwise of back-mapped cartesians, out_side_dihedrals
        >>> for data in e_map.dataset.take(1):
        ...     pass
        >>> out = e_map.model(data)
        >>> print([i.shape for i in out])
        [TensorShape([256, 472]), TensorShape([256, 471]), TensorShape([256, 474, 3]), TensorShape([256, 112101]), TensorShape([256, 112101]), TensorShape([256, 316])]
        >>> # get output of latent space by providing central_angles, central_dihedrals, sidehcain_dihedrals
        >>> latent = e_map.encoder([data[0], data[1], data[-1]])
        >>> print(latent.shape)
        (256, 2)
        >>> # Rebuild central_angles, central_dihedrals and sidechain_angles from latent
        >>> dih, ang, side_dih = e_map.decode(latent)
        >>> print(dih.shape, ang.shape, side_dih.shape)
        (256, 472) (256, 471) (256, 316)

    """

    def __init__(
        self,
        trajs: encodermap.TrajEnsemble,
        parameters: Optional[encodermap.ADCParameters] = None,
        model: Optional[tensorflow.keras.Model] = None,
        read_only: bool = False,
        cartesian_loss_step: int = 0,
        top: Optional[mdtraj.Topology] = None,
    ) -> None:
        """Instantiate the `AngleDihedralCartesianEncoderMap` class.

        Args:
            trajs (em.TrajEnsemble): The trajectories to be used as input. If trajs contain no CVs, correct CVs will be loaded.
            parameters (Optional[em.ACDParameters]): The parameters for the current run. Can be set to None and the
                default parameters will be used. Defaults to None.
            model (Optional[tf.keras.models.Model]): The keras model to use. You can provide your own model
                with this argument. If set to None, the model will be built to the specifications of parameters using
                either the functional or sequential API. Defaults to None
            read_only (bool): Whether to write anything to disk (False) or not (True). Defaults to False.
            cartesian_loss_step (int, optional): For loading and re-training the model. The cartesian_distance_loss
                is tuned in step-wise. For this the start step of the training needs to be accounted for. If the
                scale of the cartesian loss should increase from epoch 6 to epoch 12 and the model is saved at
                epoch 9, this argument should also be set to 9, to continue training with the correct scaling
                factor. Defaults to 0.

        """
        # parameters
        if parameters is None:
            self.p = ADCParameters()
        else:
            self.p = parameters

        # seed
        if self.p.seed is not None:
            tf.random.set_seed(self.p.seed)

        # read_only
        self.read_only = read_only

        # will be saved and overwritten when loading.
        self.cartesian_loss_step = cartesian_loss_step

        # save params and create dir
        if not self.read_only:
            self.p.save()
            print(
                "Output files are saved to {}".format(self.p.main_path),
                "as defined in 'main_path' in the parameters.",
            )

        # check whether Tensorboard and Read-Only makes Sense
        if self.read_only and self.p.tensorboard:
            raise BadError(
                "Setting tensorboard and read_only True is not possible. Tensorboard will always write to disk."
                " If you received this Error while loading a trained model, pass read_only=False as an argument"
                f" or set overwrite_tensorboard_bool True to overwrite the tensorboard parameter."
            )

        # clear old sessions
        tf.keras.backend.clear_session()

        # get the CVs:
        if isinstance(trajs, str):
            self.trajs = TrajEnsemble([trajs], [top])
        else:
            self.trajs = trajs

        # load missing values
        should_be = set(
            [
                "central_angles",
                "central_cartesians",
                "central_dihedrals",
                "central_distances",
                "side_dihedrals",
            ]
        )

        if self.trajs.CVs_in_file:
            raise NotImplementedError(
                "Write a tf.data.Dataset.from_generator function in enocdermap.data using the data from the netCDF files"
            )
        elif self.trajs.CVs:
            missing = list(should_be - set(trajs.CVs.keys()))
            if missing != []:
                print("loading missing values: ", missing)
                self.trajs.load_CVs(missing, ensemble=False)
        else:
            self.trajs.load_CVs(list(should_be), ensemble=False)

        if not should_be - set(self.trajs.CVs.keys()) == set():
            raise BadError(
                f"Could not load CVs. Should be {should_be}, but currenlty only {set(trajs.CVs.keys())} are loaded"
            )

        # define inputs
        self.sparse, self.train_data, self.inp_CV_data = self.get_train_data_from_trajs(
            self.trajs, self.p
        )

        # create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.inp_CV_data["central_angles"],
                self.inp_CV_data["central_dihedrals"],
                self.inp_CV_data["central_cartesians"],
                self.inp_CV_data["central_distances"],
                self.inp_CV_data["side_dihedrals"],
            )
        )
        dataset = dataset.shuffle(
            buffer_size=self.inp_CV_data["central_cartesians"].shape[0],
            reshuffle_each_iteration=True,
        )
        dataset = dataset.repeat()
        self.dataset = dataset.batch(self.p.batch_size)

        # ToDo: Make training faster with Autotune, XLA (jit) compilation, DataRecords
        # self.dataset = self.dataset.prefetch(self.p.batch_size * 4)
        # self.dataset = self.dataset.interleave(num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # num_batches = 20
        # num_threads = 8
        # ds = tf.data.Dataset.from_tensors(self.train_data).repeat(self.p.batch_size * num_batches)
        # ds = ds.batch(self.p.batch_size)
        # self.dataset = ds.interleave(lambda *args:tf.data.Dataset.from_tensor_slices(args), num_threads, 1, num_threads)

        # create model based on user input
        if model is None:
            self.model = self.p.model_api
        else:
            self._model = model

        # setup callbacks
        self._setup_callbacks()

        # create loss based on user input
        self.loss = self.p.loss

        # choose optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)

        # compile model
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

        # do this if tensorboard is true.
        if self.p.tensorboard:
            # print shapes
            print("input shapes are:")
            print({k: v.shape for k, v in self.inp_CV_data.items()})
            # set _log_images to False to fix the backend after training
            self._log_images = False
            # get the output from model summary.
            with Capturing() as output:
                self.model.summary()
            with open(self.p.main_path + "/model_summary.txt", "w") as f:
                f.write("\n".join(output))
            try:
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=self.p.main_path + "/model_summary.png",
                    show_shapes=True,
                    rankdir="TB",
                    expand_nested=True,
                )
            except Exception as e:
                print(f"saving image gave error: {e}")
            # todo: add image of cat
            # from ..parameters import parameters as _p
            # cat_image = os.path.split(os.path.split(os.path.split(_p.__file__)[0])[0])[0] + '/pic/priscilla-du-preez-8NXmaXg5xL0-unsplash.jpg'
            # image = plt.imread(cat_image)
            # plt.imshow(image)
            # print(cat_image)
            print(
                f"Saved a text-summary of the model and an image in {self.p.main_path},",
                "as specified in 'main_path' in the parameters.",
            )

            # sets up the tb callback to plot the model
            self.tb_callback = tf.keras.callbacks.TensorBoard(
                self.p.main_path, write_graph=True
            )
            self.tb_callback.set_model(self.model)

    @staticmethod
    def get_train_data_from_trajs(trajs, p, attr="CVs"):
        if not any([np.isnan(x).any() for x in getattr(trajs, attr).values()]):
            inp_CV_data = {
                key: val.astype("float32") for key, val in getattr(trajs, attr).items()
            }
            sparse = False
        else:
            sparse = True
            print("Input contains nans. Using sparse network.")
            inp_CV_data = {
                key: val.astype("float32") for key, val in getattr(trajs, attr).items()
            }

            # squeeze, if xarray is provided
            if all([hasattr(v, "values") for v in inp_CV_data.values()]):
                inp_CV_data = {k: v.values.squeeze() for k, v in inp_CV_data.items()}

            for k, v in inp_CV_data.items():
                if np.any(np.isnan(v)):
                    values = v
                    if k == "central_cartesians":
                        values = values.reshape(len(values), -1)
                    indices = np.stack(np.where(~np.isnan(values))).T.astype("int64")
                    dense_shape = values.shape
                    values = values[~np.isnan(values)].flatten()
                    sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
                    inp_CV_data[k] = sparse_tensor

        if not p.use_backbone_angles and not p.use_sidechains:
            train_data = inp_CV_data["central_dihedrals"]
        elif p.use_backbone_angles and not p.use_sidechains:
            train_data = [
                inp_CV_data["central_angles"],
                inp_CV_data["central_dihedrals"],
            ]
            if p.model_api == "sequential" and not sparse:
                train_data = np.hstack(train_data)
        elif p.use_backbone_angles and p.use_sidechains:
            train_data = [
                inp_CV_data["central_angles"],
                inp_CV_data["central_dihedrals"],
                inp_CV_data["side_dihedrals"],
            ]
            if p.model_api == "sequential" and not sparse:
                train_data = np.hstack(train_data)
        else:
            raise Exception(
                "Cannot train model with central dihedrals and side dihedrals only. Backbone angles are required."
            )

        # some checks for the length of the train data
        if p.model_api == "functional":
            if not p.use_backbone_angles and not p.use_sidechains:
                assert isinstance(train_data, tf.sparse.SparseTensor)
            elif p.use_backbone_angles and not p.use_sidechains:
                assert len(train_data) == 2
            else:
                assert len(train_data) == 3

        return sparse, train_data, inp_CV_data

    @classmethod
    def from_checkpoint(
        cls, trajs, checkpoint_path, read_only=True, overwrite_tensorboard_bool=False
    ):
        """Reconstructs the model from a checkpoint."""
        # Is this classmethod necessary? We need to make sure the class knows all losses.
        # And I don't know if the parent class calls the correct loss.setter
        return load_model(
            cls, checkpoint_path, read_only, overwrite_tensorboard_bool, trajs
        )

    def _setup_callbacks(self) -> None:
        """Overwrites the parent class' `_setup_callbacks` method.

        Due to the 'soft start' of the cartesian cost, the `cartesiand_increase_callback`
        needs to be added to the list of callbacks.

        """
        super(self.__class__, self)._setup_callbacks()
        if self.p.cartesian_cost_scale_soft_start != (None, None):
            self.cartesian_increase_callback = IncreaseCartesianCost(
                self.p, start_step=self.cartesian_loss_step
            )
            self.callbacks.append(self.cartesian_increase_callback)

    def save(self, step: Optional[int] = None) -> None:
        """Saves the model to the current path defined in `parameters.main_path`.

        Args:
            step (Optional[int]): Does not actually save the model at the given training step, but rather
                changes the string used for saving the model from an datetime format to another.

        """
        if not self.read_only:
            save_model(
                self.model,
                self.p.main_path,
                self.__class__.__name__,
                step=step,
                current_step=self.cartesian_loss_step,
            )

    @Autoencoder.loss.setter
    def loss(self, loss):
        if loss == "reconstruction_loss":
            self._loss = reconstruction_loss(self.model)
        elif loss == "emap_cost":
            self.dihedral_loss = dihedral_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.angle_loss = angle_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            if self.p.cartesian_cost_scale_soft_start != (None, None):
                self.cartesian_loss = cartesian_loss(
                    self.model,
                    self.cartesian_increase_callback,
                    self.p,
                    self.tensorboard_write_bool,
                )
            else:
                self.cartesian_loss = cartesian_loss(
                    self.model, None, self.p, self.tensorboard_write_bool
                )

            self.distance_loss = distance_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.cartesian_distance_loss = cartesian_distance_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.center_loss = center_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.regularization_loss = regularization_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self._loss = [
                self.dihedral_loss,
                self.angle_loss,
                self.cartesian_loss,
                self.distance_loss,
                self.cartesian_distance_loss,
                self.center_loss,
                self.regularization_loss,
            ]
            if self.p.use_sidechains:
                self.side_dihedral_loss = side_dihedral_loss(
                    self.model, self.p, self.tensorboard_write_bool
                )
                self._loss.append(self.side_dihedral_loss)
        elif loss == "mse":
            self._loss = "mse"
        else:
            raise ValueError(
                f"loss argument needs to be `reconstruction_loss`, `mse` or `emap_cost`. You provided '{loss}'."
            )

    def train(self) -> None:
        """Overrides the parent class' `train` method.

        After the training is finished, an additional file is written to disk,
        which saves the current epoch. In the event that training will continue,
        the current state of the soft-start cartesian cost is read from that file.

        """
        super(self.__class__, self).train()
        self.cartesian_loss_step += self.p.n_steps
        fname = f"{self.p.main_path}/saved_model_{self.p.n_steps}.model"
        with open(fname + "_current_step.txt", "w") as f:
            f.write(str(self.cartesian_loss_step))

    def encode(self, data=None):
        if hasattr(data, "_traj_file"):
            _, data, __ = self.get_train_data_from_trajs(data, self.p, attr="_CVs")
        elif hasattr(data, "traj_files"):
            _, data, __ = self.get_train_data_from_trajs(data, self.p)
        return super().encode(data)

    def generate(
        self,
        points: np.ndarray,
        top: Optional[str, int, mdtraj.Topology] = None,
        backend: Literal["mdtraj", "mdanalysis"] = "mdtraj",
    ) -> Union[MDAnalysis.Universe, mdtraj.Trajectory]:
        """Overrides the parent class' `generate` method and builds a trajectory.

        Instead of just providing data to `decode` using the decoder part of the
        network, this method also takes a molecular topology as its `top`
        argument. This topology is then used to rebuild a time-resolved
        trajectory.

        Args:
            points (np.ndarray): The low-dimensional points from which the
                trajectory should be rebuilt.
            top (Optional[str, int, mdtraj.Topology]): The topology to be used for rebuilding the
                trajectory. This should be a string pointing towards a <*.pdb,
                *.gro, *.h5> file. Alternatively, None can be provided, in which
                case, the internal topology (`self.top`) of this class is used.
                Defaults to None.
            backend (str): Defines what MD python package to use, to build the
                trajectory and also what type this method returns, needs to be
                one of the following:
                * "mdtraj"
                * "mdanalysis"

        Returns:
            Union[mdtraj.Trajectory, MDAnalysis.universe]: The trajectory after
                applying the decoded structural information. The type of this
                depends on the chosen `backend` parameter.

        """
        # get the output this can be done regardless
        out = self.decode(points)

        if top is None:
            top = self.trajs.top_files
            if len(top) > 1:
                print(
                    f"Please specify which topology you would like to use for generating "
                    f"conformations. You can either provide a `str` to a topology file "
                    f"(file extension .pdb, .h5, .gro) on disk, or a `int` specifying the "
                    f"`SingleTraj` object in this class' {self.trajs.n_trajs} trajs, or "
                    f"you can also specify a `mdtraj.Topology` object."
                )
                return
            else:
                top = top[0]
                trajs = self.trajs
                if top not in self.trajs.top_files:
                    raise Exception(
                        "Provided topology was not used to train Encodermap."
                    )

                # get the output
                if not self.p.use_backbone_angles and not self.p.use_sidechains:
                    dihedrals = self.decode(points)
                elif self.p.use_backbone_angles and not self.p.use_sidechains:
                    splits = [trajs.CVs["central_angles"].shape[1]]
                    out = self.decode(points)
                    if isinstance(out, np.ndarray):
                        angles, dihedrals = np.split(out, splits, axis=1)
                elif self.p.use_backbone_angles and self.p.use_sidechains:
                    splits = [
                        trajs.CVs["central_angles"].shape[1],
                        trajs.CVs["central_angles"].shape[1]
                        + trajs.CVs["central_dihedrals"].shape[1],
                    ]

                    if isinstance(out, np.ndarray):
                        angles, dihedrals, sidechain_dihedrals = np.array_split(
                            out, splits, axis=1
                        )
                    else:
                        angles, dihedrals, sidechain_dihedrals = out

            # in this case we can just use any traj from self.trajs
            traj = self.trajs

        else:
            if len(self.trajs.top_files) == 1:
                trajs = self.trajs
                if top not in self.trajs.top_files:
                    raise Exception(
                        "Provided topology was not used to train Encodermap."
                    )
            else:
                if isinstance(top, str):
                    pass
                elif isinstance(top, int):
                    top_ = self.trajs[top].traj[0]
                    top_.save_pdb("/tmp/tmp.pdb")
                    top = "/tmp/tmp.pdb"
                elif isinstance(top, mdtraj.Topology):
                    top.save_pdb("/tmp/tmp.pdb")
                    top = "/tmp/tmp.pdb"
                else:
                    raise TypeError(
                        f"Provided type for `top` must be `str`, `int`, or `mdtraj.Topology`, "
                        f"you provided {type(top)}."
                    )

                # align the topology with the trajs in self.trajs
                from ..loading import features
                from ..loading.featurizer import UNDERSOCRE_MAPPING

                UNDERSOCRE_MAPPING = {v: k for k, v in UNDERSOCRE_MAPPING.items()}
                labels = {}
                feature_names = [
                    "CentralCartesians",
                    "CentralBondDistances",
                    "CentralAngles",
                    "CentralDihedrals",
                    "SideChainDihedrals",
                ]

                for feature in feature_names:
                    feature = getattr(features, feature)(top_.top, generic_labels=True)
                    labels[UNDERSOCRE_MAPPING[feature.name]] = feature.describe()

                return_values = [
                    "central_dihedrals",
                    "central_angles",
                    "side_dihedrals",
                ]
                splits = {}
                for i, k in enumerate(return_values):
                    split = np.isin(
                        self.trajs[0]._CVs.coords[k.upper()].values, labels[k]
                    )
                    splits[k] = split

                # split the output
                if not self.p.use_backbone_angles and not self.p.use_sidechains:
                    dihedrals = out[:, splits["central_dihedrals"]]
                elif self.p.use_backbone_angles and not self.p.use_sidechains:
                    dihedrals = out[1][:, splits["central_dihedrals"]]
                    angles = out[2][:, splits["central_angles"]]
                elif self.p.use_backbone_angles and self.p.use_sidechains:
                    dihedrals = out[1][:, splits["central_dihedrals"]]
                    angles = out[0][:, splits["central_angles"]]
                    sidechain_dihedrals = out[2][:, splits["side_dihedrals"]]

                # if the backend is mdanalysis we need to save the pdb
                if backend == "mdanalysis":
                    top_.save_pdb("/tmp/tmp.pdb")
                    top = "/tmp/tmp.pdb"
                else:
                    # in this case we need to use a traj, which topolgy matches top
                    for i, traj in self.trajs.itertrajs():
                        if traj.top == top_.top:
                            break
                    else:
                        raise Exception(
                            "Could not find a trajectory in self.trajs, "
                            "that matches the topology provided as `top`."
                        )
                    traj = deepcopy(traj)

        # do the backmapping
        if backend == "mdanalysis":
            uni = dihedral_backmapping(top, dihedrals, sidechains=sidechain_dihedrals)
            return uni
        elif backend == "mdtraj":
            traj = mdtraj_backmapping(top, dihedrals, sidechain_dihedrals, traj)
            return traj
        else:
            raise TypeError(
                f"backend must be 'mdtraj' or 'mdanalysis', but you provided {backend}"
            )
