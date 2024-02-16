# -*- coding: utf-8 -*-
# encodermap/autoencoder/autoencoder.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
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


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

# Third Party Imports
import matplotlib
import numpy as np
import tensorflow as tf
from optional_imports import _optional_import
from tqdm import tqdm

# Local Folder Imports
from ..callbacks import ADCClashMetric, ADCRMSDMetric
from ..callbacks.callbacks import (
    CheckpointSaver,
    ImageCallback,
    IncreaseCartesianCost,
    ProgressBar,
    TensorboardWriteBool,
)
from ..encodermap_tf1.backmapping import chain_in_plane, dihedrals_to_cartesian_tf
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
from ..misc.distances import pairwise_dist
from ..misc.misc import create_n_cube, plot_model
from ..misc.saving_loading_models import load_model, save_model
from ..models.models import gen_functional_model, gen_sequential_model
from ..parameters.parameters import ADCParameters, Parameters
from ..trajinfo.info_single import Capturing, TrajEnsemble


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
mda = _optional_import("MDAnalysis")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)


AutoencoderType = TypeVar("AutoencoderType", bound="Parent")
EncoderMapType = TypeVar("EncoderMapType", bound="Parent")
DihedralEncoderMapType = TypeVar("DihedralEncoderMapType", bound="Parent")
AngleDihedralCartesianEncoderMapType = TypeVar(
    "AngleDihedralCartesianEncoderMapType", bound="Parent"
)


if TYPE_CHECKING:
    # Third Party Imports
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


    To understand the necessity of this function, we need to have a look how
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
    does not make sense here. That's why EncoderMap's `function` decorator
    takes an additional argument:

    """

    def wrapper(*args, **kwargs):
        tensorboard = kwargs.pop("tensorboard", False)
        """Wrapper of `encodermap.function`."""
        if tensorboard:
            warnings.warn(
                "Running in tensorboard mode writes a lot of stuff to "
                "tensorboard. For speed boost deactivate tensorboard mode."
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
    """Main Autoencoder class. Presents all high-level functions.

    This is the main class for neural networks inside EncoderMap. The class
    prepares the data (batching and shuffling), creates a `tf.keras.Model`
    of layers specified by the attributes of the `encodermap.Parameters` class.
    Depending on what Parent/Child-Class is instantiated, a combination of
    various cost functions is set up. Callbacks to Tensorboard are also set up.

    Attributes:
        train_data (np.ndarray): The numpy array of the train data passed at init.
        p (AnyParameters): An `encodermap.Parameters` class
            containing all info needed to set up the network.
        dataset (tensorflow.data.Dataset): The dataset that is actually used
            in training the keras model. The dataset is a batched, shuffled,
            infinitely-repeating dataset.
        read_only (bool): Variable telling the class whether it is allowed to
            write to disk (False) or not (True).
        metrics (list[Any]): A list of metrics passed to the model when it is compiled.
        callbacks (list[Any]): A list of tf.keras.callbacks.Callback subclasses
            changing the behavior of the model during training.
            Some standard callbacks are always present like:
                * encodermap.callbacks.callbacks.ProgressBar:
                    A progress bar callback using tqdm giving the current
                    progress of training and the current loss.
                * CheckPointSaver:
                    A callback that saves the model every
                    `parameters.checkpoint_step` steps into the main directory.
                    This callback will only be used, when `read_only` is False.
                * TensorboardWriteBool:
                    A callback that contains a boolean Tensor that will be
                    True or False, depending on the current training step and
                    the summary_step in the parameters class. The loss functions
                    use this callback to decide whether they should write to
                    Tensorboard. This callback will only be present when
                    `read_only` is False and `parameters.tensorboard` is True.
            You can append your own callbacks to this list before executing
            `self.train()`.
        encoder (tf.keras.Model): The encoder (sub)model of `self.model`.
        decoder (tf.keras.Model): The decoder (sub)model of `self.model`.

    Methods:
        from_checkpoint: Rebuild the model from a checkpoint.
        add_images_to_tensorboard: Make tensorboard plot images.
        train: Starts the training of the tf.keras.models.Model.
        plot_network: Tries to plot the network. For this method to work
            graphviz, pydot and pydotplus need to be installed.
        encode: Takes high-dimensional data and sends it through the encoder.
        decode: Takes low-dimensional data and sends it through the encoder.
        generate: Same as `decode`. For AngleDihedralCartesianAutoencoder classes,
            this will build a protein strutcure.

    Note:
        Performance of tensorflow is not only dependent on your system's
        hardware and how the data is presented to the network
        (for this check out https://www.tensorflow.org/guide/data_performance),
        but also how you compiled tensorflow. Normal tensorflow
        (pip install tensorflow) is build without CPU extensions to work on
        many CPUs. However, Tensorflow can greatly benefit from using CPU
        instructions like AVX2, AVX512 that bring a speed-up in linear algebra
        computations of 300%. By building tensorflow from source,
        you can activate these extensions. However, the speed-up of using
        tensorflow with a GPU dwarfs the CPU speed-up. To check whether a
        GPU is available run: `print(len(tf.config.list_physical_devices('GPU')))`.
        Refer to these pages to install tensorflow for the best performance:
        https://www.tensorflow.org/install/pip and
        https://www.tensorflow.org/install/gpu

    Examples:
        >>> import encodermap as em
        >>> # without providing any data, default parameters and a 4D
        >>> # hypercube as input data will be used.
        >>> e_map = em.EncoderMap(read_only=True)
        >>> print(e_map.train_data.shape)
        (16000, 4)
        >>> print(e_map.dataset)  # doctest: +SKIP
        <BatchDataset element_spec=(TensorSpec(shape=(None, 4), dtype=tf.float32, name=None), TensorSpec(shape=(None, 4), dtype=tf.float32, name=None))>
        >>> print(e_map.encode(e_map.train_data).shape)
        (16000, 2)

    """

    def __init__(
        self,
        parameters=None,
        train_data: Optional[Union[np.ndarray, tf.data.Dataset]] = None,
        model: Optional[tf.keras.Model] = None,
        read_only: bool = False,
        sparse: bool = False,
    ) -> None:
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
            self.p.write_summary = True
            self.p.save()
            print(
                "Output files are saved to {}".format(self.p.main_path),
                "as defined in 'main_path' in the parameters.",
            )

        # check whether Tensorboard and Read-Only makes Sense
        if self.read_only and self.p.tensorboard:
            raise NotImplementedError

        # clear old sessions
        tf.keras.backend.clear_session()
        self.sparse = sparse

        # set up train_data
        self.set_train_data(train_data)

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
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        # do this if tensorboard is true.
        if self.p.tensorboard:
            self._log_images = False
            # get the output from model summary.
            with Capturing() as output:
                self.model.summary()
            with open(self.p.main_path + "/model_summary.txt", "w") as f:
                f.write("\n".join(output))
            self.plot_network()
            print(
                f"Saved a text-summary of the model and an image in {self.p.main_path},",
                "as specified in 'main_path' in the parameters.",
            )

            # sets up the tb callback to plot the model
            self.tb_callback = tf.keras.callbacks.TensorBoard(
                self.p.main_path, write_graph=True
            )
            self.tb_callback.set_model(self.model)

    def set_train_data(self, data: Union[np.ndarray, tf.data.Dataset]) -> None:
        """Resets the train data for reloaded models."""
        self._using_hypercube = False
        if data is None:
            self._using_hypercube = True
            self.train_data = create_n_cube(4, seed=self.p.seed)[0].astype("float32")
            self.p.periodicity = float("inf")
        elif isinstance(data, np.ndarray):
            if np.any(np.isnan(data)):
                self.sparse = True
                print("Input contains nans. Using sparse network.")
                indices = np.stack(np.where(~np.isnan(data))).T.astype("int64")
                dense_shape = data.shape
                values = data[~np.isnan(data)].flatten().astype("float32")
                sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
                self.train_data = sparse_tensor
            else:
                self.train_data = data.astype("float32")
        elif isinstance(data, tf.data.Dataset):
            self.dataset = data
            try:
                _, __ = self.dataset.take(1)
            except ValueError:
                if self.p.training == "auto":
                    print(
                        f"It seems like your dataset only yields tensors and not "
                        f"tuples of tensors. TensorFlow is optimized for classification "
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
                    _ = self.dataset.take(1)
            self.train_data = _
        else:
            raise TypeError(
                f"train_data must be `None`, `np.ndarray` or `tf.data.Dataset`. You supplied {type(data)}."
            )

        # check data and periodicity
        if not self.sparse and data is not None:
            if isinstance(data, np.ndarray):
                if np.any(data > self.p.periodicity):
                    raise Exception(
                        "There seems to be an error regarding the periodicity "
                        f"of your data. The chosen periodicity is {self.p.periodicity}, "
                        f"but there are datapoints outwards of this range: {data.max()}"
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

    def _setup_callbacks(self) -> None:
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
        else:
            self.tensorboard_write_bool = None

    @classmethod
    def from_checkpoint(
        cls: Type[AutoencoderType],
        checkpoint_path: Union[str, Path],
        train_data: Optional[np.ndarray] = None,
        sparse: bool = False,
        use_previous_model: bool = False,
        compat: bool = False,
    ) -> AutoencoderType:
        """Reconstructs the class from a checkpoint.

        Args:
            checkpoint_path (Union[str, Path]): The path to the checkpoint. Can
                be either a directory, in which case the most recently saved
                model will be loaded. Or a direct .keras file, in which case, this
                specific model will be loaded.
            train_data (Optional[np.ndarray]). When you want to retrain this model, you
                can provide the train data here.
            sparse (bool): Whether the reloaded model should be sparse.
            use_previous_model (bool): Set this flag to True, if you load a model
                from an in-between checkpoint step (e.g., to continue training with
                different parameters). If you have the files saved_model_0.keras,
                saved_model_500.keras and saved_model_1000.keras, setting this to
                True and loading the saved_model_500.keras will back up the
                saved_model_1000.keras.
            compat (bool): Whether to use compatibility mode when missing or wrong
                parameter files are present. In this special case, some assumptions
                about the network architecture are made from the model and the
                parameters in parameters.json overwritten accordingly (a backup
                will also be made).

        Returns:
            Autoencoder: Encodermap `Autoencoder` class.

        """
        return load_model(
            cls,
            checkpoint_path,
            sparse=sparse,
            dataset=train_data,
            use_previous_model=use_previous_model,
            compat=compat,
        )

    @property
    def model(self) -> tf.keras.Model:
        """tf.keras.Model: The tf.keras.Model model used for training."""
        return self._model

    @model.setter
    def model(self, model: str):
        """sets self.model according to `model_api` argument in self.parameters."""
        if model == "functional":
            d = self.dataset.take(1)
            if any([isinstance(_, tf.SparseTensorSpec) for _ in d.element_spec]):
                self.sparse = True
            self._model = gen_functional_model(
                self.dataset,
                self.p,
                sparse=self.sparse,
            )
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
                f"API argument needs to be one of `functional`, `sequential`, "
                f"`custom`. You provided '{model}'."
            )

    @property
    def encoder(self) -> tf.keras.Model:
        """tf.keras.Model: Encoder part of the model."""
        return self._model.encoder

    @property
    def decoder(self) -> tf.keras.Model:
        """tf.keras.Model: Decoder part of the model."""
        return self._model.decoder

    @property
    def loss(self) -> Sequence[Callable]:
        """(Sequence[Callable]): A list of loss functions passed to the model
        when it is compiled. When the main `Autoencoder` class is used and
        `parameters.loss` is 'emap_cost', this list comprises center_cost,
        regularization_cost, auto_cost. When the `EncoderMap` sub-class is
        used and `parameters.loss` is 'emap_cost', distance_cost is added to
        the list. When `parameters.loss` is not 'emap_cost', the loss can either
        be a string ('mse'), or a function, that both are acceptable
        arguments for loss, when a keras model is compiled.

        """
        return self._loss

    @loss.setter
    def loss(self, loss: str):
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

    def train(self) -> Optional[tf.keras.callbacks.History]:
        """Starts the training of the model.

        Returns:
            Union[tf.keras.callbacks.History, None]: If training succeeds, an
                instance of `tf.keras.callbacks.History` is returned. If not,
                None is returned.

        """
        if self.p.current_training_step >= self.p.n_steps:
            print(
                f"This {self.__class__.__name__} instance has already been trained "
                f"for {self.p.current_training_step} steps. Increase the training "
                f"steps by calling `{self.__class__.__name__}.p.n_steps += new_steps` "
                f"and then call `{self.__class__.__name__}.train()` again."
            )
            return

        if self._using_hypercube and self.train_data.shape[1] != self.model.input_shape:
            print(
                "This reloaded model was not yet provided with train data. Please "
                "use the `set_train_data()` method to provide new train data and "
                "continue training. You could also provide the training data when "
                f"reloading the model by calling `{self.__class__.__name__}.from"
                "_checkpoint()` constructor with the `train_data` argument."
            )
            return

        if self.p.training == "custom" and self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "custom" and not self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "auto":
            if self.p.tensorboard and self._log_images:
                # get the old backend because the Tensorboard Images callback will set 'Agg'
                # and without re-setting the old backend the user won't get
                # output when calling fig.show() in a notebook.
                old_backend = matplotlib.get_backend()
            epochs = self.p.n_steps - self.p.current_training_step
            history = self.model.fit(
                self.dataset,
                batch_size=self.p.batch_size,
                epochs=epochs,
                steps_per_epoch=1,
                verbose=0,
                callbacks=self.callbacks,
            )
        else:
            raise ValueError(
                f"training argument needs to be `auto` or `custom`. You provided '{self.training}'."
            )
        self.p.current_training_step += self.p.n_steps - self.p.current_training_step
        self.p.save()
        self.save()

        # reset the backend.
        if self.p.tensorboard and self._log_images:
            matplotlib.use(old_backend)

        return history

    def add_images_to_tensorboard(
        self,
        data: Optional[Union[np.ndarray, list[float]]] = None,
        image_step: Optional[int] = None,
        max_size: int = 10_000,
        scatter_kws: Optional[dict] = None,
        hist_kws: Optional[dict] = None,
        additional_fns: Optional[Sequence[Callable]] = None,
        when: Literal["epoch", "batch"] = "epoch",
    ) -> None:
        """Adds images to Tensorboard using the data in data and the ids in ids.

        Args:
            data (Optional[Union[np.ndarray, Sequence[np.ndarray]]): The input-data will
                be passed through the encoder part of the autoencoder. If None
                is provided, a set of 10_000 points from `self.train_data` will
                be taken. A list[np.ndarray] is needed for the functional API of the
                `AngleDihedralCartesianEncoderMap`, that takes a list of
                [angles, dihedrals, side_dihedrals]. Defaults to None.
            image_step (Optional[int]): The interval in which to plot
                images to tensorboard. If None is provided, the `image_step`
                will be the same as `Parameters.summary_step`. Defaults to None.
            max_size (int): The maximum size of the high-dimensional data, that is
                projected. Prevents excessively large-datasets from being projected
                at every `image_step`. Defaults to 10_000.
            scatter_kws (Optional[dict]): A dict with items that
                `matplotlib.pyplot.scatter()` will accept. If None is provided,
                a dict with size 20 will be passed to `plt.scatter(**{'s': 20})`,
                which sets an appropriate size of scatter points for the size of
                datasets encodermap is usually used for.
            hist_kws ( Optional[dict]): A dict with items that
                `matplotlib.pyplot.scatter()` will accept. If None is provided a
                dict with bins 50 will be passed to `plt.hist2D(**{'bins': 50})`.
                You can choose a colormap here by providing `{'bins": 50, 'cmap':
                'plasma'}` for this argument.
            additional_fns (Optional[Sequence[Callable]]): A list of functions
                that will accept the low-dimensional output of the `Autoencoder`
                latent/bottleneck layer and return a tf.Tensor that can be logged
                by `tf.summary.image()`. See the notebook
                'writing_custom_images_to_tensorboard.ipynb' in
                tutorials/notebooks_customization for more info. If None is
                provided, no additional functions will be used to plot to
                tensorboard. Defaults to None.
            when (Literal["epoch", "batch"]): When to log the images can be
                either 'batch', then the images will be logged after every step
                during training, or 'epoch', then only after every image_step
                epoch the images will be written. Defaults to 'epoch'.

        """
        if not self.p.tensorboard:
            print(
                "Nothing is written to Tensorboard for this Model. "
                "Please change parameters.tensorboard to True."
            )
            return
        if image_step is None:
            image_step = self.p.summary_step

        if scatter_kws is None:
            scatter_kws = {"s": 20}

        if hist_kws is None:
            hist_kws = {"bins": 50}

        self._log_images = True

        # make a dataset for images
        if data is None:
            if hasattr(self, "train_data"):
                if isinstance(self.train_data, np.ndarray):
                    data = self.train_data
                elif isinstance(self.train_data, list) or self.sparse:
                    data = self.train_data
                else:
                    data = list(self.dataset.take(int(10000 / self.p.batch_size)))
                    data = np.stack(data)[:, 0, :].reshape(-1, self.train_data.shape[1])
            else:
                if hasattr(self, "trajs"):
                    _data = (
                        self.trajs._CVs.stack({"frame": ("traj_num", "frame_num")})
                        .transpose("frame", ...)
                        .dropna("frame", how="all")
                    )
                    _data_size = _data.dims["frame"]
                    idx = np.unique(
                        np.round(np.linspace(0, _data_size - 1, max_size)).astype(int)
                    )
                    data = [_data.central_dihedrals.values[idx]]
                    if self.p.use_backbone_angles:
                        data.insert(0, _data.central_angles.values[idx])
                    if self.p.use_sidechains:
                        data.append(_data.side_dihedrals.values[idx])
                    if len(data) == 1:
                        data = data[0]
                else:
                    raise Exception(f"Please provide `data` to plot.")
        else:
            if hasattr(self, "train_data"):
                if type(data) != type(self.train_data):
                    raise Exception(
                        f"Provided data has wrong type. Train data in this class is "
                        f"{type(self.train_data)}, provided data is {type(data)}"
                    )

        # select a subset
        if isinstance(data, (tuple, list)):
            _data = []
            _lengths = set()
            for i, d in enumerate(data):
                if i == 0:
                    idx = np.unique(
                        np.round(np.linspace(0, len(d) - 1, max_size)).astype(int)
                    )
                assert isinstance(d, np.ndarray)
                _lengths.add(len(data))
                assert len(_lengths) == 1, (
                    f"The tuple of numpy arrays you provided as data has uneven "
                    f"lengths: {_lengths}."
                )
                _data.append(d[idx])
            else:
                data = tuple(_data)
        elif isinstance(data, np.ndarray):
            idx = np.unique(
                np.round(np.linspace(0, len(data) - 1, max_size)).astype(int)
            )
            data = data[idx]
        else:
            raise ValueError(
                f"Argument `data` must be np.ndarray of sequence thereof. You "
                f"provided: {type(data)=}."
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
                f"Logging images with {data.shape}-shaped data every "
                f"{image_step} epochs to Tensorboard at {self.p.main_path}"
            )
        else:
            print(
                f"Logging images with {[i.shape for i in data]}-shaped data "
                f"every {image_step} epochs to Tensorboard at {self.p.main_path}"
            )

    def plot_network(self) -> None:
        """Tries to plot the network using pydot, pydotplus and graphviz.
        Doesn't raise an exception if plotting is not possible.

        Note:
            Refer to this guide to install these programs:
            https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py

        """
        out = plot_model(self.model, self.train_data.shape[1])
        if out is not None:
            out.save(Path(self.p.main_path) / "network.png")

    @overload
    def encode(self, data: Optional[Sequence[np.ndarray]] = None) -> list[np.ndarray]:
        ...

    def encode(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """Calls encoder part of `self.model`.

        Args:
            data (Optional[np.ndarray]): The data to be passed top the encoder part.
                It can be either numpy ndarray or None. If None is provided,
                a set of 10000 points from the provided train data will be taken.
                Defaults to None.

        Returns:
            np.ndarray: The output from the bottleneck/latent layer.

        """
        if data is None:
            data = self.train_data

        # check the shapes:
        if not isinstance(data, (list, tuple)):
            if data.shape[1] * 2 == self.model.encoder_model.input_shape[1]:
                out = self.model.encoder(data)
            elif data.shape[1] == self.model.encoder_model.input_shape[1]:
                out = self.model.encoder_model(data)
            else:
                msg = (
                    f"The shape of the provided data {data.shape=} does not "
                    f"match the expected shape {self.model.encoder_model.input_shape=}."
                )
                if self.p.periodicity < float("inf"):
                    msg += f" Not even considering the periodicity of {self.p.periodicity}."
                raise Exception(msg)
        else:
            for d, in_shape in zip(data, self.model.encoder_model.input_shape):
                assert d.shape[1] == in_shape[1], (
                    f"The shape of the provided data ({d.shape}) does not match "
                    f"the expected shape {in_shape}."
                )
            out = self.model.encoder_model(data)

        if isinstance(out, (list, tuple)):
            out = [o.numpy() for o in out]
        else:
            out = out.numpy()
        return out

    def generate(self, data: np.ndarray) -> np.ndarray:
        """Duplication of `self.decode`.

        In `Autoencoder` and `EncoderMap` this method is equivalent to `decode()`.
        In `AngleDihedralCartesianEncoderMap` this method will be overwritten
        to produce output molecular conformations.

        Args:
            data (np.ndarray): The data to be passed to the decoder part of the
                model. Make sure that the shape of the data matches the number
                of neurons in the latent space.

        Returns:
            np.ndarray: Outputs from the decoder part. For
                `AngleDihedralCartesianEncoderMap`, this will either be a
                `mdtraj.Trajectory` or `MDAnalysis.Universe`.
        """
        return self.model.decoder(data)

    def decode(self, data: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        """Calls the decoder part of the model.

        `AngleDihedralCartesianAutoencoder` will, like the other two classes'
        output a list of np.ndarray.

        Args:
            data (np.ndarray):  The data to be passed to the decoder part of
                the model. Make sure that the shape of the data matches the
                number of neurons in the latent space.

        Returns:
            Union[list[np.ndarray], np.ndarray]: Outputs from the decoder part.
                For `AngleDihedralCartesianEncoderMap`, this will be a list of
                np.ndarray.

        """
        out = self.decoder(data)
        if isinstance(out, (list, tuple)):
            out = [o.numpy() for o in out]
        else:
            out = out.numpy()
        return out

    def save(self, step: Optional[int] = None) -> None:
        """Saves the model to the current path defined in `parameters.main_path`.

        Args:
            step (Optional[int]): Does not actually save the model at the given
                training step, but rather changes the string used for saving
                the model from a datetime format to another.

        """
        if not self.read_only:
            save_model(
                self.model,
                self.p.main_path,
                inp_class_name=self.__class__.__name__,
                step=step,
                print_message=True,
            )
        else:
            print(
                f"This {self.__class__.__name__} is set to read_only. Set "
                f"`{self.__class__.__name__}.read_only=False` to save the "
                f"current state of the model."
            )

    def close(self) -> None:
        """Clears the current keras backend and frees up resources."""
        # clear old sessions
        tf.keras.backend.clear_session()


class EncoderMap(Autoencoder):
    """Complete copy of Autoencoder class but uses additional distance cost
    scaled by the SketchMap sigmoid params"""

    @classmethod
    def from_checkpoint(
        cls: Type[EncoderMapType],
        checkpoint_path: Union[str, Path],
        train_data: Optional[np.ndarray] = None,
        sparse: bool = False,
        use_previous_model: bool = False,
        compat: bool = False,
    ) -> EncoderMapType:
        """Reconstructs the class from a checkpoint.

        Args:
            checkpoint_path (Union[str, Path]): The path to the checkpoint. Can
                be either a directory, in which case the most recently saved
                model will be loaded. Or a direct .keras file, in which case, this
                specific model will be loaded.
            train_data (Optional[np.ndarray]). When you want to retrain this model, you
                can provide the train data here.
            sparse (bool): Whether the reloaded model should be sparse.
            use_previous_model (bool): Set this flag to True, if you load a model
                from an in-between checkpoint step (e.g., to continue training with
                different parameters). If you have the files saved_model_0.keras,
                saved_model_500.keras and saved_model_1000.keras, setting this to
                True and loading the saved_model_500.keras will back up the
                saved_model_1000.keras.
            compat (bool): Whether to use compatibility mode when missing or wrong
                parameter files are present. In this special case, some assumptions
                about the network architecture are made from the model and the
                parameters in parameters.json overwritten accordingly (a backup
                will also be made).

        Returns:
            EncoderMap: EncoderMap `EncoderMap` class.

        """
        return load_model(
            cls,
            checkpoint_path,
            sparse=sparse,
            dataset=train_data,
            use_previous_model=use_previous_model,
        )

    @Autoencoder.loss.setter
    def loss(self, loss: str):
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

    def generate(
        self,
        data: np.ndarray,
        top: Union[Path, str],
    ) -> mda.Universe:
        """Overwrites `EncoderMap`'s generate method and actually does
        backmapping if a list of dihedrals is provided.

        Args:
            data (np.ndarray): The low-dimensional/latent/bottleneck data.
                A ndim==2 numpy array with xy coordinates of points in latent space.
            top (str): Topology file for this run of EncoderMap (can be .pdb, .gro, .. etc.).

        Returns:
            MDAnalysis.Universe: The topology with the provided backbone torsions.

        Examples:
            >>> # get some time-resolved pdb files
            >>> import requests
            >>> import numpy as np
            >>> pdb_link = 'https://files.rcsb.org/view/1YUF.pdb'
            >>> contents = requests.get(pdb_link).text
            >>> print(contents.splitlines()[0])  # doctest: +SKIP
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
            >>> print(ags[0][0])  # doctest: +SKIP
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
        >>> from pathlib import Path
        >>> # Load two trajectories
        >>> test_data = Path(em.__file__).parent.parent / "tests/data"
        >>> test_data.is_dir()
        True
        >>> xtcs = [test_data / "1am7_corrected_part1.xtc", test_data / "1am7_corrected_part2.xtc"]
        >>> tops = [test_data / "1am7_protein.pdb", test_data  /"1am7_protein.pdb"]
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
        >>> e_map = em.AngleDihedralCartesianEncoderMap(trajs, p, read_only=True)  # doctest: +ELLIPSIS
        Model...
        >>> # dataset contains these inputs:
        >>> # central_angles, central_dihedrals, central_cartesians, central_distances, sidechain_dihedrals
        >>> print(e_map.dataset)  # doctest: +SKIP
        <BatchDataset element_spec=(TensorSpec(shape=(None, 472), dtype=tf.float32, name=None), TensorSpec(shape=(None, 471), dtype=tf.float32, name=None), TensorSpec(shape=(None, 474, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 473), dtype=tf.float32, name=None), TensorSpec(shape=(None, 316), dtype=tf.float32, name=None))>
        >>> # output from the model contains the following data:
        >>> # out_angles, out_dihedrals, back_cartesians, pairwise_distances of inp cartesians, pairwise of back-mapped cartesians, out_side_dihedrals
        >>> for data in e_map.dataset.take(1):
        ...     pass
        >>> out = e_map.model(data)
        >>> print([i.shape for i in out])  # doctest: +SKIP
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
        trajs: Optional[TrajEnsemble] = None,
        parameters: Optional[ADCParameters] = None,
        model: Optional[tf.keras.Model] = None,
        read_only: bool = False,
        top: Optional[md.Topology] = None,
        dataset: Optional[tf.data.Dataset] = None,
    ) -> None:
        """Instantiate the `AngleDihedralCartesianEncoderMap` class.

        Args:
            trajs (em.TrajEnsemble): The trajectories to be used as input. If trajs contain no CVs, correct CVs will be loaded.
            parameters (Optional[em.ADCParameters]): The parameters for the current run. Can be set to None and the
                default parameters will be used. Defaults to None.
            model (Optional[tf.keras.models.Model]): The keras model to use. You can provide your own model
                with this argument. If set to None, the model will be built to the specifications of parameters using
                either the functional or sequential API. Defaults to None
            read_only (bool): Whether to write anything to disk (False) or not (True). Defaults to False.

        """
        # parameters
        self._using_hypercube = False
        if parameters is None:
            self.p = ADCParameters()
        else:
            self.p = parameters

        # seed
        if self.p.seed is not None:
            tf.random.set_seed(self.p.seed)

        # read_only
        self.read_only = read_only

        # save params and create dir
        if not self.read_only:
            self.p.write_summary = True
            self.p.save()
            print(
                "Output files are saved to {}".format(self.p.main_path),
                "as defined in 'main_path' in the parameters.",
            )

        # check whether Tensorboard and Read-Only makes Sense
        if self.read_only and self.p.tensorboard:
            raise Exception("Can't use tensorboard, when `read_only` is set to True.")

        # clear old sessions
        tf.keras.backend.clear_session()

        # get the CVs:
        if trajs is not None:
            if isinstance(trajs, str):
                self.trajs = TrajEnsemble([trajs], [top])
            else:
                self.trajs = trajs

            if (
                all([traj._traj_file.suffix in [".h5", ".nc"] for traj in trajs])
                and trajs.CVs_in_file
            ):
                dataset = trajs.tf_dataset(batch_size=self.p.batch_size)
                self.inp_CV_data = trajs.CVs
            else:
                # load missing values
                should_be = {
                    "central_angles",
                    "central_cartesians",
                    "central_dihedrals",
                    "central_distances",
                    "side_dihedrals",
                }
                if dataset is None:
                    if not self.trajs.CVs:
                        missing = list(should_be - set(trajs.CVs.keys()))
                        if missing != []:
                            print("loading missing values: ", missing)
                            self.trajs.load_CVs(missing, ensemble=False)
                    else:
                        if not should_be.issubset(set(self.trajs.CVs.keys())):
                            self.trajs.load_CVs(list(should_be), ensemble=False)

                    if not should_be.issubset(set(self.trajs.CVs.keys())):
                        raise Exception(
                            f"Could not load CVs. Should be {should_be}, but "
                            f"currently only {set(trajs.CVs.keys())} are loaded."
                        )

        # create dataset
        if dataset is None:
            (
                self.sparse,
                self.train_data,
                self.inp_CV_data,
            ) = self.get_train_data_from_trajs(self.trajs, self.p)
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
        else:
            self.dataset = dataset
            self.sparse = any(
                [isinstance(t, tf.SparseTensorSpec) for t in self.dataset.element_spec]
            )

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
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

        # do this if tensorboard is true.
        if self.p.tensorboard:
            # print shapes
            print("input shapes are:")
            if hasattr(self, "inp_CV_data"):
                print({k: v.shape for k, v in self.inp_CV_data.items()})
            else:
                for d in self.dataset:
                    break
                print([v.shape for v in d])

            # set _log_images False to fix the backend after training
            self._log_images = False
            # get the output from model summary.
            with Capturing() as output:
                self.model.summary()
            with open(self.p.main_path + "/model_summary.txt", "w") as f:
                f.write("\n".join(output))
            self.plot_network()
            print(
                f"Saved a text-summary of the model and an image in {self.p.main_path},",
                "as specified in 'main_path' in the parameters.",
            )

            # sets up the tb callback to plot the model
            self.tb_callback = tf.keras.callbacks.TensorBoard(
                self.p.main_path, write_graph=True
            )
            self.tb_callback.set_model(self.model)

    def train(self):
        """Overwrites the parent class' `train()` method to implement references."""
        if all([v == 1 for k, v in self.p.__dict__.items() if "reference" in k]):
            self.train_for_references()
        else:
            print("References are already provided. Skipping reference training.")
        super().train()

    def train_for_references(
        self,
    ) -> None:
        """Calculates the angle, dihedral, and cartesian costs to so-called
        references, which can be used to bring these costs to a similar
        magnitude.

        """
        p = ADCParameters(
            cartesian_cost_scale=1,
            angle_cost_scale=1,
            dihedral_cost_scale=1,
        )
        if hasattr(self, "trajs"):
            nsteps = int(self.trajs.n_frames / self.p.batch_size)
        else:
            return
        # fmt: off
        costs = {
            "dihedral_cost": ["central_dihedrals", 1, dihedral_loss(self.model, p)],
            "angle_cost": ["central_angles", 0, angle_loss(self.model, p)],
            "cartesian_cost": ["central_cartesians", 2, cartesian_loss(self.model, parameters=p)],
        }
        # fmt: on

        cost_references = {key: [] for key in costs.keys()}
        for key, val in costs.items():
            if key in ["dihedral_cost", "angle_cost"]:
                means = np.repeat(
                    np.expand_dims(
                        np.mean(self.trajs.CVs[val[0]], 0),
                        axis=0,
                    ),
                    repeats=self.p.batch_size,
                    axis=0,
                )
                costs[key].append(means)
            else:
                mean_lengths = np.expand_dims(
                    np.mean(self.trajs.CVs["central_distances"], axis=0), axis=0
                )
                chain = chain_in_plane(mean_lengths, costs["angle_cost"][3])
                gen_cartesians = dihedrals_to_cartesian_tf(
                    costs["dihedral_cost"][3] + np.pi, chain
                )
                pd = pairwise_dist(
                    gen_cartesians[
                        :,
                        self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
                    ],
                    flat=True,
                )
                costs[key].append(pd)

        with tqdm(
            desc="Calculating references",
            total=nsteps,
            position=0,
            leave=True,
        ) as pbar:
            for i, data in zip(range(nsteps), self.dataset):
                angles, dihedrals, cartesians = data[:3]
                for key, val in costs.items():
                    if key in ["dihedral_cost", "angle_cost"]:
                        cost_references[key].append(
                            val[2](data[val[1]], val[3]).numpy()
                        )
                    if key == "cartesian_cost":
                        pd = pairwise_dist(
                            data[val[1]][
                                :,
                                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
                            ],
                            flat=True,
                        )
                        c = val[2](val[3], pd).numpy()
                        cost_references["cartesian_cost"].append(c)
                pbar.update()
        s = {k: np.mean(v) for k, v in cost_references.items()}
        print(f"Setting cost references: {s} to parameters.")
        self.p.angle_cost_reference = float(np.mean(cost_references["angle_cost"]))
        self.p.dihedral_cost_reference = float(
            np.mean(cost_references["dihedral_cost"])
        )
        self.p.cartesian_cost_reference = float(
            np.mean(cost_references["cartesian_cost"])
        )
        if not self.read_only:
            self.p.save()
        return cost_references

    def set_train_data(self, data: TrajEnsemble) -> None:
        """Resets the train data for reloaded models."""
        (
            sparse,
            self.train_data,
            self.inp_CV_data,
        ) = self.get_train_data_from_trajs(self.trajs, self.p)
        if not self.sparse and sparse:
            print(
                f"The provided data contains nan's, but the model was trained "
                f"on dense input data."
            )
            return
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

    def plot_network(self) -> None:
        """Tries to plot the network using pydot, pydotplus and graphviz.
        Doesn't raise an exception if plotting is not possible.

        Note:
            Refer to this guide to install these programs:
            https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py

        """
        out = plot_model(self.model, None)
        if out is not None:
            out.save(Path(self.p.main_path) / "network.png")

    @staticmethod
    def get_train_data_from_trajs(
        trajs: TrajEnsemble,
        p: ADCParameters,
        attr: str = "CVs",
    ) -> tuple[bool, np.ndarray, dict[str, np.ndarray]]:
        """Builds train data from a `TrajEnsemble`.

        Args:
            trajs (TrajEnsemble): A `TrajEnsemble` instance.
            p (ADCParameters): An instance of `ADCParameters`.
            attr (str): Which attribute to get from `TrajEnsemble`. This defaults
                to 'CVs', because 'CVs' is usually a dict containing the CV data.
                However, you can build the train data from any dict in the `TrajEnsemble`.

        Returns:
            tuple: A tuple containing the following:
                bool: A bool that shows whether some 'CV' values are `np.nan` (True),
                    which will be used to decide whether the sparse training
                    will be used.
                np.ndarray: An array of features fed into the autoencoder,
                    concatenated along the feature axis. The order of the
                    features is: central_angles, central_dihedral, (side_dihedrals
                    if p.use_sidechain_dihedrals is True).
                dict[str, np.ndarray]: The training data as a dict. Containing
                    all values in `trajs.CVs`.

        """
        if not any([np.isnan(x).any() for x in getattr(trajs, attr).values()]):
            inp_CV_data = {
                key: val.astype("float32") for key, val in getattr(trajs, attr).items()
            }
            sparse = False
        else:
            sparse = True
            print("Input contains nans. Using sparse network.")

            # check whether the nans are correctly distributed
            for k, v in trajs.CVs.items():
                if v.ndim == 3:
                    v = np.any(np.all(np.isnan(v), (1, 2)))
                else:
                    v = np.any(np.all(np.isnan(v), 1))
                if v:
                    raise Exception(
                        f"Stacking of frames for CV `{k}` did not "
                        f"succeed. There are frames full of nans."
                    )

            # build the CV data
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
                pass
            elif p.use_backbone_angles and not p.use_sidechains:
                assert len(train_data) == 2
            else:
                assert len(train_data) == 3

        return sparse, train_data, inp_CV_data

    @classmethod
    def from_checkpoint(
        cls: Type[AngleDihedralCartesianEncoderMapType],
        trajs: Union[None, TrajEnsemble],
        checkpoint_path: Union[Path, str],
        dataset: Optional[tf.data.Dataset] = None,
        use_previous_model: bool = False,
        compat: bool = False,
    ) -> AngleDihedralCartesianEncoderMapType:
        """Reconstructs the model from a checkpoint.

        Although the model can be loaded from disk without any form of data and
        still yield the correct input and output shapes, it is required to either
        provide `trajs` or `dataset` to double-check, that the correct model will
        be reloaded.

        This is also, whe the `sparse` argument is not needed, as sparcity of the
        input data is a property of the `TrajEnsemble` provided.

        Args:
            trajs (Union[None, TrajEnsemble]): Either None (in which case, the
                argument `dataset` is required), or an instance of `TrajEnsemble`,
                which was used to instantiate the `AngleDihedralCartesianEncoderMap`,
                before it was saved to disk.
            checkpoint_path (Union[Path, str]): The path to the checkpoint. Can
                either be the path to a .keras file or to a directory containing
                .keras files, in which case the most recently created .keras
                file will be used.
            dataset (Optional[tf.data.Dataset]): If `trajs` is not provided, a
                dataset is required to make sure the input shapes match the model,
                that is stored on the disk.
            use_previous_model (bool): Set this flag to True, if you load a model
                from an in-between checkpoint step (e.g., to continue training with
                different parameters). If you have the files saved_model_0.keras,
                saved_model_500.keras and saved_model_1000.keras, setting this to
                True and loading the saved_model_500.keras will back up the
                saved_model_1000.keras.
            compat (bool): Whether to use compatibility mode when missing or wrong
                parameter files are present. In this special case, some assumptions
                about the network architecture are made from the model and the
                parameters in parameters.json overwritten accordingly (a backup
                will also be made).

        Returns:
            AngleDihedralCartesianEncoderMapType: An instance of `AngleDihedralCartesianEncoderMap`.

        """
        return load_model(
            cls,
            checkpoint_path,
            trajs=trajs,
            dataset=dataset,
            use_previous_model=use_previous_model,
            compat=compat,
        )

    def _setup_callbacks(self) -> None:
        """Overwrites the parent class' `_setup_callbacks` method.

        Due to the 'soft start' of the cartesian cost, the `cartesian_increase_callback`
        needs to be added to the list of callbacks.

        """
        super(self.__class__, self)._setup_callbacks()
        if self.p.cartesian_cost_scale_soft_start != (None, None):
            self.cartesian_increase_callback = IncreaseCartesianCost(self.p)
            self.callbacks.append(self.cartesian_increase_callback)
        if self.p.track_clashes:
            print("Clash metric not yet implemented.")
            # self.metrics.append(ADCClashMetric(parameters=self.p))
        if self.p.track_RMSD:
            print("RMSD metric not yet implemented.")
            # self.metrics.append(ADCRMSDMetric(parameters=self.p))

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
                    print_current_scale=False,
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

    def encode(
        self,
        data: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Runs the central_angles, central_dihedrals, (side_dihedrals) through the
        autoencoder. Make sure that `data` has the correct shape.

        Args:
             data (Sequence[np.ndarray]): Provide a sequence of angles, and
                central_dihedrals, if you used sidechain_dihedrals during training
                append these to the end of the sequence.

        Returns:
            np.ndarray: The latent space representation of the provided `data`.

        """
        if hasattr(data, "_traj_file"):
            _, data, __ = self.get_train_data_from_trajs(data, self.p, attr="_CVs")
        elif hasattr(data, "traj_files"):
            _, data, __ = self.get_train_data_from_trajs(data, self.p)
        return super().encode(data)

    @overload
    def generate(
        self,
        points: np.ndarray,
        top: Optional[Union[str, int, md.Topology]] = None,
        backend: Literal["mdtraj"] = "mdtraj",
    ) -> Union[md.Trajectory]:
        ...

    @overload
    def generate(
        self,
        points: np.ndarray,
        top: Optional[Union[str, int, md.Topology]] = None,
        backend: Literal["mdanalysis"] = "mdanalysis",
    ) -> Union[mda.Universe]:
        ...

    def generate(
        self,
        points: np.ndarray,
        top: Optional[Union[str, int, md.Topology]] = None,
        backend: Literal["mdtraj", "mdanalysis"] = "mdtraj",
    ) -> Union[mda.Universe, md.Trajectory]:
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
                *.gro, *.h5> file. Alternatively, None can be provided; in which
                case, the internal topology (`self.top`) of this class is used.
                Defaults to None.
            backend (str): Defines what MD python package is to use, to build the
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
        angles, dihedrals, sidechain_dihedrals = out

        if top is None:
            if len(self.trajs.top) > 1:
                print(
                    f"Please specify which topology you would like to use for generating "
                    f"conformations. You can either provide a `str` to a topology file "
                    f"(file extension .pdb, .h5, .gro) on disk, or a `int` specifying the "
                    f"`SingleTraj` object in this class' {self.trajs.n_trajs} trajs, or "
                    f"you can also specify a `mdtraj.Topology` object."
                )
                return
            else:
                traj = self.trajs[0]
                mdanalysis_traj = self.trajs[0][0]
        elif isinstance(top, int):
            mdanalysis_traj = self.trajs[top][0]
        elif isinstance(top, str):
            mdanalysis_traj = md.load(top)
        elif isinstance(top, md.Topology):
            mdanalysis_traj = top
        else:
            raise ValueError(
                f"Type of argument `top` must be int, str, md.Topology. You provided {type(top)}."
            )

        # do the backmapping
        if backend == "mdanalysis":
            with NamedTemporaryFile(suffix=".pdb") as f:
                mdanalysis_traj.save_pdb(f.name)
                uni = dihedral_backmapping(
                    f.name, dihedrals, sidechains=sidechain_dihedrals
                )
            return uni
        elif backend == "mdtraj":
            traj = mdtraj_backmapping(top, dihedrals, sidechain_dihedrals, self.trajs)
            return traj
        else:
            raise TypeError(
                f"backend must be 'mdtraj' or 'mdanalysis', but you provided {backend}"
            )
