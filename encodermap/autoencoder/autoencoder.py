# -*- coding: utf-8 -*-
# encodermap/autoencoder/autoencoder.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
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
import os
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

# Third Party Imports
import numpy as np
import tensorflow as tf
from optional_imports import _optional_import
from tqdm import tqdm

# Encodermap imports
from encodermap.callbacks.callbacks import (
    CheckpointSaver,
    ImageCallback,
    IncreaseCartesianCost,
    ProgressBar,
    TensorboardWriteBool,
)
from encodermap.callbacks.metrics import ADCClashMetric, ADCRMSDMetric
from encodermap.encodermap_tf1.backmapping import (
    chain_in_plane,
    dihedrals_to_cartesian_tf,
)
from encodermap.loss_functions.loss_functions import (
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
from encodermap.misc.backmapping import dihedral_backmapping, mdtraj_backmapping
from encodermap.misc.distances import pairwise_dist
from encodermap.misc.misc import create_n_cube, plot_model
from encodermap.misc.saving_loading_models import load_model, save_model
from encodermap.models.models import gen_functional_model, gen_sequential_model
from encodermap.parameters.parameters import ADCParameters, Parameters
from encodermap.trajinfo.info_all import TrajEnsemble
from encodermap.trajinfo.info_single import Capturing, SingleTraj


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
    Any,
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
AutoencoderClass = Union[
    AutoencoderType,
    EncoderMapType,
    DihedralEncoderMapType,
    AngleDihedralCartesianEncoderMapType,
]


if TYPE_CHECKING:
    # Third Party Imports
    from MDAnalysis import Universe
    from mdtraj import Topology, Trajectory


################################################################################
# Globals
################################################################################


__all__: list[str] = [
    "Autoencoder",
    "EncoderMap",
    "AngleDihedralCartesianEncoderMap",
    "DihedralEncoderMap",
]


################################################################################
# Utils
################################################################################


def np_to_sparse_tensor(a: np.ndarray) -> tf.sparse.SparseTensor:
    """Converts a numpy array with nans to a SparseTensor.

    Args:
        a (np.ndarray): The input array.

    Returns:
        tf.sparse.SparseTensor: The corresponding SparseTensor.

    """
    orig_shape = a.shape
    indices = np.stack(np.where(~np.isnan(a))).T.astype("int64")
    dense_shape = a.shape
    a = a[~np.isnan(a)].flatten()
    if np.any(np.isnan(a)):
        raise Exception(
            f"NaN values in array with shape {orig_shape} could not be removed "
            f"by indexing with {indices=}. This will result in the SparseTensor "
            f"containing NaN values."
        )
    return tf.sparse.SparseTensor(indices, a, dense_shape)


def _add_images_to_tensorboard(
    autoencoder: AutoencoderClass,
    data: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    image_step: Optional[int] = None,
    max_size: int = 10_000,
    mpl_scatter_kws: Optional[dict] = None,
    mpl_hist_kws: Optional[dict] = None,
    plotly_scatter_kws: Optional[dict] = None,
    plotly_hist_kws: Optional[dict] = None,
    additional_fns: Optional[Sequence[Callable]] = None,
    when: Literal["epoch", "batch"] = "epoch",
    save_to_disk: bool = False,
) -> None:
    """Adds images to Tensorboard using the data in data and the ids in ids.

    Args:
        data (Optional[Union[np.ndarray, Sequence[np.ndarray]]): The input-data will
            be passed through the encoder part of the autoencoder. If None
            is provided, a set of 10_000 points from `self.train_data` will
            be taken. A list[np.ndarray] is needed for the functional API of the
            `AngleDihedralCartesianEncoderMap`, that takes a list of
            [angles, dihedrals, side_dihedrals]. Defaults to None.
        backend (Literal["matplotlib", "plotly"]: Which backend to use for
            plotting. Defaults to 'matplotlib'.
        mpl_scatter_kws (Optional[dict]): A dictionary, that `matplotlib.pyplot.scatter`
            takes as keyword args. If None is provided, the default dict
            is {"s": 20}. Defaults to None.
        mpl_hist_kws (Optional[dict]): A dictionary, that `matplotlib.pyplot.histogram`
            takes as keyword args. If None is provided, the default dict
            is {"bins": 50}. Defaults to None.
        plotly_scatter_kws (Optional[dict[str, Any]]): A dict with items that
            `plotly.express.scatter()` will accept. If None is provided,
            a dict with size 20 will be passed to
            `px.scatter(**{'size_max': 10, 'opacity': 0.2})`,
            which sets an appropriate size of scatter points for the size of
            datasets encodermap is usually used for.
        plotly_hist_kws (Optional[dict[str, Any]]): A dict with items that
            `encodermap.plot.plotting._plot_free_energy()` will accept. If None is provided a
            dict with bins 50 will be passed to
            `encodermap.plot.plotting._plot_free_energy(**{'bins': 50})`.
            You can choose a colormap here by providing `{'bins': 50, 'cmap':
            'plasma'}` for this argument.
        image_step (Optional[int]): The interval in which to plot
            images to tensorboard. If None is provided, the `image_step`
            will be the same as `Parameters.summary_step`. Defaults to None.
        max_size (int): The maximum size of the high-dimensional data, that is
            projected. Prevents excessively large-datasets from being projected
            at every `image_step`. Defaults to 10_000.
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
        save_to_disk (bool): Whether to also write the images to disk.

    """
    if not autoencoder.p.tensorboard:
        warnings.warn(
            "Nothing is written to Tensorboard for this model. "
            "Please change parameters.tensorboard to True."
        )
        return
    if any([isinstance(i, ImageCallback) for i in autoencoder.callbacks]):
        warnings.warn(
            f"This instance of {autoencoder.__class__.__name__} already has an "
            f"`ImageCallback`. While it's allowed to have multiple ImageCallbacks, "
            f"training performance suffers by adding more."
        )
    if image_step is None:
        image_step = autoencoder.p.summary_step

    if mpl_scatter_kws is None:
        mpl_scatter_kws = {"s": 20}
    if mpl_hist_kws is None:
        mpl_hist_kws = {"bins": 50}
    if plotly_scatter_kws is None:
        plotly_scatter_kws = {"size_max": 1, "opacity": 0.2}
    if plotly_hist_kws is None:
        plotly_hist_kws = {"bins": 50}
    if backend not in ["matplotlib", "plotly"]:
        raise Exception(f"Argument `backend` must be either 'plotly' or 'matplotlib'.")

    autoencoder._log_images = True
    if data is None:
        if hasattr(autoencoder, "train_data"):
            data = autoencoder.train_data
        else:
            if hasattr(autoencoder, "trajs"):
                data = autoencoder.get_train_data_from_trajs(
                    autoencoder.trajs, autoencoder.p, max_size=max_size
                )[1]
            else:
                if (
                    not autoencoder.p.use_backbone_angles
                    and not autoencoder.p.use_sidechains
                ):
                    data = [[]]
                elif (
                    autoencoder.p.use_backbone_angles
                    and not autoencoder.p.use_sidechains
                ):
                    data = [[], []]
                elif autoencoder.p.use_sidechains and autoencoder.p.use_backbone_angles:
                    data = [[], [], []]
                else:
                    raise Exception(
                        "Only allowed combinations are:\n"
                        "   * No sidechains, no backbone angles\n"
                        "   * No sidechains, yes backbone angles\n"
                        "   * Yes Sidechains, yes backbone angles\n"
                        f"Your parameters are: {autoencoder.p.use_sidechains=}. {autoencoder.p.use_backbone_angles=}"
                    )
                length = 0
                for d in autoencoder.dataset:
                    if len(data) == 1:
                        data[0].append(d[1])
                    elif len(data) == 2:
                        data[0].append(d[0])
                        data[1].append(d[1])
                    elif len(data) == 3:
                        data[0].append(d[0])
                        data[1].append(d[1])
                        data[2].append(d[-1])
                    length += autoencoder.p.batch_size
                    if length > max_size:
                        break
                for i, o in enumerate(data):
                    if any([isinstance(d, tf.sparse.SparseTensor) for d in o]):
                        o = [
                            tf.sparse.to_dense(_, default_value=np.nan).numpy()
                            for _ in o
                        ]
                        o = np.concatenate(o)
                        data[i] = np_to_sparse_tensor(o)
                    else:
                        data[i] = np.concatenate(o)
    else:
        max_size = -1

    if isinstance(data, (np.ndarray, tf.sparse.SparseTensor)):
        if hasattr(autoencoder, "_tensorboard_data_req_shape"):
            assert np.array_equal(
                tf.shape(data).numpy()[1:], autoencoder._tensorboard_data_req_shape[1:]
            ), (
                f"The provided `data` has the wrong shape. The provided data has "
                f"shape {tf.shape(data).numpy()}, whereas {autoencoder._tensorboard_data_req_shape} "
                f"was expected."
            )
        else:
            for d in autoencoder.dataset:
                break
            if isinstance(data, (tuple, list)):
                assert data[0].shape[1:] == d[1].shape[1:], (
                    f"The provided `data` has the wrong shape. The provided data has "
                    f"shape {data[0].shape[1:]}, whereas {d[1].shape[1:]} "
                    f"was expected."
                )
            else:
                assert data.shape[1:] == d[1].shape[1:], (
                    f"The provided `data` has the wrong shape. The provided data has "
                    f"shape {data[0].shape[1:]}, whereas {d[1].shape[1:]} "
                    f"was expected."
                )
        if data.shape[0] > max_size and max_size >= 0:
            idx = np.unique(
                np.round(np.linspace(0, data.shape[0] - 1, max_size)).astype(int)
            )
            if isinstance(data, tf.sparse.SparseTensor):
                data = tf.sparse.to_dense(data, default_value=np.nan).numpy()[idx]
                data = np_to_sparse_tensor(data)
            else:
                data = data[idx]
        if isinstance(data, np.ndarray):
            if np.any(np.isnan(data)):
                data = np_to_sparse_tensor(data)
    elif isinstance(data, (tuple, list)):
        for d in autoencoder.dataset:
            break
        if len(data) == 1:
            assert data[0].shape[1:] == d[1].shape[1:], (
                f"The provided `data` has the wrong shape. The provided data has "
                f"shape {data[0].shape[1:]}, whereas {d[1].shape[1:]} "
                f"was expected."
            )
            data = data[0]
        elif len(data) == 2:
            assert (
                data[0].shape[1:] == d[0].shape[1:]
                and data[1].shape[1:] == d[1].shape[1:]
            ), (
                f"The provided `data` has the wrong shape. The provided data has "
                f"shape {[_.shape[1:] for _ in data]}, whereas {[d[0].shape[1:], d[1].shape[1:]]} "
                f"was expected."
            )
        elif len(data) == 3:
            assert (
                data[0].shape[1:] == d[0].shape[1:]
                and data[1].shape[1:] == d[1].shape[1:]
                and data[2].shape[1:] == d[-1].shape[1:]
            ), (
                f"The provided `data` has the wrong shape. The provided data has "
                f"shape {[_.shape[1:] for _ in data]}, whereas {[d[0].shape[1:], d[1].shape[1:], d[-1].shape[1:]]} "
                f"was expected."
            )
    else:
        raise TypeError(
            f"Argument `data` should be of type None, np.ndarray, tuple, or "
            f"list, you provided {type(data)}."
        )

    # add the callback
    if save_to_disk:
        save_dir = Path(autoencoder.p.main_path) / "train_images"
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = None

    autoencoder.callbacks.append(
        ImageCallback(
            parameters=autoencoder.p,
            highd_data=data,
            image_step=image_step,
            backend=backend,
            mpl_scatter_kws=mpl_scatter_kws,
            mpl_hist_kws=mpl_hist_kws,
            plotly_scatter_kws=plotly_scatter_kws,
            plotly_hist_kws=plotly_hist_kws,
            additional_fns=additional_fns,
            when=when,
            save_dir=save_dir,
        )
    )
    autoencoder.callbacks[-1].model = autoencoder.model
    if isinstance(data, (np.ndarray, tf.sparse.SparseTensor)):
        print(
            f"Logging images with {data.shape}-shaped data every "
            f"{image_step} epochs to Tensorboard at {autoencoder.p.main_path}"
        )
    else:
        print(
            f"Logging images with {[i.shape for i in data]}-shaped data "
            f"every {image_step} epochs to Tensorboard at {autoencoder.p.main_path}"
        )


def _print_save_message(autoencoder: AutoencoderClass) -> None:
    if autoencoder.p.main_path == Path(os.getcwd()):
        print(
            f"Output files are saved to {autoencoder.p.main_path}, which is the "
            f"current working trajectory."
        )
    else:
        print(
            f"Output files are saved to {autoencoder.p.main_path} as defined "
            f"in 'main_path' in the parameters.",
        )


def _get_model(autoencoder: AutoencoderClass) -> tf.keras.Model:
    """sets self.model according to `model_api` argument in self.parameters."""
    model = autoencoder.p.model_api
    if model == "functional":
        assert isinstance(autoencoder, AngleDihedralCartesianEncoderMap)
        d = autoencoder.dataset.take(1)
        if any(isinstance(_, tf.SparseTensorSpec) for _ in d.element_spec):
            autoencoder.sparse = True
        if hasattr(autoencoder.p, "reconstruct_sidechains"):
            if autoencoder.p.reconstruct_sidechains:
                assert len(d.element_spec) == 7
        model = gen_functional_model(
            autoencoder.dataset,
            autoencoder.p,
            sparse=autoencoder.sparse,
        )
    elif model == "sequential":
        assert (
            isinstance(autoencoder, (Autoencoder, EncoderMap, DihedralEncoderMap))
            or autoencoder.__class__.__name__ == "EncoderMap"
        )
        if isinstance(autoencoder.train_data, tf.sparse.SparseTensor):
            autoencoder.sparse = True
        try:
            model = gen_sequential_model(
                autoencoder.train_data.shape[1],
                autoencoder.p,
                sparse=autoencoder.sparse,
            )
        except AttributeError:
            if autoencoder.p.training == "custom":
                for d in autoencoder.train_data:
                    break
                model = gen_sequential_model(
                    d[0].get_shape().as_list()[1],
                    autoencoder.p,
                    sparse=autoencoder.sparse,
                )
            elif autoencoder.p.training == "auto":
                for d, _ in autoencoder.train_data:
                    break
                model = gen_sequential_model(
                    d.get_shape().as_list()[1],
                    autoencoder.p,
                    sparse=autoencoder.sparse,
                )
            else:
                raise Exception(
                    f"Parameter `training` has to be one of 'custom', 'auto'. "
                    f"You supplied '{autoencoder.p.training}'."
                )

    elif model == "custom":
        raise NotImplementedError("No custom API currently supported")
    else:
        raise ValueError(
            f"API argument needs to be one of `functional`, `sequential`, "
            f"`custom`. You provided '{model}'."
        )
    assert not isinstance(model, str)
    return model


##############################################################################
# Function definition which allows self.p.tensorboard to be passed
##############################################################################


def function(f, tensorboard=False):
    """Compiles functions with `tensorflow.function` based on a `tensorboard`
    parameter.


    To understand the necessity of this function, we need to have a look at how
    tensorflow executes computations. There are two modes of execution:
    * eager mode: In eager mode, the computations are handled by python.
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
        encoder (tf.keras.Model): The encoder submodel of `self.model`.
        decoder (tf.keras.Model): The decoder submodel of `self.model`.
        loss (Sequence[Callable]): A list of loss functions passed to the model
            when it is compiled. When the main `Autoencoder` class is used and
            `parameters.loss` is 'emap_cost', this list comprises center_cost,
            regularization_cost, auto_cost. When the `EncoderMap` sub-class is
            used and `parameters.loss` is 'emap_cost', distance_cost is added to
            the list. When `parameters.loss` is not 'emap_cost', the loss can either
            be a string ('mse'), or a function, that both are acceptable
            arguments for loss, when a keras model is compiled.

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
            parameters (Union[encodermap.Parameters, None], optional): The
                parameters to be used. If None is provided default values
                (check them with `print(em.Parameters.defaults_description()`))
                are used. Defaults to None.
            train_data (Union[np.ndarray, tf.data.Dataset, None], optional):
                The train data. Can be one of the following:
                    * None: If None is provided points on the edges of a
                        4-dimensional hypercube will be used as train data.
                    * np.ndarray: If a numpy array is provided, it will be
                        transformed into a batched tf.data.Dataset by first
                        making it an infinitely repeating dataset, shuffling
                        it and the batching it with a batch size specified
                        by parameters.batch_size.
                    * tf.data.Dataset: If a dataset is provided it will be
                        used without making any adjustments. Make sure, that the
                        dataset uses `float32` as its type.
                Defaults to None.
            model (Union[tf.keras.models.Model, None], optional): Providing
                a keras model to this argument will make the Autoencoder/EncoderMap
                class use this model instead of the predefined ones. Make sure
                the model can accept EncoderMap's loss functions. If None is
                provided the model will be built using the specifications in
                parameters. Defaults to None.
            read_only (bool, optional): Whether the class is allowed to write
                to disk (False) or not (True). Defaults to False and will allow
                the class to write to disk.

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
            self._print_save_message()

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
            self.model = self._get_model()
        else:
            self.model = model

        # setup callbacks for nice progress bars and saving every now and then
        self._setup_callbacks()

        # create loss based on user input
        self.loss = self._get_loss()

        # choose optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.p.learning_rate, clipvalue=1.0
        )

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

    def _print_save_message(self) -> None:
        """Prints a save message to inform the user where the model is saved."""
        _print_save_message(self)

    def _print_missing_data(self) -> None:
        print(
            f"This model was reloaded from disk, but not yet provided with train "
            f"data. Use the `set_train_data()` method to provide the train "
            f"data and call `train()` again to train the model. Alternatively, "
            f"you could directly provide the train data, when reloading by "
            f"calling the `{self.__class__.__name__}.from_checkpoint()` "
            f"constructor with the `train_data` argument. Expected shape = "
            f"{self.model.encoder_model.input_shape[1]}, received shape = "
            f"{self._tensorboard_data_req_shape} {self._using_hypercube=} "
            f"{self.p.using_hypercube=} {self.dataset.element_spec=}"
        )
        return

    def set_train_data(self, data: Union[np.ndarray, tf.data.Dataset]) -> None:
        """Resets the train data for reloaded models."""
        self._using_hypercube = False
        if data is None:
            self._using_hypercube = True
            self.p.using_hypercube = True
            self.train_data = create_n_cube(4, seed=self.p.seed)[0].astype("float32")
            self.p.periodicity = float("inf")
        elif isinstance(data, np.ndarray):
            if np.any(np.isnan(data)):
                self.sparse = True
                print("Input contains nans. Using sparse network.")
                self.train_data = np_to_sparse_tensor(data)
            else:
                self.train_data = data.astype("float32")
        elif isinstance(data, tf.data.Dataset):
            self.dataset = data
            try:
                _, __ = self.dataset.take(1)
            except ValueError:
                if self.p.training == "auto":
                    if self.p.model_api == "custom":
                        print(
                            f"It seems like your dataset only yields tensors and not "
                            f"tuples of tensors. TensorFlow is optimized for classification "
                            f"tasks, where datasets yield tuples of (data, classes). EncoderMap,"
                            f"however is a regression task, but uses the same code as the "
                            f"classification tasks. This requires the dataset provided "
                            f"for a tensorflow model.fit() method to return tuples "
                            f"of (data, classes). Your dataset does not do this. "
                            f"I will transform your dataset using "
                            f"the `tf.data.Dataset.zip()` function of `tf.data`. "
                            f"This might break your custom model. You can "
                            f"set the `training` parameter in the parameter class to "
                            f"'custom' to not alter your dataset."
                        )
                    self.dataset = tf.data.Dataset.zip((self.dataset, self.dataset))
                    _ = self.dataset.take(1)
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
            self._tensorboard_data_req_shape = tf.shape(self.train_data).numpy()
        else:
            d = self.train_data.element_spec[0]
            if isinstance(d, tuple):
                self._tensorboard_data_req_shape = d[0].shape.as_list()
            else:
                self._tensorboard_data_req_shape = d.shape.as_list()

        if isinstance(data, np.ndarray):
            assert data.shape[1] == self._tensorboard_data_req_shape[1]

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

    def _get_model(self) -> tf.keras.Model:
        return _get_model(self)

    @property
    def encoder(self) -> tf.keras.Model:
        """tf.keras.Model: Encoder part of the model."""
        return self.model.encoder

    @property
    def decoder(self) -> tf.keras.Model:
        """tf.keras.Model: Decoder part of the model."""
        return self.model.decoder

    def _get_loss(self):
        """sets self.loss according to `loss` in self.parameters."""
        loss = self.p.loss
        if loss == "reconstruction_loss":
            loss = reconstruction_loss(self.model)
        elif loss == "emap_cost":
            self.auto_loss = auto_loss(self.model, self.p, self.tensorboard_write_bool)
            self.regularization_loss = regularization_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            self.center_loss = center_loss(
                self.model, self.p, self.tensorboard_write_bool
            )
            loss = [self.auto_loss, self.regularization_loss, self.center_loss]
        elif loss == "mse":
            loss = "mse"
        else:
            raise ValueError(
                f"loss argument needs to be `reconstruction_loss`, `mse` or `emap_cost`. You provided '{loss}'."
            )
        return loss

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

        if self._using_hypercube == self.p.using_hypercube:
            if self.p.periodicity == float("inf"):
                if (
                    self._tensorboard_data_req_shape[1]
                    != self.model.encoder_model.input_shape[1]
                ):
                    return self._print_missing_data()
            else:
                if (
                    self._tensorboard_data_req_shape[1]
                    != self.model.encoder_model.input_shape[1] // 2
                ):
                    return self._print_missing_data()
        else:
            return self._print_missing_data()

        if self.p.training == "custom" and self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "custom" and not self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "auto":
            epochs = self.p.n_steps - self.p.current_training_step
            try:
                history = self.model.fit(
                    self.dataset,
                    batch_size=self.p.batch_size,
                    epochs=epochs,
                    steps_per_epoch=1,
                    verbose=0,
                    callbacks=self.callbacks,
                )
            except ValueError:
                raise Exception(
                    f"{self.model.encoder_model.input_shape=} {self._tensorboard_data_req_shape=} "
                    f"{self.train_data.shape=} {self.dataset.element_spec=} {self.p.using_hypercube=} {self._using_hypercube=}"
                )
        else:
            raise ValueError(
                f"training argument needs to be `auto` or `custom`. You provided '{self.training}'."
            )
        self.p.current_training_step += self.p.n_steps - self.p.current_training_step
        self.p.save()
        self.save()

        return history

    def add_loss(self, loss):
        """Adds a new loss to the existing losses."""
        try:
            self.loss.append(loss(self.model, self.p))
        except TypeError:
            self.loss.append(loss(self.model))

    def add_callback(self, callback):
        """Adds a new callback to the existing callbacks."""
        try:
            self.callbacks.append(callback(self.p))
        except TypeError:
            self.callbacks.append(callback)

    def add_metric(self, metric):
        """Adds a new metric to the existing metrics."""
        self.metrics.append(metric(self.p))

    def add_images_to_tensorboard(self, *args: Any, **kwargs: Any) -> None:
        """Adds images of the latent space to tensorboard.

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
            scatter_kws (Optional[dict[str, Any]]): A dict with items that
                `plotly.express.scatter()` will accept. If None is provided,
                a dict with size 20 will be passed to
                `px.scatter(**{'size_max': 10, 'opacity': 0.2})`,
                which sets an appropriate size of scatter points for the size of
                datasets encodermap is usually used for.
            hist_kws (Optional[dict[str, Any]]): A dict with items that
                `encodermap.plot.plotting._plot_free_energy()` will accept. If None is provided a
                dict with bins 50 will be passed to
                `encodermap.plot.plotting._plot_free_energy(**{'bins': 50})`.
                You can choose a colormap here by providing `{'bins': 50, 'cmap':
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
            save_to_disk (bool): Whether to also write the images to disk.

        """
        _add_images_to_tensorboard(self, *args, **kwargs)

    def plot_network(self) -> None:
        """Tries to plot the network using pydot, pydotplus and graphviz.
        Doesn't raise an exception if plotting is not possible.

        Note:
            Refer to this guide to install these programs:
            https://stackoverflow.com/questions/47605558/importerror-failed-to-import-pydot-you-must-install-pydot-and-graphviz-for-py

        """
        try:
            out = plot_model(self.model, self.train_data.shape[1])
            if out is not None:
                out.save(Path(self.p.main_path) / "network.png")
        except:
            pass

    def encode(self, data: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
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

    def decode(self, data: np.ndarray) -> Sequence[np.ndarray]:
        """Calls the decoder part of the model.

        `AngleDihedralCartesianAutoencoder` will, like the other two classes'
        output a list of np.ndarray.

        Args:
            data (np.ndarray): The data to be passed to the decoder part of
                the model. Make sure that the shape of the data matches the
                number of neurons in the latent space.

        Returns:
            Union[list[np.ndarray], np.ndarray]: Outputs from the decoder part.
                For `AngleDihedralCartesianEncoderMap`, this will be a list of
                np.ndarray.

        """
        out = self.model.decoder(data)
        if isinstance(out, (list, tuple)):
            out = [o.numpy() for o in out]
        else:
            out = out.numpy()
        return out

    def save(self, step: Optional[int] = None) -> None | Path:
        """Saves the model to the current path defined in `parameters.main_path`.

        Args:
            step (Optional[int]): Does not save the model at the given
                training step, but rather changes the string used for saving
                the model from a datetime format to another.

        Returns:
            Union[None, Path]: When the model has been saved, the Path will
                be returned. If the model could not be saved. None will be
                returned.

        """
        if not self.read_only:
            return save_model(
                self.model,
                self.p.main_path,
                inp_class_name=self.__class__.__name__,
                step=step,
                print_message=True,
            ).parent
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

    def _get_loss(self):
        loss = self.p.loss
        if loss == "reconstruction_loss":
            loss = reconstruction_loss(self.model)
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
            loss = [
                self.auto_loss,
                self.regularization_loss,
                self.center_loss,
                self.distance_loss,
            ]
        elif loss == "mse":
            loss = "mse"
        else:
            raise ValueError(
                f"loss argument needs to be `reconstruction_loss`, `mse` or `emap_cost`. You provided '{loss}'."
            )
        return loss


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
    ) -> Universe:
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


class AngleDihedralCartesianEncoderMap:
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
        encodermap.TrajEnsemble object. Current backend is no_load. Containing 2 trajectories. Not containing any CVs.
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
        dataset: Optional[tf.data.Dataset] = None,
        ensemble: bool = False,
        use_dataset_when_possible: bool = True,
        deterministic: bool = False,
    ) -> None:
        """Instantiate the `AngleDihedralCartesianEncoderMap` class.

        Args:
            trajs (Optional[TrajEnsemble]): The trajectories to be used as input.
                If trajs contain no CVs, correct CVs will be loaded. can be None,
                in which case the argument `dataset` should be provided.
                Defaults to None.
            parameters (Optional[em.ADCParameters]): The parameters for the
                current run. Can be set to None and the default parameters will
                be used. Defaults to None.
            model (Optional[tf.keras.models.Model]): The keras model to use. You
                can provide your own model with this argument. If set to None,
                the model will be built to the specifications of parameters using
                either the functional API. Defaults to None,
            read_only (bool): Whether to write anything to disk
                (False) or not (True). Defaults to False.
            dataset (Optional[tf.data.Dataset]): The `dataset` argument takes
                precedent over the `trajs` argument. If None, the dataset will
                be constructed from the `trajs` argument (see
                `em.trajinfo.TrajEnsemble.tf_dataset` for more info). Defaults
                to None.
            ensemble (bool): Whether to allow non-defined features when
                featurizing the provided `trajs`. Only takes effect, when
                the `trajs` don't already have the features (central_cartesians,
                central_distances, central_angles, central_dihedrals, side_dihedrals)
                loaded. Defaults to False.
            use_dataset_when_possible (bool): Whether to use the `trajs` method
                `tf_dataset()` to get a dataset for training or constructy a
                dataset from the `trajs` CVs numpy arrays. For large datasets the
                first method can be advantageous as not all data will end up in
                memory and the dataset can be larger than the memory allows. For
                small datasets the second method is faster, as all data is in
                memory. Defaults to True.

        """
        # parameters
        if parameters is None:
            self.p = ADCParameters()
        else:
            assert isinstance(parameters, ADCParameters), (
                f"Please provide an instance of `ADCParameters` for the argument "
                f"'parameters' and not {type(parameters)}."
            )
            self.p = parameters

        # seed
        if self.p.seed is not None:
            tf.random.set_seed(self.p.seed)

        # check some sidechain
        if self.p.reconstruct_sidechains:
            assert (
                self.p.use_sidechains
            ), "If you want to reconstruct sidechains, you should also set `use_sidechains` to True."

        # read_only
        self.read_only = read_only

        # save params and create dir
        if not self.read_only:
            self.p.write_summary = True
            self.p.save()
            self._print_save_message()

        # check whether Tensorboard and read_only make sense
        if self.read_only and self.p.tensorboard:
            raise Exception("Can't use tensorboard, when `read_only` is set to True.")

        # clear old sessions
        tf.keras.backend.clear_session()

        # get the CVs:
        if trajs is not None:
            if trajs.__class__.__name__ == "SingleTraj":
                trajs = trajs._gen_ensemble()
            self.trajs = trajs

            # add the sidechain_info if sidechains need to be reconstructed
            if self.p.reconstruct_sidechains:
                self.p.sidechain_info = self.trajs.sidechain_info()

            # decide on the dataset
            if (
                all([traj._traj_file.suffix in [".h5", ".nc"] for traj in trajs])
                and trajs.CVs_in_file
                and use_dataset_when_possible
            ):
                # if all CVs in a h5 file, we can load get batches from there
                dataset = trajs.tf_dataset(
                    batch_size=self.p.batch_size,
                    sidechains=self.p.use_sidechains,
                    reconstruct_sidechains=self.p.reconstruct_sidechains,
                    deterministic=deterministic,
                )
                self.inp_CV_data = trajs.CVs

            else:
                # if not, we need to load them
                if not self.p.reconstruct_sidechains:
                    should_be = {
                        "central_angles",
                        "central_cartesians",
                        "central_dihedrals",
                        "central_distances",
                        "side_dihedrals",
                    }
                else:
                    raise NotImplementedError(
                        f"Loading CVs with reconstruct_sidechains is currently not implemented."
                    )
                if dataset is None:
                    if not self.trajs.CVs:
                        missing = list(should_be - set(trajs.CVs.keys()))
                        if missing != []:
                            print("loading missing values: ", missing)
                            self.trajs.load_CVs(missing, ensemble=ensemble)
                    else:
                        if not should_be.issubset(set(self.trajs.CVs.keys())):
                            self.trajs.load_CVs(list(should_be), ensemble=ensemble)

                    if not should_be.issubset(set(self.trajs.CVs.keys())):
                        if not ensemble:
                            msg = (
                                f" You can try to set `ensemble=True` to load "
                                f"these trajectories into an ensemble, which "
                                f"allows features with different feature length."
                            )
                        else:
                            msg = ""
                        raise Exception(
                            f"Could not load CVs. Should be {should_be}, but "
                            f"currently only {set(trajs.CVs.keys())} are loaded.{msg}"
                        )

        # create dataset
        if dataset is None:
            (
                self.sparse,
                self.train_data,
                self.inp_CV_data,
            ) = self.get_train_data_from_trajs(self.trajs, self.p)
            if not self.p.reconstruct_sidechains:
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
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (
                        self.inp_CV_data["central_angles"],
                        self.inp_CV_data["central_dihedrals"],
                        self.inp_CV_data["all_cartesians"],
                        self.inp_CV_data["central_distances"],
                        self.inp_CV_data["side_dihedrals"],
                        self.inp_CV_data["side_angles"],
                        self.inp_CV_data["side_dihedrals"],
                        self.inp_CV_data["side_distances_"],
                    )
                )
                dataset = dataset.shuffle(
                    buffer_size=self.inp_CV_data["all_cartesians"].shape[0],
                    reshuffle_each_iteration=True,
                )
            dataset = dataset.repeat()
            self.dataset = dataset.batch(self.p.batch_size)
        else:
            self.dataset = dataset
            self.sparse = any(
                [isinstance(t, tf.SparseTensorSpec) for t in self.dataset.element_spec]
            )

        # create model based on user input
        if model is None:
            self.model = self._get_model()
        else:
            self.model = model

        # setup callbacks
        self._setup_callbacks()

        # create loss based on user input
        self.loss = self._get_loss()

        # choose optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.p.learning_rate, clipvalue=1.0
        )

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

    def _print_save_message(self) -> None:
        """Prints a message, to inform user where model is saved."""
        _print_save_message(self)

    def _get_model(self) -> tf.keras.Model:
        """Constructs a model from an instance of this class."""
        return _get_model(self)

    def add_loss(self, loss):
        """Adds a new loss to the existing losses."""
        try:
            self.loss.append(loss(self.model, self.p))
        except TypeError:
            self.loss.append(loss(self.model))

    def add_callback(self, callback):
        """Adds a new callback to the existing callbacks."""
        try:
            self.callbacks.append(callback(self.p))
        except TypeError:
            self.callbacks.append(callback)

    def add_metric(self, metric):
        """Adds a new metric to the existing metrics."""
        self.metrics.append(metric(self.p))

    def add_images_to_tensorboard(self, *args: Any, **kwargs: Any) -> None:
        """Adds images of the latent space to tensorboard.

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
            scatter_kws (Optional[dict[str, Any]]): A dict with items that
                `plotly.express.scatter()` will accept. If None is provided,
                a dict with size 20 will be passed to
                `px.scatter(**{'size_max': 10, 'opacity': 0.2})`,
                which sets an appropriate size of scatter points for the size of
                datasets encodermap is usually used for.
            hist_kws (Optional[dict[str, Any]]): A dict with items that
                `encodermap.plot.plotting._plot_free_energy()` will accept. If None is provided a
                dict with bins 50 will be passed to
                `encodermap.plot.plotting._plot_free_energy(**{'bins': 50})`.
                You can choose a colormap here by providing `{'bins': 50, 'cmap':
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
            save_to_disk (bool): Whether to also write the images to disk.

        """
        _add_images_to_tensorboard(self, *args, **kwargs)

    def train(self) -> Optional[tf.keras.callbacks.History]:
        """Overwrites the parent class' `train()` method to implement references."""
        if all([v == 1 for k, v in self.p.__dict__.items() if "reference" in k]):
            self.train_for_references()
        else:
            print("References are already provided. Skipping reference training.")
        if self.p.current_training_step >= self.p.n_steps:
            print(
                f"This {self.__class__.__name__} instance has already been trained "
                f"for {self.p.current_training_step} steps. Increase the training "
                f"steps by calling `{self.__class__.__name__}.p.n_steps += new_steps` "
                f"and then call `{self.__class__.__name__}.train()` again."
            )
            return

        if self.p.training == "custom" and self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "custom" and not self.p.batched:
            raise NotImplementedError()
        elif self.p.training == "auto":
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
        if not self.read_only:
            self.p.save()
            self.save()

        return history

    def train_for_references(self, subsample: int = 100, maxiter: int = 500) -> None:
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
            nsteps = min(maxiter, max(1, int(self.trajs.n_frames / self.p.batch_size)))
        else:
            return
        # fmt: off
        costs = {
            "dihedral_cost": ["central_dihedrals", 1, dihedral_loss(self.model, p)],
            "angle_cost": ["central_angles", 0, angle_loss(self.model, p)],
            "cartesian_cost": ["central_cartesians", 2, cartesian_loss(self.model, parameters=p)],
        }
        # fmt: on
        # Local Folder Imports
        from ..models.models import ADCSparseFunctionalModel

        if isinstance(self.model, ADCSparseFunctionalModel):
            to_dense_models = {
                "dihedral_cost": self.model.get_dense_model_central_dihedrals,
                "angle_cost": self.model.get_dense_model_central_angles,
                "cartesian_cost": self.model.get_dense_model_distances,
            }

        cost_references = {key: [] for key in costs.keys()}
        for key, val in costs.items():
            if key in ["dihedral_cost", "angle_cost"]:
                inp = self.trajs.CVs[val[0]]
                if np.any(np.isnan(inp)):
                    inp = np_to_sparse_tensor(inp[::subsample])
                    inp = to_dense_models[key](inp).numpy()
                means = np.repeat(
                    np.expand_dims(
                        np.mean(inp, 0),
                        axis=0,
                    ),
                    repeats=self.p.batch_size,
                    axis=0,
                )
                costs[key].append(means)
            else:
                inp = self.trajs.CVs["central_distances"]
                if np.any(np.isnan(inp)):
                    inp = np_to_sparse_tensor(inp[::subsample])
                    inp = to_dense_models[key](inp).numpy()
                mean_lengths = np.expand_dims(np.mean(inp, axis=0), axis=0)
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
                for d in data:
                    if not isinstance(d, tf.sparse.SparseTensor):
                        if np.any(np.isnan(d)) and not self.sparse:
                            raise Exception(
                                f"Received data containing nans from `self.dataset` ({d=}),"
                                f"while `self.sparse` is set to True ({self.sparse=}). "
                                f"This training won't work as nans compromise the weights "
                                f"of the whole model. Try to explicitly set `sparse=True`, "
                                f"when instantiating the {self.__class__.__name__} class."
                            )
                for key, val in costs.items():
                    if key in ["dihedral_cost", "angle_cost"]:
                        if isinstance(data[val[1]], tf.sparse.SparseTensor):
                            d = to_dense_models[key](data[val[1]]).numpy()
                        else:
                            d = data[val[1]]
                        cost_references[key].append(val[2](d, val[3]).numpy())
                    if key == "cartesian_cost":
                        if isinstance(data[val[1]], tf.sparse.SparseTensor):
                            d = self.model.get_dense_model_cartesians(
                                data[val[1]]
                            ).numpy()
                            # un-flatten the cartesian coordinates
                            d = d.reshape(len(d), -1, 3)
                        else:
                            d = data[val[1]]
                        pd = pairwise_dist(
                            d[
                                :,
                                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
                            ],
                            flat=True,
                        )
                        c = val[2](val[3], pd).numpy()
                        cost_references["cartesian_cost"].append(c)
                pbar.update()
        s = {k: np.mean(v) for k, v in cost_references.items()}
        print(f"After {i} steps setting cost references: {s} to parameters.")
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

    def save(self, step: Optional[int] = None) -> None | Path:
        """Saves the model to the current path defined in `parameters.main_path`.

        Args:
            step (Optional[int]): Does not save the model at the given
                training step, but rather changes the string used for saving
                the model from a datetime format to another.

        Returns:
            Union[None, Path]: When the model has been saved, the Path will
                be returned. If the model could not be saved. None will be
                returned.

        """
        if not self.read_only:
            return save_model(
                self.model,
                self.p.main_path,
                inp_class_name=self.__class__.__name__,
                step=step,
                print_message=True,
            ).parent
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

    def set_train_data(self, data: TrajEnsemble) -> None:
        """Resets the train data for reloaded models."""
        (
            sparse,
            self.train_data,
            self.inp_CV_data,
        ) = self.get_train_data_from_trajs(data, self.p)
        self._using_hypercube = False
        self.p.using_hypercube = False
        if not self.sparse and sparse:
            print(
                f"The provided data contains nan's, but the model was trained "
                f"on dense input data."
            )
            return
        if not self.parameters.reconstruct_sidechains:
            data = [
                self.inp_CV_data["central_angles"],
                self.inp_CV_data["central_dihedrals"],
                self.inp_CV_data["central_cartesians"],
                self.inp_CV_data["central_distances"],
                self.inp_CV_data["side_dihedrals"],
            ]
        else:
            data = [
                self.inp_CV_data["central_angles"],
                self.inp_CV_data["central_dihedrals"],
                self.inp_CV_data["all_cartesians"],
                self.inp_CV_data["central_distances"],
                self.inp_CV_data["side_dihedrals"],
                self.inp_CV_data["side_angles"],
                self.inp_CV_data["side_dihedrals"],
                self.inp_CV_data["side_distances_"],
            ]
        dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
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
        try:
            out = plot_model(self.model, None)
            if out is not None:
                out.save(Path(self.p.main_path) / "network.png")
        except:
            pass

    @staticmethod
    def get_train_data_from_trajs(
        trajs: Union[TrajEnsemble, SingleTraj],
        p: ADCParameters,
        attr: str = "CVs",
        max_size: int = -1,
    ) -> tuple[bool, list[np.ndarray], dict[str, np.ndarray]]:
        """Builds train data from a `TrajEnsemble`.

        Args:
            trajs (TrajEnsemble): A `TrajEnsemble` instance.
            p (encodermap.parameters.ADCParameters): An instance of `encodermap.parameters.ADCParameters`.
            attr (str): Which attribute to get from `TrajEnsemble`. This defaults
                to 'CVs', because 'CVs' is usually a dict containing the CV data.
                However, you can build the train data from any dict in the `TrajEnsemble`.
            max_size (int): When you only want a subset of the CV data. Set this
                to the desired size.

        Returns:
            tuple: A tuple containing the following:
                - bool: A bool that shows whether some 'CV' values are `np.nan` (True),
                    which will be used to decide whether the sparse training
                    will be used.
                - list[np.ndarray]: An array of features fed into the autoencoder,
                    concatenated along the feature axis. The order of the
                    features is: central_angles, central_dihedral, (side_dihedrals
                    if p.use_sidechain_dihedrals is True).
                - dict[str, np.ndarray]: The training data as a dict. Containing
                    all values in `trajs.CVs`.

        """
        # Local Folder Imports
        from ..misc.misc import FEATURE_NAMES

        assert hasattr(trajs, attr), (
            f"Can't load train data from the attribute {attr}. "
            f"{trajs.__class__.__name__} has no attribute '{attr}'"
        )
        if not any([np.isnan(x).any() for x in getattr(trajs, attr).values()]):
            inp_CV_data = {
                key: val.astype("float32") for key, val in getattr(trajs, attr).items()
            }

            # squeeze, if xarray is provided
            if all([hasattr(v, "values") for v in inp_CV_data.values()]):
                inp_CV_data = {k: v.values.squeeze() for k, v in inp_CV_data.items()}
            sparse = False
        else:
            sparse = True

            # check whether the nans are correctly distributed
            for k, v in trajs.CVs.items():
                if k not in list(FEATURE_NAMES.values()):
                    continue
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

            # subsample if required
            if max_size > -1:
                for k, v in inp_CV_data.items():
                    idx = np.unique(
                        np.round(np.linspace(0, v.shape[0] - 1, max_size)).astype(int)
                    )
                    inp_CV_data[k] = v[idx]

            for k, v in inp_CV_data.items():
                if np.any(np.isnan(v)):
                    values = v
                    if k == "central_cartesians":
                        values = values.reshape(len(values), -1)
                    sparse_tensor = np_to_sparse_tensor(values)
                    inp_CV_data[k] = sparse_tensor

        if not p.reconstruct_sidechains:
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
                    "Cannot train model with central dihedrals and side "
                    "dihedrals only. Backbone angles are required."
                )
            # some checks for the length of the train data
            if p.model_api == "functional":
                if not p.use_backbone_angles and not p.use_sidechains:
                    pass
                elif p.use_backbone_angles and not p.use_sidechains:
                    assert len(train_data) == 2
                else:
                    assert len(train_data) == 3
        else:
            train_data = [
                inp_CV_data["central_angles"],
                inp_CV_data["central_dihedrals"],
                inp_CV_data["side_angles"],
                inp_CV_data["side_dihedrals"],
            ]
        return sparse, train_data, inp_CV_data

    @property
    def encoder(self) -> tf.keras.Model:
        """tf.keras.Model: The encoder Model."""
        return self.model.encoder_model

    @property
    def decoder(self) -> tf.keras.Model:
        """tf.keras.Model: The decoder Model."""
        return self.model.decoder_model

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
        self.metrics = []
        self.callbacks = []
        self.callbacks.append(ProgressBar(parameters=self.p))
        if not self.read_only:
            self.callbacks.append(CheckpointSaver(self.p))
        if self.p.tensorboard:
            self.tensorboard_write_bool: Union[TensorboardWriteBool, None] = (
                TensorboardWriteBool(self.p)
            )
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
        if self.p.cartesian_cost_scale_soft_start != (None, None):
            self.cartesian_increase_callback = IncreaseCartesianCost(self.p)
            self.callbacks.append(self.cartesian_increase_callback)
        if self.p.track_clashes:
            self.metrics.append(ADCClashMetric(parameters=self.p, distance_unit="nm"))
        if self.p.track_RMSD:
            self.metrics.append(ADCRMSDMetric(parameters=self.p))

    def _get_loss(self):
        loss = self.p.loss
        if loss == "reconstruction_loss":
            loss = reconstruction_loss(self.model)
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
            loss = [
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
                loss.append(self.side_dihedral_loss)
        elif loss == "mse":
            loss = "mse"
        else:
            raise ValueError(
                f"loss argument needs to be `reconstruction_loss`, `mse` or `emap_cost`. You provided '{loss}'."
            )
        return loss

    def encode(
        self,
        data: Optional[Union[TrajEnsemble, SingleTraj, Sequence[np.ndarray]]] = None,
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
        if data is None:
            if hasattr(self, "trajs"):
                data = self.trajs
            else:
                for data in self.dataset:
                    break
                if not self.p.use_sidechains and not self.p.use_backbone_angles:
                    data: np.ndarray = data[1]  # type: ignore[no-redef]
                elif self.p.use_backbone_angles and not self.p.use_sidechains:
                    data: list[np.ndarray] = [data[0], data[1]]  # type: ignore[no-redef]
                else:
                    data: list[np.ndarray] = [data[0], data[1], data[-1]]  # type: ignore[no-redef]

        if not hasattr(data, "trajs") and hasattr(data, "_CVs") and data is not None:
            traj: SingleTraj = data  # type: ignore[assignment]
            _, data, __ = self.get_train_data_from_trajs(traj, self.p, attr="_CVs")
            if isinstance(data, (tf.SparseTensor)):
                shape = data.dense_shape[0] * data.dense_shape[1]
            elif isinstance(data, np.ndarray):
                shape = data.size
                data = np.expand_dims(data, 0)
            elif isinstance(data[0], (np.ndarray, tf.Tensor)):
                shape = data[0].size
            elif isinstance(data[0], tf.sparse.SparseTensor):
                shape = data[0].values.shape[0]
            else:
                raise Exception(f"Unexpected datatype {data=}")
            if shape > 100_000:
                print(
                    f"Due to the size of the provided data {shape}, I "
                    f"need to chunk it, which takes longer. Sit back, grab a coffee..."
                )
                indices = np.split(
                    np.arange(traj.n_frames), np.arange(100, traj.n_frames, 100)
                )
                # single frame encoding does not work, because the frame axis is dropped
                if len(indices[-1]) == 1:
                    indices = np.split(
                        np.arange(traj.n_frames), np.arange(100, traj.n_frames, 101)
                    )
                lowd = []
                for i, ind in enumerate(indices):
                    _, data, __ = self.get_train_data_from_trajs(
                        traj[ind], self.p, attr="_CVs"
                    )
                    try:
                        lowd.append(self.model.encoder_model(data))
                    except IndexError as e:
                        raise Exception(f"{i=} {ind=} {data=}") from e
                return np.vstack(lowd)
            else:
                return self.encode(data)
        elif hasattr(data, "trajs"):
            lowd = []
            for traj in data.trajs:  # type: ignore[union-attr]
                lowd.append(self.encode(traj))
            return np.vstack(lowd)
        elif isinstance(data, Sequence):
            # Standard Library Imports
            from functools import reduce
            from operator import mul

            size = reduce(mul, data[0].shape)
            if size > 100_000:
                indices = np.split(
                    np.arange(data[0].shape[0]),
                    np.arange(100, data[0].shape[0], 100),
                )
                if len(indices[-1]) == 1:
                    indices = np.split(
                        np.arange(data[0].shape[0]),
                        np.arange(100, data[0].shape[0], 101),
                    )
                lowd = []
                for i, ind in enumerate(indices):
                    datum = []
                    for d in data:
                        if isinstance(d, tf.sparse.SparseTensor):
                            d = tf.sparse.to_dense(d, default_value=np.nan).numpy()[ind]
                            datum.append(np_to_sparse_tensor(d))
                        else:
                            datum.append(d[ind])
                    lowd.append(self.model.encoder_model(datum))
                return np.vstack(lowd)
            else:
                return self.model.encoder_model(data).numpy()  # type: ignore[no-any-return]
        elif hasattr(data, "shape") or hasattr(data, "dense_shape"):
            return self.model.encoder_model(data).numpy()  # type: ignore[no-any-return]
        else:
            raise TypeError(f"Wrong type for argument `data`: {type(data)=} {data=}.")

    def decode(self, data: np.ndarray) -> Sequence[np.ndarray]:
        """Calls the decoder part of the model.

        `AngleDihedralCartesianAutoencoder` will, like the other two classes'
        output a list of np.ndarray.

        Args:
            data (np.ndarray): The data to be passed to the decoder part of
                the model. Make sure that the shape of the data matches the
                number of neurons in the latent space.

        Returns:
            Union[list[np.ndarray], np.ndarray]: Outputs from the decoder part.
                For `AngleDihedralCartesianEncoderMap`, this will be a list of
                np.ndarray.

        """
        out = self.model.decoder(data)
        if isinstance(out, (list, tuple)):
            out = [o.numpy() for o in out]
        else:
            out = out.numpy()
        return out  # type: ignore[no-any-return]

    @overload
    def generate(
        self,
        points: np.ndarray,
        top: Optional[Union[str, int, Topology]],
        backend: Literal["mdtraj"],
        progbar: Optional[Any],
    ) -> Trajectory: ...

    @overload
    def generate(
        self,
        points: np.ndarray,
        top: Optional[Union[str, int, Topology]],
        backend: Literal["mdanalysis"],
        progbar: Optional[Any],
    ) -> Universe: ...

    def generate(
        self,
        points: np.ndarray,
        top: Optional[Union[str, int, Topology]] = None,
        backend: Literal["mdtraj", "mdanalysis"] = "mdtraj",
        progbar: Optional[Any] = None,
    ) -> Union[Universe, Trajectory]:
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
        if isinstance(out, np.ndarray):
            dihedrals = out
            sidechain_dihedrals = None
        elif (
            isinstance(out, (list, tuple))
            and len(out) == 2
            and all([isinstance(a, np.ndarray) for a in out])
        ):
            angles, dihedrals = out
            sidechain_dihedrals = None
        elif (
            isinstance(out, (list, tuple))
            and len(out) == 3
            and all([isinstance(a, np.ndarray) for a in out])
        ):
            angles, dihedrals, sidechain_dihedrals = out
        else:
            raise Exception(
                f"Unexpected length of out detected: ({len(out)}=). Maybe also "
                f"unexpected dtypes: ({[type(a) for a in out]=})."
            )

        assert isinstance(self.trajs, TrajEnsemble), (
            f"`generate()` can only work, when the Autoencoder was built with a "
            f"`TrajEnsemble` and not just a dataset. You can set the `TrajEnsemble` "
            f"of this object with `AngleDihedralCartesianEncoderMap.trajs = TrajEnsemble`."
        )

        if top is None:
            if len(self.trajs.top) > 1:
                print(
                    f"Please specify which topology you would like to use for generating "
                    f"conformations. You can either provide a `str` to a topology file "
                    f"(file extension .pdb, .h5, .gro) on disk, or a `int` specifying the "
                    f"one of the ensembles {len(self.trajs.top)} topologies "
                    f"(see `AngleDihedralCartesianEncoderMap.trajs.top` for available "
                    f"topologies). You can also directly supply a "
                    f"you can also specify a `mdtraj.Topology` object."
                )
                return  # type: ignore[return-value]
        elif isinstance(top, int):
            mdanalysis_traj = self.trajs[top][0].traj
        elif isinstance(top, str) and top not in self.trajs.common_str:
            mdanalysis_traj = md.load(top)
        elif isinstance(top, str) and top in self.trajs.common_str:
            mdanalysis_traj = self.trajs.trajs_by_common_str[top][0].traj
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
            traj = mdtraj_backmapping(
                top=top,
                dihedrals=dihedrals,
                sidechain_dihedrals=sidechain_dihedrals,
                trajs=self.trajs,
                progbar=progbar,
            )  # type: ignore[call-overload]
            return traj
        else:
            raise TypeError(
                f"backend must be 'mdtraj' or 'mdanalysis', but you provided {backend}"
            )
