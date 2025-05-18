# -*- coding: utf-8 -*-
# encodermap/callbacks/callbacks.py
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
"""Callbacks to strew into the Autoencoder classes."""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

# Third Party Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm

# Encodermap imports
from encodermap.callbacks.metrics import rmsd_tf
from encodermap.misc.saving_loading_models import save_model
from encodermap.misc.summaries import image_summary
from encodermap.parameters.parameters import ADCParameters, AnyParameters, Parameters


################################################################################
# Globals
################################################################################


__all__: list[str] = [
    "ProgressBar",
    "EarlyStop",
    "CheckpointSaver",
    "TensorboardWriteBool",
    "IncreaseCartesianCost",
    "NoneInterruptCallback",
    "ImageCallback",
    "EncoderMapBaseCallback",
]


################################################################################
# Helpers
################################################################################


def np_to_sparse_tensor(a: np.ndarray) -> tf.sparse.SparseTensor:
    indices = np.stack(np.where(~np.isnan(a))).T.astype("int64")
    dense_shape = a.shape
    a = a[~np.isnan(a)].flatten()
    return tf.sparse.SparseTensor(indices, a, dense_shape)


################################################################################
# Public Classes
################################################################################


class NoneInterruptCallback(tf.keras.callbacks.Callback):
    """A callback that interrupts training, when NaN is encountered in weights."""

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Gets the current loss at the end of the batch compares it to previous batches."""
        for w in self.model.get_weights():
            if not isinstance(w, np.ndarray):
                continue
            if np.any(np.isnan(w)):
                print(
                    f"At batch {self.model._my_train_counter.numpy()}, the "
                    f"model has NaNs in one of its weights. Because "
                    f"multiplication with NaN yields NaN, this NaN value will now "
                    f"propagate through the network until all weights are tensors of "
                    f"NaNs. I stopped the training at this point, as further training is "
                    f"pointless. This error might originate from your input. You "
                    f"can run the training with `deterministic=True` and check "
                    f"whether this problems happens at the same training step "
                    f"for multiple trainings. If yes, your input contains NaNs. "
                    f"If no, you can try to lower the learning rate."
                )
                self.model.stop_training = True
                break


class EncoderMapBaseCallback(tf.keras.callbacks.Callback):
    """Base class for callbacks in EncoderMap.

    The `Parameters` class in EncoderMap has a `summary_step` variable that
    dictates when variables and other tensors are logged to TensorBoard. No
    matter what property is logged there will always be a code section
    executing a `if train_step % summary_step == 0` code snippet. This is
    handled centrally in this class. This class is instantiated inside the
    user-facing `AutoEncoderClass` classes and is provided with the appropriate
    parameters (`Parameters` for `EncoderMap` and `ADCParameters` for
    `AngleDihedralCartesianEncoderMap`). Thus, subclassing this class does not
    need to implement a new `__init__` method. Only the `on_summary_step` and
    the `on_checkpoint_step` methods need to be implemented for sub-classes
    if this class with code that should happen when these events happen.

     Examples:

         In this example, the `on_summary_step` method causes an exception.

        >>> from typing import Optional
        >>> import encodermap as em
        ...
        >>> class MyCallback(em.callbacks.EncoderMapBaseCallback):
        ...     def on_summary_step(self, step: int, logs: Optional[dict] = None) -> None:
        ...         raise Exception(f"Summary step {self.steps_counter} has been reached.")
        ...
        >>> emap = em.EncoderMap()  # doctest: +ELLIPSIS
        Output...
        >>> emap.add_callback(MyCallback)
        >>> emap.train()  # doctest: +ELLIPSIS, +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        Exception: Summary step 10 has been reached.


    Attributes:
        steps_counter (int): The current step counter. Increases every `on_train_batch_end`.
        p (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters]:
            The parameters for this callback. Based on the `summary_step` and
            `checkpoint_step` of the `encodermap.parameters.Parameters` class different
            class-methods are called.

    """

    def __init__(self, parameters: Optional["AnyParameters"] = None) -> None:
        """Instantiate the EncoderMapBaseCallback class.

        Args:
            parameters (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters, None], optional):
                Parameters that will be used to print out the progress bar. If None is passed
                default values (check them with print(em.ADCParameters.defaults_description())) will be used.
                Defaults to None.

        """
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters
        super().__init__()
        self.steps_counter = 0

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Called after a batch ends. The number of batch is provided by keras.

        This method is the backbone of all of EncoderMap's callbacks. After
        every batch is method is called by keras. When the number of that
        batch matches either `encodermap.Parameters.summary_step` or `encodermap.Parameters.checkpoint_step`
        the code on `self.on_summary_step`, or `self.on_checkpoint_step` is
        executed. These methods should be overwritten by child classes.

        Args:
            batch (int): The number of the current batch. Provided by keras.
            logs (Optional[dict]): `logs` is a dict containing the metrics results.

        """
        self.steps_counter += 1
        if self.steps_counter % self.p.checkpoint_step == 0:
            self.on_checkpoint_step(self.steps_counter, logs=logs)
        if self.steps_counter % self.p.summary_step == 0:
            self.on_summary_step(self.steps_counter, logs=logs)

    def on_summary_step(self, step: int, logs: Optional[dict] = None) -> None:
        """Executed, when the currently finished batch matches `encodermap.Parameters.summary_step`

        Args:
            step (int): The number of the current step.
            logs (Optional[dict]): `logs` is a dict containing the metrics results.

        """
        pass

    def on_checkpoint_step(self, step: int, logs: Optional[dict] = None) -> None:
        """Executed, when the currently finished batch matches `encodermap.Parameters.checkpoint_step`

        Args:
            step (int): The number of the current step.
            logs (Optional[dict]): `logs` is a dict containing the metrics results.

        """
        pass


##############################################################################
# Public Classes
##############################################################################


class EarlyStop(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience (int): Number of epochs to wait after min has been hit. After this
            number of no improvement, training stops.

    """

    def __init__(self, patience: int = 0) -> None:
        """Instantiate the `EarlyStop` class.

        Args:
            patience (int): Number of training steps to wait after min has been hit.
            Training is halted after this number of steps without improvement.

        """
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Sets some attributes at the beginning of training."""
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_batch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Gets the current loss at the end of the batch compares it to previous batches."""
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_batch = batch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Prints a message after training, if an early stop occured."""
        if self.stopped_batch > 0:
            print("Step %05d: early stopping" % (self.stopped_batch + 1))


class ProgressBar(EncoderMapBaseCallback):
    """Progressbar Callback. Mix in with model.fit() and make sure to set verbosity to zero."""

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Simply creates the progressbar once training starts."""
        self.pbar = tqdm(
            total=self.p.n_steps,
            initial=self.p.current_training_step,
            position=0,
            leave=True,
        )
        postfix = {f"Loss after step ?": "?"}
        if isinstance(self.p, ADCParameters):
            postfix["Cartesian cost scale"] = "?"
        self.pbar.set_postfix(postfix)

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Overwrites the parent class' `on_train_batch_end` and adds a progress-bar update."""
        super().on_train_batch_end(batch, logs=logs)
        self.pbar.update()

    def on_summary_step(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Update the progress bar after an epoch with the current loss.

        Args:
            epoch(int): Current epoch. Will be automatically passed by tensorflow.
            logs (Optional[dict]): Also automatically passed by tensorflow.
                Contains metrics and losses. logs['loss'] will be written to the progress bar.

        """
        if logs != {}:
            postfix = {f"Loss after step {epoch}": logs["loss"]}
        epoch += self.p.current_training_step
        if isinstance(self.p, ADCParameters):
            if self.p.cartesian_cost_scale_soft_start != (None, None):
                if self.p.cartesian_cost_scale is not None:
                    if (
                        self.p.cartesian_cost_scale_soft_start[0] is None
                        or epoch is None
                    ):
                        scale = self.p.cartesian_cost_scale
                    else:
                        a, b = self.p.cartesian_cost_scale_soft_start
                        if epoch < a:
                            scale = 0
                        elif a <= epoch <= b:
                            scale = self.p.cartesian_cost_scale / (b - a) * (epoch - a)
                        else:
                            scale = self.p.cartesian_cost_scale
                else:
                    scale = 0
            else:
                scale = self.p.cartesian_cost_scale
            postfix["Cartesian cost scale"] = np.round(scale, 2)
        self.pbar.set_postfix(postfix)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Close the Progress Bar"""
        self.pbar.close()


class ImageCallback(tf.keras.callbacks.Callback):
    """Writes images to tensorboard."""

    def __init__(
        self,
        parameters: AnyParameters,
        highd_data: np.ndarray,
        image_step: int,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        mpl_scatter_kws: Optional[dict] = None,
        mpl_hist_kws: Optional[dict] = None,
        plotly_scatter_kws: Optional[dict] = None,
        plotly_hist_kws: Optional[dict] = None,
        additional_fns: Optional[list[Callable]] = None,
        when: Literal["batch", "epoch"] = "batch",
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """Instantiate the ImageCallback.

        This class uses `encodermap.misc.summaries` to plot images to tensorboard.

        Args:
            highd_data (np.ndarray): The high-dimensional data, that will be provided
                to the plotting functions.
            image_step (int): When to plot the images. This argument is combined
                with the `when` argument to either plot every "epoch" % `image_step` == 0
                or every "batch" % `image_step` == 0 steps.
            backend (Literal["matplotlib", "plotly"]: Which backend to use for
                plotting. Defaults to "matplotlib".
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
            additional_fns (Optional[list[Callabe]]): Can be None or a list
                of functions, that return `io.BytesIO()` buffered images (see
                Example).
            when (Literal["batch", "epoch"]): When to plot the images. Works in
                conjunction with the argument `image_step`.
            save_dir (Optional[Union[str, Path]]): When specified, images are
                saved to the specified directory during training.

        Here's an example of how to use this class:
        .. code-block:: python
            def return_hist(data, hist_kws):
                plt.close("all")
                matplotlib.use("Agg")  # overwrites current backend of notebook
                plt.figure()
                plt.hist2d(*data.T, **hist_kws)
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), 4)
                image = tf.expand_dims(image, 0)
                return image

        """
        super().__init__()
        self.backend = backend
        self.p = parameters
        self.highd_data = highd_data
        self.image_step = image_step
        if self.backend == "matplotlib":
            scatter_kws = mpl_scatter_kws
            hist_kws = mpl_hist_kws
            if scatter_kws is None:
                scatter_kws = {"s": 20}
            if hist_kws is None:
                hist_kws = {"bins": 50}
        elif self.backend == "plotly":
            scatter_kws = plotly_scatter_kws
            hist_kws = plotly_hist_kws
            if scatter_kws is None:
                scatter_kws = {"size_max": 1, "opacity": 0.2}
            if hist_kws is None:
                hist_kws = {"bins": 50}
        else:
            raise Exception(
                f"Argument `backend` must be either 'plotly' or 'matplotlib'."
            )
        self.scatter_kws = scatter_kws
        self.hist_kws = hist_kws
        self.additional_fns = additional_fns
        self.when = when
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = Path(save_dir)

    def get_lowd(self):
        if isinstance(self.highd_data, (list, tuple)):
            if self.highd_data[0].shape[0] * self.highd_data[0].shape[1] > 100_000:
                indices = np.split(
                    np.arange(self.highd_data[0].shape[0]),
                    np.arange(100, self.highd_data[0].shape[0], 100),
                )
                if len(indices[-1]) == 1:
                    indices = np.split(
                        np.arange(self.highd_data[0].shape[0]),
                        np.arange(100, self.highd_data[0].shape[0], 101),
                    )
                lowd = []
                for i, ind in enumerate(indices):
                    data = []
                    for d in self.highd_data:
                        if isinstance(d, tf.sparse.SparseTensor):
                            d = tf.sparse.to_dense(d, default_value=np.nan).numpy()[ind]
                            data.append(np_to_sparse_tensor(d))
                        else:
                            data.append(d[ind])
                    lowd.append(self.model.encoder(data).numpy())
                return np.vstack(lowd)
        lowd = self.model.encoder(self.highd_data).numpy()
        return lowd

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Calls `encodermap.misc.summaries.image_summary` on epoch end."""
        if self.when == "epoch":
            if self.image_step != 0:
                if epoch % self.image_step != 0:
                    return
            lowd = self.get_lowd()
            assert lowd is not None
            image_summary(
                lowd=lowd,
                step=epoch,
                scatter_kws=self.scatter_kws,
                hist_kws=self.hist_kws,
                additional_fns=self.additional_fns,
                backend=self.backend,
            )
            if self.save_dir is not None:
                self.save_image_to_dir(lowd, epoch)
            if isinstance(self.p, ADCParameters):
                if self.p.track_RMSD and self.save_dir is not None:
                    rmsds = self.model.compiled_metrics._metrics[-1].result().numpy()
                    np.save(self.save_dir / f"rmsds_epoch_{epoch}.npy", rmsds)

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Calls `encodermap.misc.summaries.image_summary` on batch end."""
        if self.when == "batch":
            if self.image_step != 0:
                if batch % self.image_step != 0:
                    return
            lowd = self.get_lowd()
            assert lowd is not None
            image_summary(
                lowd=lowd,
                step=batch,
                scatter_kws=self.scatter_kws,
                hist_kws=self.hist_kws,
                additional_fns=self.additional_fns,
                backend=self.backend,
            )
            if self.save_dir is not None:
                self.save_image_to_dir(lowd, batch)
            if isinstance(self.p, ADCParameters):
                if self.p.track_RMSD and self.save_dir is not None:
                    rmsds = self.model.compiled_metrics._metrics[-1].result().numpy()
                    np.save(self.save_dir / f"rmsds_batch_{batch}.npy", rmsds)

    def save_image_to_dir(self, lowd: np.ndarray, step: int) -> None:
        """Saves the lowd representation to disk, so it can be looked at later."""
        outfile = self.save_dir / f"{self.when}_{step}.png"
        if not np.any(np.isnan(lowd)):
            plt.close("all")
            matplotlib.use("Agg")  # overwrites current backend of notebook
            plt.figure()
            plt.hist2d(*lowd.T, **self.hist_kws)
            plt.savefig(outfile, format="png")
        outfile = outfile.with_suffix(".npy")
        np.save(outfile, lowd)


class CheckpointSaver(EncoderMapBaseCallback):
    """Callback that saves an `encodermap.models` model."""

    def on_checkpoint_step(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Overwrites parent class' `on_checkpoint_step` method.

        Uses `encodermap.misc.saving_loading_models.save_model` to save the model.
        Luckily, the keras callbacks contain the model as an attribute (self.model).

        """
        save_model(self.model, self.p.main_path, step=epoch)


class IncreaseCartesianCost(tf.keras.callbacks.Callback):
    """Callback for the `enocdermap.autoencoder.AngleDihedralCarteisanEncoderMap`.

    This callback implements the soft-start of the cartesian cost.

    """

    def __init__(
        self,
        parameters: Optional[ADCParameters] = None,
    ) -> None:
        """Instantiate the callback.

        Args:
            parameters (Optional[encodermap.parameters.ADCParameters]: Can be either None, or an instance
                of `encodermap.parameters.ADCParameters`. These parameters define the
                steps at which the cartesian cost scaling factor needs to be adjusted.
                If None is provided, the default values `(None, None)`, i.e. no
                cartesian cost, will be used. Defaults to None.
            start_step (int): The current step of training. This argument
                is important is training is stopped using the scaling cartesian
                cost. This argument will usually be loaded from a file in the saved model.

        """
        if parameters is None:
            self.p = ADCParameters()
        else:
            self.p = parameters
        super().__init__()
        self.a, self.b = self.p.cartesian_cost_scale_soft_start
        self.last_compilation: bool = False
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.p.learning_rate, clipvalue=1.0
        )
        # use a instance variable for the case the model is reloaded and re-trained.
        self.current_step = self.p.current_training_step
        self.current_cartesian_cost_scale = K.variable(
            0.0, dtype="float32", name="current_cartesian_cost_scale"
        )
        K.set_value(
            self.current_cartesian_cost_scale,
            self.calc_current_cartesian_cost_scale(self.current_step),
        )

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None):
        "Sets the value of the keras backend variable `self.current_cartesian_cost_scale`"
        self.current_step += 1
        K.set_value(
            self.current_cartesian_cost_scale,
            self.calc_current_cartesian_cost_scale(self.current_step),
        )

    def calc_current_cartesian_cost_scale(self, epoch):
        """Calculates the current cartesian distance scale, based on the parameters
        `self.a`, `self.b` `self.p.cartesian_cost_scale`.
        """
        assert isinstance(epoch, int)
        if self.p.cartesian_cost_scale is not None:
            if self.p.cartesian_cost_scale_soft_start[0] is None or epoch is None:
                scale = self.p.cartesian_cost_scale
            else:
                if epoch < self.a:
                    scale = 0.0
                elif self.a <= epoch <= self.b:
                    scale = (
                        self.p.cartesian_cost_scale
                        / (self.b - self.a)
                        * (epoch - self.a)
                    )
                else:
                    scale = self.p.cartesian_cost_scale
        else:
            scale = 0.0
        # scale = K.variable(scale, dtype='float32', name='current_cartesian_cost_scale')
        return scale


class TensorboardWriteBoolAlwaysFalse(tf.keras.callbacks.Callback):
    """A tensorboard callback, that is always False. Used for debugging."""

    def __init__(self) -> None:
        """Instantiate this class."""
        self.log_bool = K.variable(False, bool, "log_scalar")
        K.set_value(self.log_bool, K.variable(False, bool, "log_scalar"))


class TensorboardWriteBool(tf.keras.callbacks.Callback):
    """This class saves the value of the keras variable `log_bool`.

    Based on this variable, stuff will be written to tensorboard, or not.

    """

    def __init__(self, parameters: Optional["AnyParameters"] = None) -> None:
        """Instantiate the class.

        Args:
            parameters (Union[encodermap.parameters.Parameters, encodermap.parameters.ADCParameters, None], optional):
                Parameters that will be used to check when data should be written to tensorboard. If None is passed
                default values (check them with print(em.ADCParameters.defaults_description())) will be used.
                Defaults to None.

        """
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters
        super().__init__()
        self.log_bool = K.variable(False, bool, "log_scalar")
        K.set_value(self.log_bool, K.variable(False, bool, "log_scalar"))
        self.current_training_step = 0

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        """Sets the value of the keras backend variable `log_bool`.

        This method does not use the `batch` argument because the variable
        `self.current_training_step` is used.

        """
        self.current_training_step += 1
        if self.p.tensorboard:
            if self.current_training_step % self.p.summary_step == 0:
                K.set_value(self.log_bool, K.variable(True, bool, "log_scalar"))
            else:
                K.set_value(self.log_bool, K.variable(False, bool, "log_scalar"))
