# -*- coding: utf-8 -*-
# encodermap/models/models.py
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
"""

"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os
import warnings
from collections.abc import Iterable, Sequence
from math import pi
from typing import TYPE_CHECKING, Any, Literal, Optional, Type, TypeVar, Union, overload

# Third Party Imports
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda

# Encodermap imports
from encodermap.encodermap_tf1.backmapping import (
    chain_in_plane,
    dihedrals_to_cartesian_tf,
)
from encodermap.loss_functions.loss_classes import testing
from encodermap.misc.distances import pairwise_dist
from encodermap.misc.summaries import add_layer_summaries
from encodermap.models.layers import (
    BackMapLayer,
    BackMapLayerTransformations,
    BackMapLayerWithSidechains,
    MeanAngles,
    PairwiseDistances,
    PeriodicInput,
    PeriodicOutput,
)
from encodermap.parameters.parameters import ADCParameters, AnyParameters, Parameters
from encodermap.trajinfo.info_single import Capturing


################################################################################
# Typing
################################################################################


SequentialModelType = TypeVar(
    "SequentialModelType",
    bound="SequentialModel",
)
ADCSparseFunctionalModelType = TypeVar(
    "ADCSparseFunctionalModelType",
    bound="ADCSparseFunctionalModel",
)
ADCFunctionalModelTestingType = TypeVar(
    "ADCFunctionalModelTestingType",
    bound="ADCFunctionalModelTesting",
)
ADCFunctionalModelType = TypeVar(
    "ADCFunctionalModelType",
    bound="ADCFunctionalModel",
)
ADCFunctionalModelSidechainReconstructionType = TypeVar(
    "ADCFunctionalModelSidechainReconstructionType",
    bound="ADCFunctionalModelSidechainReconstruction",
)
ADCFunctionalModelInputType = Union[
    tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
]


################################################################################
# Globals
################################################################################


__all__: list[str] = ["gen_sequential_model", "gen_functional_model"]


################################################################################
# Helper Classes
################################################################################


class MyKernelInitializer(tf.keras.initializers.Initializer):
    """Custom Kernel initializer to make weights deterministic.

    Gets a numpy array called weights. When called, it checks whether the requested
    shape matches the shape of the numpy array and then returns the array.
    For example, see the documentation of `MyBiasInitializer`.

    """

    def __init__(
        self,
        weights: np.ndarray,
    ) -> None:
        """Initialize the KernelInitializer.

        Args:
            weights (np.ndarray): The weights which will be returned when the initializer
                is called.

        """
        self.weights = weights

    def __call__(
        self,
        shape: Sequence[int],
        dtype: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the weight as a float32 numpy array.

        Returns:
            np.ndarray: The weight.

        """
        assert tuple(shape) == self.weights.shape, (
            f"Can't initialize Kernel. Requested shape: {tuple(shape)} shape "
            f"of pre-set weights: {self.weights.shape}"
        )
        return self.weights.astype("float32")


class MyBiasInitializer(tf.keras.initializers.Initializer):
    """Custom Bias initializer to make bias deterministic.

    Gets a numpy array called bias. When called, it checks whether the requested
    shape matches the shape of the numpy array and then returns the array.

    Examples:
        >>> # Imports
        >>> from encodermap.models.models import MyBiasInitializer
        >>> import numpy as np
        >>> import tensorflow as tf
        >>> from tensorflow import keras
        >>> from tensorflow.keras import layers
        ...
        >>> # Create a model with the bias initializer
        >>> model = tf.keras.models.Sequential(
        ...     [
        ...         layers.Dense(
        ...             2,
        ...             activation="relu",
        ...             name="layer1",
        ...             bias_initializer=MyBiasInitializer(np.array([1.0, 0.5])),
        ...         ),
        ...         layers.Dense(
        ...             3,
        ...             activation="relu",
        ...             name="layer2",
        ...             bias_initializer=MyBiasInitializer(np.array([0.1, 0.2, 0.3])),
        ...         ),
        ...         layers.Dense(4, name="layer3"),
        ...     ]
        ... )
        ...
        >>> model.build(input_shape=(10, 2))
        >>> for layer in model.layers:
        ...     print(layer.get_weights()[1])
        [1.  0.5]
        [0.1 0.2 0.3]
        [0. 0. 0. 0.]
        >>> # This example fails with an AssertionError, because the
        >>> # bias shape of the second layer is wrong:
        >>> model = tf.keras.models.Sequential(
        ...     [
        ...         layers.Dense(
        ...             2,
        ...             activation="relu",
        ...             name="layer1",
        ...             bias_initializer=MyBiasInitializer(np.array([1.0, 0.5])),
        ...         ),
        ...         layers.Dense(
        ...             3,
        ...             activation="relu",
        ...             name="layer2",
        ...             bias_initializer=MyBiasInitializer(np.array([0.1, 0.2])),
        ...         ),
        ...         layers.Dense(4, name="layer3"),
        ...     ]
        ... )
        ...
        >>> model.build(input_shape=(10, 2))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        AssertionError: Can't initialize Bias. Requested shape: (3,) shape of pre-set bias: (2,)

    """

    def __init__(
        self,
        bias: np.ndarray,
    ) -> None:
        """Initialize the BiasInitializer.

        Args:
            bias (np.ndarray): The bias which will be returned when the initializer
                is called.

        """
        self.bias = bias

    def __call__(
        self,
        shape: Sequence[int],
        dtype: Optional[Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the bias as a float32 numpy array.

        Returns:
            np.ndarray: The bias.

        """
        assert tuple(shape) == self.bias.shape, (
            f"Can't initialize Bias. Requested shape: {tuple(shape)} shape "
            f"of pre-set bias: {self.bias.shape}"
        )
        return self.bias.astype("float32")


################################################################################
# Public Functions
################################################################################


def gen_sequential_model(
    input_shape: int,
    parameters: Optional[AnyParameters] = None,
    sparse: bool = False,
) -> SequentialModel:
    """Returns a tf.keras model with the specified input shape and the
    parameters in the Parameters class.

    Args:
        input_shape (int): The input shape of the returned model. In most cases
            that is data.shape[1] of your data.
        parameters (Optional[AnyParameters]): The parameters to use on the
            returned model. If None is provided the default parameters in
            encodermap.Parameters.defaults is used. You can look at the defaults
            with print(em.Parameters.defaults_description()). Defaults to None.
        sparse (bool): Whether sparse inputs are expected. Defaults to False.

    Returns:
        em.SequentialModel: A subclass of tf.keras.Model build with specified parameters.

    """
    if parameters is None:
        parameters = Parameters()
    else:
        if isinstance(parameters, Parameters):
            return SequentialModel(input_shape, parameters, sparse=sparse)
        elif isinstance(parameters, ADCParameters):
            return ADCSequentialModel(input_shape, parameters)
        else:
            p = parameters
            raise TypeError(
                f"parameters need to be ecodermap.Parameters or encodermap.ADCParameters. You supplied {type(p)}"
            )


def _get_deterministic_random_normal(
    mean: float = 0.1,
    stddev: float = 0.05,
    seed: Optional[int] = None,
) -> tf.compat.v1.random_normal_initializer:
    """Returns a deterministic random_normal_initializer wit tensorflow1.

     For the tf2 implementation, look into `MyKernelInitializer`.
     Moving from tf1 to tf2, the seeding method has changed, so that the same
    seed can't be used to get the same random data in tf1 and tf2.

    """
    # Third Party Imports
    import tensorflow.compat.v1 as tf

    return tf.random_normal_initializer(mean, stddev, seed=seed)


def _get_deterministic_variance_scaling(
    seed: Optional[int] = None,
) -> tf.compat.v1.variance_scaling_initializer:
    """Returns a deterministic variance_scaling_initializer wit tensorflow1.

    For the tf2 implementation, look into `MyBiasInitializer`.
    Moving from tf1 to tf2, the seeding method has changed, so that the same
    seed can't be used to get the same random data in tf1 and tf2.

    """
    # Third Party Imports
    import tensorflow.compat.v1 as tf

    return tf.variance_scaling_initializer(seed=seed)


@overload
def gen_functional_model(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[tuple[int], tuple[int], tuple[int, int], tuple[int], tuple[int]],
    ],
    parameters: Optional[ADCParameters] = None,
    sparse: bool = False,
    sidechain_only_sparse: bool = False,
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = True,
    use_experimental_model: bool = True,
) -> ADCFunctionalModelTesting: ...


@overload
def gen_functional_model(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[tuple[int], tuple[int], tuple[int, int], tuple[int], tuple[int]],
    ],
    parameters: Optional[ADCParameters] = None,
    sparse: bool = False,
    sidechain_only_sparse: bool = False,
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = True,
    use_experimental_model: bool = False,
) -> ADCFunctionalModel: ...


@overload
def gen_functional_model(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[tuple[int], tuple[int], tuple[int], tuple[int], tuple[int]],
    ],
    parameters: Optional[ADCParameters] = None,
    sparse: bool = False,
    sidechain_only_sparse: bool = False,
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = True,
    use_experimental_model: bool = False,
) -> ADCSparseFunctionalModel: ...


def gen_functional_model(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[
            tuple[int],
            tuple[int],
            Union[tuple[int, int], tuple[int]],
            tuple[int],
            tuple[int],
        ],
    ],
    parameters: Optional[ADCParameters] = None,
    sparse: bool = False,
    sidechain_only_sparse: bool = False,
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = True,
    use_experimental_model: bool = False,
) -> Union[ADCSparseFunctionalModel, ADCFunctionalModel, ADCFunctionalModelTesting]:
    """New implementation of the functional model API for AngleCartesianDihedralEncoderMap

    The functional API is much more flexible than the sequential API, in that
    models with multiple inputs and outputs can be defined. Custom layers and
    submodels can be intermixed. In EncoderMap's case, the functional API is used to
    build the AngleDihedralCartesianAutoencoder, which takes input data in form
    of a tf.data.Dataset with:
        * backbone_angles (angles between C, CA, N - atoms in the backbone).
        * backbone_torsions (dihedral angles in the backbone,
            commonly known as omega, phi, psi).
        * cartesian_coordinates (coordinates of the C, CA, N backbone
            atoms. This data has ndim 3, the other have ndim 2).
        * backbone_distances (distances between the C, CA, N backbone atoms).
        * sidechain_torsions (dihedral angles in the sidechain,
            commonly known as chi1, chi2, chi3, chi4, chi5).
    Packing and unpacking that data in the correct order is important.
    Make sure to double-check whether you are using angles or dihedrals.
    A simple print of the shape can be enough.

    Args:
        input_shapes(Union[tf.data.Dataset, tuple[int, int, int, int, int]]):
            The input shapes, that will be used in the construction of the model.
        parameters (Optional[encodermap.parameters.ADCParameters]): An instance
            of `encodermap.parameters.ADCParameters`,
            which holds further parameters in network construction. If None
            is provided, a new instance with default parameters will be
            created. Defaults to None.
        sparse (bool): Whether sparse inputs are expected. Defaults to False.
        sidechain_only_sparse (bool): A special case, when the proteins have
            the same number of residues, but different numbers of sidechain
            dihedrals. In that case only the sidechain dihedrals are considered
            to be sparse. Defaults to False.
        kernel_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "VarianceScaling", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "VarianceScaling" is
            provided, the weights will be initialized with `tf.keras.initializers.
            VarianceScaling()`. Defaults to "VarianceScaling". If "deterministic"
            is provided, a seed will be used with VarianceScaling. If a dict with
            weight matrices is supplied, the keys should follow this naming con-
            vention: ["dense/kernel", "dense_1/kernel", "dense_2/kernel", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        bias_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "RandomNormal", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "RandomNormal" is
            provided, the weights will be initialized with `tf.keras.initializers.
            RandomNormal(0.1, 0.05)`. Defaults to "RandomNormal". If "deterministic"
            is provided, a seed will be used with RandomNormal. If a dict with
            bias matrices is supplied, the keys should follow this naming con-
            vention: ["dense/bias", "dense_1/bias", "dense_2/bias", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        write_summary (bool): Whether to print a summary. If p.tensorboard is True
            a file will be generated. at the main_path.

    Returns:
        tf.keras.models.Model: The model.


    Here's a scheme of the generated network::

        ┌───────────────────────────────────────────────────────────────────────────────────────┐
        │A linear protein with N standard residues has N*3 backbone atoms (..C-N-CA-C-N..)      │
        │it has N*3 - 1 distances between these atoms                                           │
        │it has N*3 - 2 angles between three atoms                                              │
        │it has N*3 - 3 dihedrals between 4 atoms                                               │
        │it has S sidechain dihedrals based on the sequence                                     │
        └───────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬───────┘
                │                 │                 │                 │                 │
                │                 │                 │                 │                 │
                │                 │                 │                 │                 │
        ┌───────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐
        │cartesians     │ │distances      │ │angles         │ │dihedrals      │ │side dihedrals │
        │(batch, N*3, 3)│ │(batch, N*3-1) │ │(batch, N*3-2) │ │(batch, N*3-3) │ │(batch, S)     ├───────┐
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └────────┬──────┘       │Every type
                │                 │                 │                 │                  │              │of angular
                │                 │       ┌─────────┼─────────────────┼──────────────────┤              │input has
                │                 │       │         │                 │                  │              │its own
        ┌───────┴───────┐         │       │ ┌───────┴───────┐ ┌───────┴───────┐ ┌────────┴──────┐       │cost contri
        │pair cartesians│         │ ┌─────┼─┤unitcircle ang │ │unitcircle dih │ │unitcircle sdih│       │bution
        │(batch, batch) │         │ │if no│ │(b, (N*3-2)*2) │ │(b, (N*3-3)*2) │ │(b, S*2)       │       │which
        └───────┬───────┘         │ │angles └───────┬───────┘ └───────┬───────┘ └────────┬──────┘       │compares
                │compare the pair │ │are  │         │                 │                  │              │input
                │wise distances of│ │fed  │       if│use_backbone_angles               if│use_sidechains│and
                │the input cartesi│ │through        │                 │                  │              │output
                │ans with the gene│ │the  ┼ ┌───────┴─────────────────┴──────────────────┴──────┐       │->
                │rated cartesians │ │network│concatenate the angle-inputs. Based on parameters. │       │angle_cost
                │-> cartesian loss│ │use  │ │(batch, sum(angle_shapes)                          │       │dihedral_cost
                │                 │ │mean │ └─────────────────────────┬─────────────────────────┘       │side_dihedral
                │                 │ │angles                           │                                 │_cost
                │                 │ │     │                           │                                 │
                │                 │ │     │                           │                                 │
                │                 │ │     │             ┌─────────────┴──────────────┐                  │
                │                 │ │     │             │Encoder layers              │                  │
                │                 │ │     │             │(batch, n_neurons)          │                  │
                │                 │ │     │             └─────────────┬──────────────┘                  │
                │                 │ │     │                           │                                 │
                │                 │ │     │                           │                                 │
                │                 │ │     │add a sigmoid-weighted     │            add a loss function  │
                │      compare the│ │     │loss function that┌────────┴────────┐   to center the points │
                │      ┌──────────┼─┼─────┴──────────────────┤Bottleneck,Latent├────────────────────    │
                │      │generated │ │      compares the pair-│ (batch, 2)      │   around the origin    │
                │      │cartesians│ │      wise distances of └────────┬────────┘   -> center loss       │
                │      │with the  │ │      input and latent           │                                 │
                │      │pairwise  │ │      samples                    │                                 │
                │      │distances │ │      -> distance loss           │                                 │
                │      │of the    │ │                   ┌─────────────┴──────────────┐                  │
                │      │bottleneck│ │                   │Decoder layers              │                  │
                │      │use a 2nd │ │                   │(batch, n_neurons)          │                  │
                │      │sigmoid   │ │                   └─────────────┬──────────────┘                  │
                │      │function  │ │                                 │                                 │
                │      │for this  │ │                                 │                                 │
                │      │->        │ │                                 │                                 │
                │      │cartesian │ │       ┌─────────────────────────┴─────────────────────────┐       │
                │      │distance  │ │       │split the output of the decoder to get angles back │       │
                │      │loss      │ │       │(batch, sum(angle_shapes)                          │       │
                │      │          │ │       └───────┬─────────────────┬─────────────────┬───────┘       │
                │      │          │ │               │                 │                 │               │
                │      │          │ │               │                 │                 │               │
                │      │          │ │               │                 │                 │               │
                │      │          │ │       ┌───────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐       │
                │      │          │ │       │unitcircle ang │ │unitcircle dih │ │unitcircle sdih│       │
                │      │          │ │       │(b, (N*3-2)*2) │ │(b, (N*3-3)*2) │ │(b, S*2)       │       │
                │      │          │ │       └───────┬───────┘ └───────┬───────┘ └────────┬──────┘       │
                │      │          │ │               │                 │                  │              │
                │      │          │ │             if│use_backbone_angles               if│use_sidechains│
                │      │          │ │               │                 │                  │              │
                │      │          │ │       ┌───────┴───────┐ ┌───────┴───────┐ ┌────────┴──────┐       │
                │      │          │ └───────┤(mean) angles  │ │dihedrals      │ │side dihedrals │       │
                │      │          │         │(batch,3N*3-2) │ │(batch,3N*3-3) │ │(batch, S)     ├───────┘
                │      │          │         └───────┬───────┘ └───────┬───────┘ └───────────────┘
                │      │          │                 │                 │
                │      │          │                 │                 │
                │      │          │                 │                 │
                │      │  ┌───────┴─────────────────┴─────────────────┴──────┐
                │      │  │create new cartesians with chain-in-plane and     │
                │      │  │rotation matrices (batch, 3*N, 3)                 │
                │      │  └───────┬──────────────────────────────────────────┘
                │      │          │
                │      │          │
                │      │          │
                │      │  ┌───────┴───────┐
                │      └──┤gen pair cartes│
                │         │(batch,batch)  │
                └─────────┴───────────────┘

    """
    if isinstance(input_shapes, tuple):
        assert isinstance(input_shapes[2], tuple), (
            f"Please provide a tuple for the shape of the cartesians, so a model "
            f"with the correct sparse inputs can be created."
        )
    # if parameters are None, create a new instance of `ADCParameters`
    p = ADCParameters() if parameters is None else parameters

    # inform the user about tensorflow_graphics
    if p.multimer_training is not None:
        if p.multimer_training == "homogeneous_transform":
            try:
                # Third Party Imports
                import tensorflow_graphics as tfg
            except ModuleNotFoundError as e:
                raise Exception(
                    f"To use the 'homogeneous_transform' multimer training, please "
                    f"install the 'tensorflow_graphics' package:\n"
                    f"`pip install tensorflow_graphics`"
                ) from e

    # it is important to keep track of the inputs.
    # the inputs will always be provided in the order:
    # angles, central_dihedrals, cartesians, distances, side_dihedrals
    # these values will always be provided. They might not go through the
    # network (especially the side_dihedrals), but the shape will be provided
    # nonetheless.
    if not p.reconstruct_sidechains:
        (
            angles_input_shape,
            central_dihedrals_input_shape,
            cartesians_input_shape,
            distances_input_shape,
            side_dihedrals_input_shape,
            sparse,
            sidechain_only_sparse,
        ) = _unpack_and_assert_input_shapes(
            input_shapes,
            p,
            sparse,
            sidechain_only_sparse,
        )
    else:
        (
            central_angles_input_shape,
            central_dihedrals_input_shape,
            all_cartesians_input_shape,
            central_distances_input_shape,
            side_angles_input_shape,
            side_dihedrals_input_shape,
            side_distances_input_shape,
            sparse,
            sidechain_only_sparse,
        ) = _unpack_and_assert_input_shapes_w_sidechains(
            input_shapes,
            p,
            sparse,
            sidechain_only_sparse,
        )

    # define the regularizer, that will be used from here on out
    # the L2 regularizer adds a loss with 1/2 * sum(w ** 2) to
    # each layer
    regularizer = tf.keras.regularizers.l2(p.l2_reg_constant)

    # define the inputs
    (
        input_central_dihedrals_placeholder,
        input_central_dihedrals_unit_circle,
        input_central_dihedrals_dense_model,
    ) = _create_inputs_periodic_maybe_sparse(
        central_dihedrals_input_shape,
        p,
        name="central_dihedrals",
        sparse=sparse,
    )
    assert (
        input_central_dihedrals_placeholder.shape[1] * 2
        == input_central_dihedrals_unit_circle.shape[1]
    )

    # The split left and split right are needed for the `dihedrals_to_cartesian_tf_layers`
    # function. In this function is a for loop, that needs to iterate over a
    # conditional. That conditional needs to be set here, otherwise tensorflow
    # produces a symbolic tensor with shape [None, None], which can't be used for
    # iteration
    # The chain of atoms is split so that 3 atoms remain in on the xy plain
    # the left and right tails of these three atoms rotate into the z-axis to
    # create a 3D structure. However, the number of central cartesians can
    # be even or uneven, depending on the number of amino acids.
    # Case 1: 3 Amino Acids (9 cartesians, 6 dihedrals):
    # Here, the N-CA-C atoms of the 2nd amino acid remain on the xy plane.
    # The left and right cartesians index the atoms of the first two and last
    # two residues, respectively. Thus, they assume the shape 6 and 6.
    # The 6 dihedrals are split evenly into 3 and 3.
    # Case 2: 4 Amino acids (12 cartesians, 9 dihedrals):
    # Here, the left cartesians contain 8 atoms, the right contain 7 atoms.
    # Thus, the dihedrals are split unevenly into 5 dihedrals left, 4 right
    # Case 3: M1-connected diUbi with 152 residues (456 cartesians, 453 dihedrals):
    # Here, the dihedrals are split into 227 for left and 226 for right
    # if sparse:
    #     _cartesians_input_shape = cartesians_input_shape // 3
    # else:
    #     _cartesians_input_shape = cartesians_input_shape
    # _split = int(int(_cartesians_input_shape) / 2)
    # _cartesian_right = np.arange(_cartesians_input_shape)[_split - 1:]
    # _dihedrals_right = np.arange(central_dihedrals_input_shape)[_split - 1:]
    # _cartesian_left = np.arange(_cartesians_input_shape)[_split + 1:: -1]
    # _dihedrals_left = np.arange(central_dihedrals_input_shape)[_split - 2:: -1]
    # _n_left = int(_dihedrals_left.shape[-1])
    # _n_right = int(_dihedrals_right.shape[-1])
    if not p.reconstruct_sidechains:
        if sparse:
            left_split = cartesians_input_shape // 3 // 2 - 1
        else:
            left_split = cartesians_input_shape // 2 - 1
        right_split = central_dihedrals_input_shape // 2
    # assert _n_left == left_split, f"{_n_left=} {left_split=} {cartesians_input_shape=} {sparse=}"
    # assert _n_right == right_split

    # this input list is provided as inputs to the encoder
    # that way if the user trains with backbone angles and side-didherals
    # the encoder can be provided with a list/tuple of three numpy arrays
    # with the respective values,
    # otherwise the user would need to stack these arrays along the
    # feature axis before feeding it to the encoder
    encoder_input_list = [input_central_dihedrals_placeholder]

    # backbone angles
    # For the case of sparse and not using backbone angles, the angles will be
    # treated as a non-periodic input.
    if p.use_backbone_angles and not p.reconstruct_sidechains:
        (
            input_angles_placeholder,
            input_angles_unit_circle,
            input_angles_dense_model,
        ) = _create_inputs_periodic_maybe_sparse(
            angles_input_shape,
            p,
            name="angles",
            sparse=sparse,
        )
        assert (
            input_angles_placeholder.shape[1] * 2 == input_angles_unit_circle.shape[1]
        )
        encoder_input_list = [
            input_angles_placeholder,
            input_central_dihedrals_placeholder,
        ]
    elif p.use_backbone_angles and p.reconstruct_sidechains:
        (
            input_central_angles_placeholder,
            input_central_angles_unit_circle,
            input_central_angles_dense_model,
        ) = _create_inputs_periodic_maybe_sparse(
            central_angles_input_shape,
            p,
            name="central_angles",
            sparse=sparse,
        )
        assert (
            input_central_angles_placeholder.shape[1] * 2
            == input_central_angles_unit_circle.shape[1]
        )
        encoder_input_list = [
            input_central_angles_placeholder,
            input_central_dihedrals_placeholder,
        ]
    else:
        (
            input_angles_placeholder,
            input_angles_unit_circle,
            input_angles_dense_model,
        ) = _create_inputs_non_periodic_maybe_sparse(
            shape=(angles_input_shape,),
            p=p,
            name=f"angles",
            sparse=sparse,
        )

    # sidechain dihedrals
    if p.use_sidechains:
        # define the inputs
        (
            input_side_dihedrals_placeholder,
            input_side_dihedrals_unit_circle,
            input_side_dihedrals_dense_model,
        ) = _create_inputs_periodic_maybe_sparse(
            side_dihedrals_input_shape,
            p,
            name="side_dihedrals",
            sparse=sparse or sidechain_only_sparse,
        )
        assert (
            input_side_dihedrals_placeholder.shape[1] * 2
            == input_side_dihedrals_unit_circle.shape[1]
        )
        encoder_input_list.append(input_side_dihedrals_placeholder)
    else:
        input_side_dihedrals_placeholder = None
        input_side_dihedrals_unit_circle = None
        input_side_dihedrals_dense_model = None

    # create more input placeholders for the sidechain angles
    if p.reconstruct_sidechains:
        (
            input_side_angles_placeholder,
            input_side_angles_unit_circle,
            input_side_angles_dense_model,
        ) = _create_inputs_periodic_maybe_sparse(
            side_angles_input_shape,
            p,
            name="side_angles",
            sparse=sparse or sidechain_only_sparse,
        )
        assert (
            input_side_dihedrals_placeholder.shape[1] * 2
            == input_side_dihedrals_unit_circle.shape[1]
        )
        encoder_input_list = [
            input_central_angles_placeholder,
            input_central_dihedrals_placeholder,
            input_side_angles_placeholder,
            input_side_dihedrals_placeholder,
        ]

    # create input placeholders for the cartesians
    if not p.reconstruct_sidechains:
        if not sparse:
            maybe_sparse_cartesian_input_shape = (cartesians_input_shape, 3)
        else:
            maybe_sparse_cartesian_input_shape = (cartesians_input_shape,)
    else:
        if not sparse:
            maybe_sparse_cartesian_input_shape = (all_cartesians_input_shape, 3)
        else:
            maybe_sparse_cartesian_input_shape = (all_cartesians_input_shape,)
    (
        input_cartesians_placeholder,
        input_dense_cartesians_placeholder,
        input_cartesians_dense_model,
    ) = _create_inputs_non_periodic_maybe_sparse(
        shape=maybe_sparse_cartesian_input_shape,
        p=p,
        name="cartesians",
        sparse=sparse,
        reshape=3,
    )

    if p.reconstruct_sidechains:
        (
            input_central_distances_placeholder,
            input_dense_central_distances_placeholder,
            input_central_distances_dense_model,
        ) = _create_inputs_non_periodic_maybe_sparse(
            shape=(central_distances_input_shape,),
            p=p,
            name="central_distances",
            sparse=sparse,
        )
        (
            input_side_distances_placeholder,
            input_dense_side_distances_placeholder,
            input_side_distances_dense_model,
        ) = _create_inputs_non_periodic_maybe_sparse(
            shape=(side_distances_input_shape,),
            p=p,
            name="side_distances",
            sparse=sparse,
        )
    else:
        # create input placeholders for the distances
        (
            input_distances_placeholder,
            input_dense_distances_placeholder,
            input_distances_dense_model,
        ) = _create_inputs_non_periodic_maybe_sparse(
            shape=(distances_input_shape,), p=p, name="distances", sparse=sparse
        )

    # we can now create the input pairwise distances, which can be used in
    # the case of multimer homogeneous_transformation matrices
    input_cartesians_pairwise = PairwiseDistances(p, "input")(
        input_dense_cartesians_placeholder
    )

    # flatten the input cartesians pairwise distances to pass them through the network
    # the pairwise distances are, just like angles
    # rotationally and translationally invariant and can thus be used for
    # training
    input_cartesians_pairwise_defined_shape = None
    multimer_lengths = []
    if p.multimer_training is not None:
        if p.multimer_topology_classes is not None:
            key1 = list(p.multimer_lengths.keys())[0]
            for key, val in p.multimer_lengths.items():
                for i, v in enumerate(val):
                    assert v == p.multimer_lengths[key1][i], (
                        f"The current model for using multiple topologies with "
                        f"multimers only supports multimers with the same number "
                        f"of residues per multiimer in all topology classes."
                    )
            multimer_lengths = p.multimer_lengths[key1]
        else:
            multimer_lengths = p.multimer_lengths
        flattened_shape = int(scipy.special.binom(sum(multimer_lengths), 2))
        if p.multimer_training == "homogeneous_transformation":
            input_cartesians_pairwise_defined_shape = tf.reshape(
                input_cartesians_pairwise,
                shape=(tf.shape(input_cartesians_pairwise)[0], flattened_shape),
            )
            encoder_input_list.append(input_dense_cartesians_placeholder)

    # define the splits for the decoder output
    # because the angular inputs are concatenated for the decoder, we want
    # to keep track how to split them afterward
    if not p.reconstruct_sidechains:
        splits, encoder_input_placeholder = _concatenate_inputs(
            p,
            input_angles_unit_circle,
            input_central_dihedrals_unit_circle,
            input_side_dihedrals_unit_circle,
            input_cartesians_pairwise_defined_shape,
        )
    else:
        input_angles_unit_circle = input_central_angles_unit_circle
        splits, encoder_input_placeholder = _concatenate_inputs_reconstruct_sidechains(
            p,
            input_central_angles_unit_circle,
            input_central_dihedrals_unit_circle,
            input_side_angles_unit_circle,
            input_side_dihedrals_unit_circle,
        )

    assert encoder_input_placeholder is not None
    assert encoder_input_list is not None
    assert all([i is not None for i in encoder_input_list])

    # build the encoder provide it with the encoder_input_placeholder
    encoder_model, encoder_output_placeholder = _get_encoder_model(
        encoder_input_placeholder,
        p,
        input_list=encoder_input_list,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizer,
        bias_initializer=bias_initializer,
        write_summary=write_summary,
    )

    # build the decoder to the required shape
    (
        decoder_model,
        output_angles_placeholder,
        output_central_dihedrals_placeholder,
        output_side_dihedrals_placeholder,
        extra_output_placeholder,
    ) = _get_adc_decoder(
        p,
        splits,
        input_angles_placeholder=input_angles_unit_circle,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizer,
        bias_initializer=bias_initializer,
        write_summary=write_summary,
        input_placeholder=encoder_output_placeholder,
        n_proteins=len(multimer_lengths),
    )

    # Provide the backmap layer with all it needs
    if p.multimer_training is None and not p.reconstruct_sidechains:
        back_cartesians = BackMapLayer(
            left_split=left_split,
            right_split=right_split,
        )(
            (
                input_dense_distances_placeholder,
                output_angles_placeholder,
                output_central_dihedrals_placeholder,
            )
        )
    elif p.multimer_training is None and p.reconstruct_sidechains:
        _inputs = (
            input_dense_central_distances_placeholder,
            output_angles_placeholder,
            output_central_dihedrals_placeholder,
            input_dense_side_distances_placeholder,
            extra_output_placeholder,
            output_side_dihedrals_placeholder,
        )
        back_cartesians = BackMapLayerWithSidechains(p.sidechain_info)(_inputs)
    else:
        back_cartesians = BackMapLayerTransformations(multimer_lengths)(
            (
                input_dense_distances_placeholder,
                output_angles_placeholder,
                output_central_dihedrals_placeholder,
                extra_output_placeholder,
            )
        )

    # create the pairwise distances of the input cartesians and the output
    # back-mapped cartesians
    output_cartesians_pairwise = PairwiseDistances(p, "output")(back_cartesians)

    # create a functional model from the inputs and outputs
    # define the inputs
    if not p.reconstruct_sidechains:
        inputs = [
            input_angles_placeholder,
            input_central_dihedrals_placeholder,
            input_cartesians_placeholder,
            input_distances_placeholder,
        ]

        # the outputs depend on the parameters used
        # for use_backbone_angles, the decoder output will be a list of
        # tensors and thus needs to be unpacked
        # else, the output is a single tensor and can't be unpacked
        if p.use_backbone_angles:
            outputs = [
                *decoder_model(encoder_model(encoder_input_list)),
                back_cartesians,
                input_cartesians_pairwise,
                output_cartesians_pairwise,
            ]
        else:
            outputs = [
                output_angles_placeholder,
                decoder_model(encoder_model(encoder_input_list)),
                back_cartesians,
                input_cartesians_pairwise,
                output_cartesians_pairwise,
            ]
        if len(splits) >= 3:
            inputs.append(input_side_dihedrals_placeholder)

    else:
        inputs = [
            input_central_angles_placeholder,
            input_central_dihedrals_placeholder,
            input_cartesians_placeholder,
            input_central_distances_placeholder,
            input_side_angles_placeholder,
            input_side_dihedrals_placeholder,
            input_side_distances_placeholder,
        ]
        outputs = [
            *decoder_model(encoder_model(encoder_input_list)),
            back_cartesians,
            input_cartesians_pairwise,
            output_cartesians_pairwise,
        ]

        model = ADCFunctionalModelSidechainReconstruction(
            parameters=p,
            inputs=inputs,
            outputs=outputs,
            encoder=encoder_model,
            decoder=decoder_model,
            get_dense_model_central_dihedrals=input_central_dihedrals_dense_model,
            get_dense_model_central_angles=input_central_angles_dense_model,
            get_dense_model_side_dihedrals=input_side_dihedrals_dense_model,
            get_dense_model_cartesians=input_cartesians_dense_model,
            get_dense_model_central_distances=input_central_distances_dense_model,
            get_dense_model_side_distances=input_side_distances_dense_model,
            get_dense_model_side_angles=input_side_angles_dense_model,
        )
        return model

    # create the final model
    if not sparse and not sidechain_only_sparse:
        if use_experimental_model:
            ModelClass = ADCFunctionalModelTesting
        else:
            ModelClass = ADCFunctionalModel
        model = ModelClass(
            parameters=p,
            inputs=inputs,
            outputs=outputs,
            encoder=encoder_model,
            decoder=decoder_model,
        )
    else:
        model = ADCSparseFunctionalModel(
            parameters=p,
            inputs=inputs,
            outputs=outputs,
            encoder=encoder_model,
            decoder=decoder_model,
            get_dense_model_central_dihedrals=input_central_dihedrals_dense_model,
            get_dense_model_central_angles=input_angles_dense_model,
            get_dense_model_side_dihedrals=input_side_dihedrals_dense_model,
            get_dense_model_cartesians=input_cartesians_dense_model,
            get_dense_model_distances=input_distances_dense_model,
        )

    # write a summary
    if write_summary:
        if p.tensorboard or p.write_summary:
            with Capturing() as output:
                model.summary()
            with open(p.main_path + "/complete_model_summary.txt", "w") as f:
                f.write("\n".join(output))
        else:
            model.summary()
    return model


def _unpack_and_assert_input_shapes_w_sidechains(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[
            tuple[int],
            tuple[int],
            Union[tuple[int, int], tuple[int]],
            tuple[int],
            tuple[int],
        ],
    ],
    p: ADCParameters,
    input_sparse: bool = False,
    input_sidechain_only_sparse: bool = False,
) -> tuple[int, int, int, int, int, bool, bool]:
    """This function unpacks and asserts the input_shapes for the regular protein case.

    In contrast to `_unpack_data_and_assert_input_shapes`, a full sidechain
    reconstruction will be executed.

    Args:
        input_shapes(Union[tf.data.Dataset, tuple[int, int, int, int, int]]):
            The input shapes, that will be used in the construction of the model.
        parameters (Optional[encodermap.parametersADCParameters]): An instance
            of `encodermap.parameters.ADCParameters`,
            which holds further parameters in network construction. If None
            is provided, a new instance with default parameters will be
            created. Defaults to None.
        sparse (bool): Whether sparse inputs are expected. Defaults to False.
        input_sidechain_only_sparse (bool): Whether only the sidechain dihedrals
            are sparse. In that case, the input shape of the cartesians is
            different, because the cartesians are flattened to a rank 2 tensor
            before running them through a dense layer and then stacking them again
            to shape (n_frames, n_atoms, 3).

    Returns:
        tuple: A tuple containing the following:
            - int: The input shape for the training angles.
            - int: The input shape for the training dihedrals.
            - int: The input shape for the cartesians.
            - int: The input shape for the distances.
            - Union[int, None]: The input shape for the training sidechain dihedrals.
                Can be None, if they are not used for training.

    """
    if p.multimer_training is not None:
        assert (
            not input_sparse
        ), f"Using multimers currently not possible with sparse and/or full sidechain reconstruction."
        return _unpack_and_assert_input_shapes_multimers(input_shapes, p)
    if isinstance(input_shapes, (tuple, list)):
        assert len(input_shapes) == 7
        (
            central_angles_input_shape,
            central_dihedrals_input_shape,
            all_cartesians_input_shape,
            central_distances_input_shape,
            side_angles_input_shape,
            side_dihedrals_input_shape,
            side_distances_input_shape,
        ) = [i[0] for i in input_shapes]
        if input_sparse and len(input_shapes[2]) == 2:
            sidechain_only_sparse = True
            sparse = input_sparse
        else:
            sidechain_only_sparse = input_sidechain_only_sparse
            sparse = input_sparse
    else:
        d = input_shapes.element_spec

        # all dense
        if not any([isinstance(i, tf.SparseTensorSpec) for i in d]):
            sparse = False
            sidechain_only_sparse = False
        # only sparse sidechains
        elif all(
            [not isinstance(i, tf.SparseTensorSpec) for i in d[:-1]]
        ) and isinstance(d[-1], tf.SparseTensorSpec):
            sparse = False
            sidechain_only_sparse = True
        # other stuff sparse
        else:
            sparse = True
            sidechain_only_sparse = False

        # check if dataset is batches
        try:
            central_angles_input_shape = d[0].shape[1]
        except IndexError as e:
            raise Exception(
                f"You probably provided a tf.data.Dataset, that is not batched "
                f"and thus an index error was raised."
            ) from e

        # define shapes
        central_dihedrals_input_shape = d[1].shape[1]
        all_cartesians_input_shape = d[2].shape[1]
        central_distances_input_shape = d[3].shape[1]
        side_angles_input_shape = d[4].shape[1]
        try:
            side_dihedrals_input_shape = d[5].shape[1]
            side_distances_input_shape = d[6].shape[1]
        except IndexError:
            raise Exception(f"Not enough items in tuple for sidechain reconstruction.")

    # make sure that the inputs have had the correct order
    # because a protein with N residues has N*3 cartesians, N*3 - 1 distances
    # N*3 - 2 angles, and N*3 - 3 dihedrals
    N = (central_distances_input_shape + 1) / 3
    if not sparse or sidechain_only_sparse:
        assert all_cartesians_input_shape == N * 3 + side_distances_input_shape
    # sparse tensors have to be rank 2, so the sparse cartesians need to be
    # flattened, and the stacked back, once they are dense again
    # as of tf >= 2.16 sparse tensors can have a higher rank
    # maybe this is worth updating
    else:
        assert all_cartesians_input_shape // 3 == N * 3 + side_distances_input_shape
    assert central_angles_input_shape == N * 3 - 2
    assert central_dihedrals_input_shape == central_angles_input_shape - 1
    assert side_dihedrals_input_shape < side_angles_input_shape

    return (
        central_angles_input_shape,
        central_dihedrals_input_shape,
        all_cartesians_input_shape,
        central_distances_input_shape,
        side_angles_input_shape,
        side_dihedrals_input_shape,
        side_distances_input_shape,
        sparse,
        sidechain_only_sparse,
    )


@testing
def _unpack_and_assert_input_shapes_multimers(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[
            tuple[int],
            tuple[int],
            tuple[int, int],
            tuple[int],
            tuple[int],
        ],
    ],
    p: ADCParameters,
) -> tuple[int, int, int, int, int, bool, bool]:
    if not p.use_backbone_angles:
        raise Exception(
            f"Training with multimers currently only possible with backbone_angles"
        )
    if not p.use_sidechains:
        raise Exception(
            f"Training with multimers currently only possible with use_sidechains"
        )

    if p.multimer_topology_classes is not None:
        lengths = []
        for top in p.multimer_topology_classes:
            n_proteins = len(p.multimer_lengths[top])
            lengths.append(n_proteins)
        assert len(set(lengths)) == 1, (
            f"Can only use topology-class multimer training with a consistent "
            f"number of proteins per multimer. Got "
            f"{[len(p.multimer_lengths[top]) for top in p.multimer_topology_classes]=}"
        )
        n_proteins = lengths[0]
    else:
        n_proteins = len(p.multimer_lengths)

    if isinstance(input_shapes, tf.data.Dataset):
        for d in input_shapes:
            break
        input_shapes = tuple([i.shape[1:] for i in d])

    (
        angles_input_shape,
        central_dihedrals_input_shape,
        cartesians_input_shape,
        distances_input_shape,
        side_dihedrals_input_shape,
    ) = [i[0] for i in input_shapes]

    N = cartesians_input_shape // 3 // n_proteins
    assert distances_input_shape == n_proteins * (N * 3 - 1)
    assert angles_input_shape == n_proteins * (N * 3 - 2)
    assert central_dihedrals_input_shape == n_proteins * (N * 3 - 3)

    return (
        angles_input_shape,
        central_dihedrals_input_shape,
        cartesians_input_shape,
        distances_input_shape,
        side_dihedrals_input_shape,
        False,
        False,
    )


def _unpack_and_assert_input_shapes(
    input_shapes: Union[
        tf.data.Dataset,
        tuple[
            tuple[int],
            tuple[int],
            Union[tuple[int, int], tuple[int]],
            tuple[int],
            tuple[int],
        ],
    ],
    p: ADCParameters,
    input_sparse: bool = False,
    input_sidechain_only_sparse: bool = False,
) -> tuple[int, int, int, int, Union[int, None], bool, bool]:
    """This function unpacks and asserts the input_shapes for the regular protein case.

    Args:
        input_shapes(Union[tf.data.Dataset, tuple[int, int, int, int, int]]):
            The input shapes, that will be used in the construction of the model.
        parameters (Optional[encodermap.parameters.ADCParameters]): An instance of
            `encodermap.parameters.ADCParameters`,
            which holds further parameters in network construction. If None
            is provided, a new instance with default parameters will be
            created. Defaults to None.
        sparse (bool): Whether sparse inputs are expected. Defaults to False.
        input_sidechain_only_sparse (bool): Whether only the sidechain dihedrals
            are sparse. In that case, the input shape of the cartesians is
            different, because the cartesians are flattened to a rank 2 tensor
            before running them through a dense layer and then stacking them again
            to shape (n_frames, n_atoms, 3).

    Returns:
        tuple: A tuple containing the following:
            - int: The input shape for the training angles.
            - int: The input shape for the training dihedrals.
            - int: The input shape for the cartesians.
            - int: The input shape for the distances.
            - Union[int, None]: The input shape for the training sidechain dihedrals.
                Can be None, if they are not used for training.

    """
    if p.multimer_training is not None:
        assert not input_sparse, f"Using multimers currently not possible with sparse."
        return _unpack_and_assert_input_shapes_multimers(input_shapes, p)
    if isinstance(input_shapes, (tuple, list)):
        if len(input_shapes) == 5:
            (
                angles_input_shape,
                central_dihedrals_input_shape,
                cartesians_input_shape,
                distances_input_shape,
                side_dihedrals_input_shape,
            ) = [i[0] for i in input_shapes]
        else:
            (
                angles_input_shape,
                central_dihedrals_input_shape,
                cartesians_input_shape,
                distances_input_shape,
            ) = [i[0] for i in input_shapes]
            side_dihedrals_input_shape = None
        if input_sparse and len(input_shapes[2]) == 2:
            sidechain_only_sparse = True
            sparse = input_sparse
        else:
            sidechain_only_sparse = input_sidechain_only_sparse
            sparse = input_sparse
    else:
        d = input_shapes.element_spec

        # all dense
        if not any([isinstance(i, tf.SparseTensorSpec) for i in d]):
            sparse = False
            sidechain_only_sparse = False
        # only sparse sidechains
        elif all(
            [not isinstance(i, tf.SparseTensorSpec) for i in d[:-1]]
        ) and isinstance(d[-1], tf.SparseTensorSpec):
            sparse = False
            sidechain_only_sparse = True
        # other stuff sparse
        else:
            sparse = True
            sidechain_only_sparse = False

        # check if dataset is batches
        try:
            angles_input_shape = d[0].shape[1]
        except IndexError as e:
            raise Exception(
                f"You probably provided a tf.data.Dataset, that is not batched "
                f"and thus an index error was raised."
            ) from e

        # define shapes
        central_dihedrals_input_shape = d[1].shape[1]
        try:
            cartesians_input_shape = d[2].shape[1]
        except IndexError as e:
            raise Exception(
                f"Could not decide on a cartesian input shape for the requested "
                f"model using the provided dataset with {d=}. Normally, "
                f"it is expected for index 2 of this dataset to provide the "
                f"input shape of the cartesian coordinates. However, an "
                f"IndexError was raised, trying to access this index. "
            ) from e
        distances_input_shape = d[3].shape[1]
        if len(d) > 4:
            side_dihedrals_input_shape = d[4].shape[1]
        else:
            side_dihedrals_input_shape = None

    # make sure that the inputs have had the correct order
    # because a protein with N residues has N*3 cartesians, N*3 - 1 distances
    # N*3 - 2 angles, and N*3 - 3 dihedrals
    if not sparse or sidechain_only_sparse:
        N = cartesians_input_shape // 3
    # sparse tensors have to be rank 2, so the sparse cartesians need to be
    # flattened, and the stacked back, once they are dense again
    else:
        N = cartesians_input_shape // 3 // 3
    assert (
        distances_input_shape == N * 3 - 1
    ), f"{N=} {sparse=} {sidechain_only_sparse=}"
    assert angles_input_shape == N * 3 - 2
    assert central_dihedrals_input_shape == N * 3 - 3

    return (
        angles_input_shape,
        central_dihedrals_input_shape,
        cartesians_input_shape,
        distances_input_shape,
        side_dihedrals_input_shape,
        sparse,
        sidechain_only_sparse,
    )


def _get_adc_decoder(
    p: ADCParameters,
    splits: list[int],
    input_angles_placeholder: Optional[tf.Tensor] = None,
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    kernel_regularizer: tf.keras.regularizers.Regularizer = tf.keras.regularizers.l2(
        0.001
    ),
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = False,
    input_placeholder: Optional[tf.Tensor] = None,
    n_proteins: Optional[int] = None,
) -> tuple[
    tf.keras.models.Model,
    tf.Tensor,
    tf.Tensor,
    Union[None, tf.Tensor],
    Union[None, tf.Tensor],
]:
    """Special function to run a decoder and unpack the outputs.

    This function calls `_get_decoder_model` to get a standard decoder and then
    splits the output according to the provided `splits` and the `p`.

    Args:
        p (encodermap.parameters.ADCParameters): The parameters.
        splits (list[int]): A list of ints giving the splits of the decoder
            outputs. It is expected that the splits follow the logic of
            angles-dihedrals-sidedihedrals. If only dihedrals are used for
            training, `splits` is expected to be a list of len 1.
        input_angles_placeholder (Optional[tf.Tensor]): When only using dihedrals
            for training, this placeholder should be provided to create a
            set of mean angles. Can also be None, in case len(splits) >= 2.
        kernel_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "VarianceScaling", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "VarianceScaling" is
            provided, the weights will be initialized with `tf.keras.initializers.
            VarianceScaling()`. Defaults to "VarianceScaling". If "deterministic"
            is provided, a seed will be used with VarianceScaling. If a dict with
            weight matrices is supplied, the keys should follow this naming con-
            vention: ["dense/kernel", "dense_1/kernel", "dense_2/kernel", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        kernel_regularizer (tf.keras.regularizers.Regularizer): The regularizer
            for the kernel (i.e. the layer weights). Standard in EncoderMap is
            to use the l2 regularizer with a regularization constant of 0.001.
        bias_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "RandomNormal", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "RandomNormal" is
            provided, the weights will be initialized with `tf.keras.initializers.
            RandomNormal(0.1, 0.05)`. Defaults to "RandomNormal". If "deterministic"
            is provided, a seed will be used with RandomNormal. If a dict with
            bias matrices is supplied, the keys should follow this naming con-
            vention: ["dense/bias", "dense_1/bias", "dense_2/bias", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        write_summary (bool): Whether to print a summary. If p.tensorboard is True
            a file will be generated. at the main_path.
        n_proteins (Optional[int]): If not None, number of proteins that
            constitute the multimer group that is trained.

    Returns:
        tuple: A tuple containing the following:
            - tf.keras.models.Model: The decoder model.
            - tf.Tensor: The angles (either mean, or learned angles).
            - tf.Tensor: The dihedrals.
            - Union[None, tf.Tensor]: The sidechain dihedrals. If p.use_sidechains
                is false, None will be returned.
            - Union[None, tf.Tensor]: The homogeneous transformation matrices
                for multimer training. If p.multimer_training is None, None
                will be returned.


    """
    if len(splits) == 2:
        assert splits[0] - 2 == splits[1], (
            f"Order of splits is wrong. It is expected, that the splits for "
            f"angles at splits[0], is splits two larger than the central dihedrals "
            f"at splits[1]. However, {splits=}"
        )
    # get the standard decoder and its inputs and outputs.
    if p.multimer_training is not None:
        splits[-1] = (n_proteins - 1) * 4 * 4
    decoder, output_placeholder, input_placeholder = _get_decoder_model(
        p=p,
        out_shape=sum(splits),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        write_summary=False,
        input_placeholder=input_placeholder,
    )

    # fmt: off
    splits_side_dihedrals = None
    extra_tensor = None
    if len(splits) == 1:
        assert not p.use_backbone_angles and not p.use_sidechains, f"Parameters and splits do not coincide: {p=}, {splits=}"
        splits_central_dihedrals = PeriodicOutput(p, "dihedrals_from_unit_circle")(output_placeholder)
        splits_angles = MeanAngles(p, "mean_angles")(input_angles_placeholder)
        decoder_output = splits_central_dihedrals
    elif len(splits) == 2:
        assert p.use_backbone_angles and not p.use_sidechains, f"Parameters and splits do not coincide: {p=}, {splits=}"
        out_angles, out_dihedrals = tf.split(output_placeholder, splits, 1)
        splits_angles = PeriodicOutput(p, "angles_from_unit_circle")(out_angles)
        splits_central_dihedrals = PeriodicOutput(p, "dihedrals_from_unit_circle")(out_dihedrals)
        decoder_output = (splits_angles, splits_central_dihedrals)
    elif len(splits) == 3:
        assert p.use_backbone_angles and p.use_sidechains, f"Parameters and splits do not coincide: {p=}, {splits=}"
        out_angles, out_dihedrals, out_side_dihedrals = tf.split(output_placeholder, splits, 1)
        splits_angles = PeriodicOutput(p, "angles_from_unit_circle")(out_angles)
        splits_central_dihedrals = PeriodicOutput(p, "dihedrals_from_unit_circle")(out_dihedrals)
        splits_side_dihedrals = PeriodicOutput(p, "side_dihedrals_from_unit_circle")(out_side_dihedrals)
        decoder_output = (splits_angles, splits_central_dihedrals, splits_side_dihedrals)
    elif len(splits) == 4:
        if p.multimer_training is None and not p.reconstruct_sidechains:
            raise Exception(f"Got wrong splits: {splits=}")
        if p.multimer_training is not None and not p.reconstruct_sidechains:
            out_angles, out_dihedrals, out_side_dihedrals, out_transformation_matrices = tf.split(output_placeholder, splits, 1)
            splits_angles = PeriodicOutput(p, "angles_from_unit_circle")(out_angles)
            splits_central_dihedrals = PeriodicOutput(p, "dihedrals_from_unit_circle")(out_dihedrals)
            splits_side_dihedrals = PeriodicOutput(p, "side_dihedrals_from_unit_circle")(out_side_dihedrals)
            extra_tensor = tf.reshape(
                out_transformation_matrices,
                shape=(tf.shape(splits_angles)[0], n_proteins - 1, 4, 4)
            )
        if p.multimer_training is None and p.reconstruct_sidechains:
            out_angles, out_dihedrals, out_side_angles, out_side_dihedrals = tf.split(output_placeholder, splits, 1)
            splits_angles = PeriodicOutput(p, "central_angles_from_unit_circle")(out_angles)
            splits_central_dihedrals = PeriodicOutput(p, "central_dihedrals_from_unit_circle")(out_dihedrals)
            extra_tensor = PeriodicOutput(p, "side_angles_from_unit_circle")(out_side_angles)
            splits_side_dihedrals = PeriodicOutput(p, "side_dihedrals_from_unit_circle")(out_side_dihedrals)
        decoder_output = (splits_angles, splits_central_dihedrals, extra_tensor, splits_side_dihedrals)
    else:
        raise Exception(f"Got wrong splits: {splits=}")
    # fmt: on

    # create the model
    try:
        model = tf.keras.models.Model(
            inputs=input_placeholder,
            outputs=decoder_output,
            name="Decoder",
        )
    except ValueError as e:
        raise Exception(f"{splits=}, {decoder_output=}") from e

    # assert that the sequence of outputs is correct
    assert model.input_shape[1] == p.n_neurons[-1]
    # for use_backbone angles we can assert the shape
    if p.use_backbone_angles:
        assert model.output_shape[0][1] * 2 == splits[0]
    # if only using dihedrals, we can't assume the shape, as it is defined
    # during runtime by the batch size.
    else:
        assert model.output_shape[0] is None
    # if using backbone_angles, the output shape is a list of tuple of ints
    if p.use_backbone_angles:
        assert model.output_shape[1][1] * 2 == splits[1]
    else:
        model.output_shape[-1] == splits[0]
    if p.use_sidechains:
        assert model.output_shape[2][1] * 2 == splits[2]
    else:
        assert len(model.output_shape) == 2

    # write a summary
    if write_summary:
        if p.tensorboard or p.write_summary:
            with Capturing() as output:
                model.summary()
            with open(p.main_path + "/decoder_summary.txt", "w") as f:
                f.write("\n".join(output))
        else:
            model.summary()

    return (
        model,
        splits_angles,
        splits_central_dihedrals,
        splits_side_dihedrals,
        extra_tensor,
    )


def _get_decoder_model(
    p: ADCParameters,
    out_shape: int,
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    kernel_regularizer: tf.keras.regularizers.Regularizer = tf.keras.regularizers.l2(
        0.001
    ),
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = False,
    input_placeholder: Optional[tf.Tensor] = None,
) -> tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]:
    """Create a decoder to the requested specs.

    Contrary to the `_get_encoder_model` function, this function doesn't require
    an input placeholder. The input placeholder is created in the function body.
    Thus, a combined autoencoder model can be built by stacking the encoder and
    decoder like so: `output = decoder(encoder(input))`.

    Args:
        p (encodermap.parameters.ADCParameters): The parameters.
        out_shape (int): The output shape of the decoder. Make sure to match it
            with the input shape of the encoder.
        kernel_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "VarianceScaling", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "VarianceScaling" is
            provided, the weights will be initialized with `tf.keras.initializers.
            VarianceScaling()`. Defaults to "VarianceScaling". If "deterministic"
            is provided, a seed will be used with VarianceScaling. If a dict with
            weight matrices is supplied, the keys should follow this naming con-
            vention: ["dense/kernel", "dense_1/kernel", "dense_2/kernel", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        kernel_regularizer (tf.keras.regularizers.Regularizer): The regularizer
            for the kernel (i.e. the layer weights). Standard in EncoderMap is
            to use the l2 regularizer with a regularization constant of 0.001.
        bias_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "RandomNormal", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "RandomNormal" is
            provided, the weights will be initialized with `tf.keras.initializers.
            RandomNormal(0.1, 0.05)`. Defaults to "RandomNormal". If "deterministic"
            is provided, a seed will be used with RandomNormal. If a dict with
            bias matrices is supplied, the keys should follow this naming con-
            vention: ["dense/bias", "dense_1/bias", "dense_2/bias", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        write_summary (bool): Whether to print a summary. If p.tensorboard is True
            a file will be generated. at the main_path.

    Returns:
        tuple: A tuple containing the following:
            - tf.keras.models.Model: The decoder model.
            - tf.Tensor: The output tensor with shape `out_shape`.
            - tf.Tensor: The input placeholder tensor with shape `p.n_neurons`.

    """
    n_neurons_with_inputs = [out_shape] + p.n_neurons

    # generate a new placeholder
    # this way, the decoder is can be created as a detached model, if no
    # input placeholder is provided
    if input_placeholder is None:
        inp = Input(shape=(p.n_neurons[-1],), name="decoder_input")
    else:
        inp = input_placeholder
        assert inp.shape[1] == p.n_neurons[-1], (
            f"The input shape of the decoder does not match the requested input "
            f"shape. I got an input shape of {inp.shape[1]=}, while parameters "
            f"requested {p.n_neurons[-1]=}."
        )

    out = inp
    for i, (n_neurons, act_fun) in enumerate(
        zip(n_neurons_with_inputs[-2::-1], p.activation_functions[-2::-1])
    ):
        if act_fun:
            act_fun = getattr(tf.nn, act_fun)
        else:
            act_fun = None
        if isinstance(kernel_initializer, str):
            if kernel_initializer == "VarianceScaling":
                _kernel_initializer = tf.keras.initializers.VarianceScaling()
            elif kernel_initializer == "deterministic":
                seed = 121110987654321 + i
                _kernel_initializer = _get_deterministic_variance_scaling(seed=seed)
            elif kernel_initializer == "ones":
                _kernel_initializer = tf.keras.initializers.Constant(1)
            else:
                raise Exception(
                    f"Keyword `kernel_initializer` only supports 'VarianceScaling' "
                    f", 'ones', or 'deterministic'. Got {kernel_initializer=}"
                )
        elif isinstance(kernel_initializer, dict):
            kernel_name = f"dense_{i + len(p.n_neurons)}/kernel"
            _kernel_initializer = MyKernelInitializer(kernel_initializer[kernel_name])
        else:
            raise TypeError(
                f"Arg `kernel_initializer` must be of type str or dict, "
                f"you supplied {type(kernel_initializer)=}."
            )
        if isinstance(bias_initializer, str):
            if bias_initializer == "RandomNormal":
                _bias_initializer = tf.keras.initializers.RandomNormal(0.1, 0.05)
            elif bias_initializer == "deterministic":
                seed = 121110987654321 + i
                _bias_initializer = _get_deterministic_random_normal(
                    0.1, 0.05, seed=seed
                )
            elif bias_initializer == "ones":
                _bias_initializer = tf.keras.initializers.Constant(1)
            else:
                raise Exception(
                    f"Keyword `bias_initializer` only supports 'RandomNormal' "
                    f", 'ones' or 'deterministic'. Got {bias_initializer=}"
                )
        elif isinstance(bias_initializer, dict):
            bias_name = f"dense_{i + len(p.n_neurons)}/bias"
            _bias_initializer = MyBiasInitializer(bias_initializer[bias_name])
        else:
            raise TypeError(
                f"Arg `bias_initializer` must be of type str or dict, "
                f"you supplied {type(bias_initializer)=}."
            )
        out = tf.keras.layers.Dense(
            units=n_neurons,
            activation=act_fun,
            name=f"Decoder_{i}",
            kernel_initializer=_kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=_bias_initializer,
        )(out)

    # create the model
    model = tf.keras.models.Model(
        inputs=inp,
        outputs=out,
        name="Decoder",
    )

    # check the model input and output
    model.input_shape[1] == p.n_neurons[-1]
    model.output_shape[1] == out_shape

    if write_summary:
        if p.tensorboard or p.write_summary:
            with Capturing() as output:
                model.summary()
            with open(p.main_path + "/decoder_summary.txt", "w") as f:
                f.write("\n".join(output))
        else:
            model.summary()

    return model, out, inp


def _get_encoder_model(
    inp: tf.Tensor,
    p: "AnyParameters",
    input_list: list[tf.Tensor],
    kernel_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "VarianceScaling", "deterministic"]
    ] = "VarianceScaling",
    kernel_regularizer: tf.keras.regularizers.Regularizer = tf.keras.regularizers.l2(
        0.001
    ),
    bias_initializer: Union[
        dict[str, np.ndarray], Literal["ones", "RandomNormal", "deterministic"]
    ] = "RandomNormal",
    write_summary: bool = False,
) -> tuple[tf.keras.models.Model, tf.Tensor]:
    """Create an encoder model and feed the inp through it.

    Args:
        inp (tf.Tensor): The input tensor of the encoder.
        p (encodermap.parameters.ADCParameters): The parameters.
        input_list (list[tf.Tensor]): This list contains the input placeholders
            for the encoder. Make sure that these input tensors point to the
            `inp` tensor in some way.
        kernel_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "VarianceScaling", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "VarianceScaling" is
            provided, the weights will be initialized with `tf.keras.initializers.
            VarianceScaling()`. Defaults to "VarianceScaling". If "deterministic"
            is provided, a seed will be used with VarianceScaling. If a dict with
            weight matrices is supplied, the keys should follow this naming con-
            vention: ["dense/kernel", "dense_1/kernel", "dense_2/kernel", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        kernel_regularizer (tf.keras.regularizers.Regularizer): The regularizer
            for the kernel (i.e. the layer weights). Standard in EncoderMap is
            to use the l2 regularizer with a regularization constant of 0.001.
        bias_initializer (Union[dict[str, np.ndarray],
            Literal["ones", "RandomNormal", "deterministic"]]): How to initialize
            the weights. If "ones" is provided, the weights will be initialized
            with `tf.keras.initializers.Constant(1)`. If "RandomNormal" is
            provided, the weights will be initialized with `tf.keras.initializers.
            RandomNormal(0.1, 0.05)`. Defaults to "RandomNormal". If "deterministic"
            is provided, a seed will be used with RandomNormal. If a dict with
            bias matrices is supplied, the keys should follow this naming con-
            vention: ["dense/bias", "dense_1/bias", "dense_2/bias", etc.]
            This is tensorflow's naming convention for unnamed dense layers.
        write_summary (bool): Whether to print a summary. If p.tensorboard is True
            a file will be generated. at the main_path.

    Returns:
        tuple: A tuple containing:
            - tf.keras.models.Model: The encoder model.
            - tf.Tensor: The output of the model.

    """
    out = inp
    for i, (n_neurons, act_fun) in enumerate(
        zip(p.n_neurons, p.activation_functions[1:])
    ):
        # define the activation function for this dense layer
        if act_fun:
            act_fun = getattr(tf.nn, act_fun)
        else:
            act_fun = None

        # get the kernel initializer for that layer
        if isinstance(kernel_initializer, str):
            if kernel_initializer == "VarianceScaling":
                _kernel_initializer = tf.keras.initializers.VarianceScaling()
            elif kernel_initializer == "deterministic":
                seed = 123456789101112 + i
                _kernel_initializer = _get_deterministic_variance_scaling(seed=seed)
            elif kernel_initializer == "ones":
                _kernel_initializer = tf.keras.initializers.Constant(1)
            else:
                raise Exception(
                    f"Keyword `kernel_initializer` only supports 'VarianceScaling' "
                    f", 'ones', or 'deterministic'. Got {kernel_initializer=}"
                )
        elif isinstance(kernel_initializer, dict):
            if i == 0:
                kernel_name = "dense/kernel"
            else:
                kernel_name = f"dense_{i}/kernel"
            _kernel_initializer = MyKernelInitializer(kernel_initializer[kernel_name])
        else:
            raise TypeError(
                f"Arg `kernel_initializer` must be of type str or dict, "
                f"you supplied {type(kernel_initializer)=}."
            )
        if isinstance(bias_initializer, str):
            if bias_initializer == "RandomNormal":
                _bias_initializer = tf.keras.initializers.RandomNormal(0.1, 0.05)
            elif bias_initializer == "deterministic":
                seed = 123456789101112 + i
                _bias_initializer = _get_deterministic_random_normal(
                    mean=0.1, stddev=0.05, seed=seed
                )
            elif bias_initializer == "ones":
                _bias_initializer = tf.keras.initializers.Constant(1)
            else:
                raise Exception(
                    f"Keyword `bias_initializer` only supports 'RandomNormal' "
                    f", 'ones' or 'deterministic'. Got {bias_initializer=}"
                )
        elif isinstance(bias_initializer, dict):
            if i == 0:
                bias_name = "dense/bias"
            else:
                bias_name = f"dense_{i}/bias"
            _bias_initializer = MyBiasInitializer(bias_initializer[bias_name])
        else:
            raise TypeError(
                f"Arg `bias_initializer` must be of type str or dict, "
                f"you supplied {type(bias_initializer)=}."
            )

        # define the layer and directly call it
        layer = tf.keras.layers.Dense(
            units=n_neurons,
            activation=act_fun,
            name=f"Encoder_{i}",
            kernel_initializer=_kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=_bias_initializer,
        )
        out = layer(out)
        # if kernel_initializer == "deterministic" and i == 0:
        #     raise Exception(f"{_kernel_initializer((906, 128))[0, 0]=} {_kernel_initializer.seed=} {_bias_initializer((128, ))[0]=} {_bias_initializer.seed=}")
        #     # assert layer.weights[0].numpy()[0, 0] == 0.004596
        #     raise Exception

    # check the shape of the output
    out.shape[1] == p.n_neurons[-1]

    # create the model here
    model = tf.keras.models.Model(
        inputs=input_list,
        outputs=out,
        name="Encoder",
    )

    # assert the input of the model
    # for use_backbone_angles
    if isinstance(model.input_shape, list):
        assert len(model.input_shape) == len(input_list)
    # for only central dihedrals
    else:
        model.input_shape[1] == input_list[0].shape[1]
    if p.use_backbone_angles and p.multimer_training is None:
        assert model.input_shape[0][1] - 1 == model.input_shape[1][1]

    # print/write info
    if write_summary:
        if p.tensorboard or p.write_summary:
            with Capturing() as output:
                model.summary()
            with open(p.main_path + "/encoder_summary.txt", "w") as f:
                f.write("\n".join(output))
        else:
            model.summary()

    return model, out


def _concatenate_inputs_reconstruct_sidechains(
    p: ADCParameters,
    central_angles_unit_circle: tf.Tensor,
    central_dihedrals_unit_circle: tf.Tensor,
    side_angles_unit_circle: tf.Tensor,
    side_dihedrals_unit_circle: tf.Tensor,
) -> tuple[list[int], tf.Tensor]:  # pragma: no doccheck
    """Concatenates input Tensors for the AngleDihedralCartesianEncoderMap with
    sidechain reconstruction.

    """
    splits = [
        central_angles_unit_circle.shape[1],
        central_dihedrals_unit_circle.shape[1],
        side_angles_unit_circle.shape[1],
        side_dihedrals_unit_circle.shape[1],
    ]
    out = Concatenate(axis=1, name="concatenate_angular_inputs")(
        (
            central_angles_unit_circle,
            central_dihedrals_unit_circle,
            side_angles_unit_circle,
            side_dihedrals_unit_circle,
        )
    )
    return splits, out


def _concatenate_inputs(
    p: ADCParameters,
    angles_unit_circle: Union[tf.Tensor, None],
    central_dihedrals_unit_circle: tf.Tensor,
    side_dihedrals_unit_circle: Optional[tf.Tensor] = None,
    input_cartesians_pairwise_defined_shape: Optional[tf.Tensor] = None,
) -> tuple[list[int], tf.Tensor]:
    """Concatenates input Tensors for the AngleDihedralCartesianEncoderMap.

    As the AngleDihedralCartesianEncoderMap model can use either central_dihedrals,
    central_angles and central_dihedrals, central_angles and central_dihedrals and
    side_dihedrals for its Encoder input, these input sources need to be
    concatenated (after they have been projected onto a unit circle). This function
    concatenates these inputs in the correct order and ensures a correct shape
    of the inputs.

    Args:
        p (encodermap.parameters.ADCParameters): A parameter instance.
        angles_unit_circle (Union[tf.Tensor, None]): Can be None, in case only
            the central_dihedrals are used for training. Otherwise, needs to
            be the central angles.
        central_dihedrals_unit_circle (tf.Tensor): The unit circle projected
            central dihedrals.
        side_dihedrals_unit_circle: Can be None, if case the side dihedrals are
            not used for training. Otherwise, needs to be the side dihedrals.
        input_cartesians_pairwise_defined_shape (Optional[tf.Tensor]): The pairwise
            distances of the input cartesians.

    Returns:
        tuple: A tuple containing the following:
            - list[int]: A list of the shape[1] of the input tensors. If only
                dihedrals are used for training, this list has only one entry.
                In the other cases, this list can be used to split the output
                of the decoder again into the constituents of central_angles,
                central_dihedrals, side_dihedrals.
            - tf.Tensor: The concatenated inputs.

    """
    if not p.use_backbone_angles and not p.use_sidechains:
        splits = [central_dihedrals_unit_circle.shape[1]]
        out = central_dihedrals_unit_circle
    elif p.use_backbone_angles and not p.use_sidechains:
        splits = [angles_unit_circle.shape[1], central_dihedrals_unit_circle.shape[1]]
        out = Concatenate(axis=1, name="concatenate_angular_inputs")(
            (angles_unit_circle, central_dihedrals_unit_circle)
        )
    elif p.use_backbone_angles and p.use_sidechains:
        if p.multimer_training is None:
            splits = [
                angles_unit_circle.shape[1],
                central_dihedrals_unit_circle.shape[1],
                side_dihedrals_unit_circle.shape[1],
            ]
            out = Concatenate(axis=1, name="concatenate_angular_inputs")(
                (
                    angles_unit_circle,
                    central_dihedrals_unit_circle,
                    side_dihedrals_unit_circle,
                )
            )
        else:
            splits = [
                angles_unit_circle.shape[1],
                central_dihedrals_unit_circle.shape[1],
                side_dihedrals_unit_circle.shape[1],
                input_cartesians_pairwise_defined_shape.shape[1],
            ]
            out = Concatenate(axis=1, name="input_cartesians_pairwise_defined_shape")(
                (
                    angles_unit_circle,
                    central_dihedrals_unit_circle,
                    side_dihedrals_unit_circle,
                    input_cartesians_pairwise_defined_shape,
                )
            )
    else:
        raise Exception(
            "Only allowed combinations are:\n"
            "   * No sidechains, no backbone angles\n"
            "   * No sidechains, yes backbone angles\n"
            "   * Yes Sidechains, yes backbone angles\n"
            f"Your parameters are: {p.use_sidechains=}. {p.use_backbone_angles=}"
        )
    return splits, out


def _create_inputs_non_periodic_maybe_sparse(
    shape: Union[tuple[int], tuple[int, int]],
    p: ADCParameters,
    name: str,
    sparse: bool,
    reshape: Optional[int] = None,
) -> Union[tf.Tensor, tf.Tensor, Optional[tf.keras.Model]]:
    """Creates an input Tensor.

    Args:
        shape (Union[tuple[int], tuple[int, int]]): The shape can be either a
            tuple with one int (in case of the central distances) or a tuple
            of two ints (in case of central cartesians), in which case, the
            2nd is checked to be 3 (for the xyz coordinates).
        name (str): The name of this input tensor. Will be preceded with 'input_'.
        sparse (bool): Whether a sparse->dense model should be returned. Defaults to False.
        reshape (Optional[int]): Whether the input will be in flattened cartesians
            and thus reshaped to (shape // reshape, reshape). Thus, only the
            reshape 3 is currently used in EncoderMap. If None is specified, the
            output will not be reshaped. Defaults to None.

    Returns:
        tuple: A tuple containing the following:
            - tf.Tensor: The placeholder tensor for the input. If sparse is True,
                this Tensor will first be fed through a Dense layer to use sparse
                matrix multiplication to make it dense again.
            - Union[tf.Tensor, None]: The Dense output of the Tensor, if sparse is True.
            - Union[tf.keras.Model, None]: The model to get from sparse to dense.
                If sparse is False, None will be returned here.

    """
    if len(shape) == 2:
        assert (
            shape[1] == 3
        ), f"Provided tuple of two ints is not of cartesian xyz coordinates!."
    if not sparse:
        placeholder = Input(shape=shape, name=f"input_{name}")
        dense_model = None
        output_placeholder = placeholder
    else:
        assert len(shape) == 1, (
            f"Sparse tensors can only be of rank 2. The Input that accepts these "
            f"tensors can also just be a tuple with a single int in it. You requested "
            f"an input with rank {len(shape)} ({shape=}). Please reevaluate how you use "
            f"`gen_functional_model` and try to reshape the input."
        )
        placeholder = Input(shape=shape, name=f"sparse_input_{name}", sparse=True)

        kernel_initializer = tf.keras.initializers.VarianceScaling()
        bias_initializer = tf.keras.initializers.RandomNormal(0.1, 0.05)
        if os.getenv("CONSTANT_SPARSE_TO_DENSE", "False") == "True":
            warnings.warn("Using constant for to_dense initializers.")
            kernel_initializer = tf.keras.initializers.Constant(1)
            bias_initializer = tf.keras.initializers.Constant(1)

        output_placeholder = Dense(
            units=shape[0],
            trainable=p.trainable_dense_to_sparse,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(p.l2_reg_constant),
        )(placeholder)
        dense_model = tf.keras.Model(
            inputs=placeholder,
            outputs=output_placeholder,
        )
        if reshape is not None:
            output_placeholder = tf.keras.layers.Reshape(
                target_shape=(shape[0] // reshape, reshape),
                input_shape=shape,
                name=f"reshape_sparse_to_dense_{name}",
            )(output_placeholder)
    return placeholder, output_placeholder, dense_model


def _create_inputs_periodic_maybe_sparse(
    shape: int,
    p: ADCParameters,
    name: str,
    sparse: bool,
) -> tuple[tf.Tensor, tf.Tensor, Union[tf.keras.Model, None]]:
    """Creates an input Tensor and also projects it onto a unit circle (returns
    the sin, cos, sin, cos, ...) of the values.

    Args:
        shape (int): The shape can be either a
            tuple with one int (in case of the central distances) or a tuple
            of two ints (in case of central cartesians), in which case, the
            2nd is checked to be 3 (for the xyz coordinates).
        p (encodermap.parameters.ADCParameters): An instance of ADCParameters, which contains info
            about the periodicity of the input space.
        name (str): The name of this input tensor. Will be preceded with 'input_'.
            The to unit_circle input will be called 'input_{name}_to_unit_circle'.
        sparse (bool): Whether a sparse->dense model should be returned.

    Returns:
        tuple: A tuple containing the following:
            - tf.Tensor: The placeholder tensor for the input. If sparse is True,
                this Tensor will first be fed through a Dense layer to use sparse
                matrix multiplication to make it dense again.
            - tf.Tensor: The PeriodicInput of the same tensor.
            - Union[tf.keras.Model, None]: The model to get from sparse to dense.
                If sparse is False, a None will be returned here.

    """
    (
        placeholder,
        dense_placeholder,
        dense_model,
    ) = _create_inputs_non_periodic_maybe_sparse(
        shape=(shape,),
        p=p,
        name=name,
        sparse=sparse,
    )
    unit_circle = PeriodicInput(p, f"input_{name}_to_unit_circle")(dense_placeholder)
    return placeholder, unit_circle, dense_model


################################################################################
# Public Classes
################################################################################


class ADCFunctionalModel(tf.keras.Model):
    """A subclass of tf.keras.Model, that implements the logic for the
    AngleDihedralCartesianEncoderMap.

    """

    def __init__(
        self,
        parameters: ADCParameters,
        inputs: Iterable[tf.Tensor],
        outputs: Iterable[tf.Tensor],
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
    ) -> None:
        """Initialize the Model.

        Args:
            parameters (encodermap.parameters.ADCParameters): An instance of the ADCParameters class.
            inputs (Iterable[tf.Tensor]): The inputs of the model.
            outputs (Iterable[tf.Tensor]): The outputs of the model.
            encoder (tf.keras.Model): The encoder as its own model.
            decoder (tf.keras.Model): The decoder as its own model.

        """
        self.p = parameters
        super().__init__(inputs=inputs, outputs=outputs, name="ADCFunctionalModel")
        self.encoder_model = encoder
        self.decoder_model = decoder

        # train counter
        self._my_train_counter = K.variable(0, "int64", name="train_counter")

    def get_config(self) -> dict[str, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[str, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "parameters": self.p.to_dict(),
                "inputs": [i.shape for i in self.inputs],
                "outputs": [o.shape for o in self.outputs],
                "encoder": tf.keras.saving.serialize_keras_object(self.encoder_model),
                "decoder": tf.keras.saving.serialize_keras_object(self.decoder_model),
            }
        )
        return config

    @classmethod
    def from_config(
        cls: Type[ADCFunctionalModelType],
        config: dict[str, Any],
        custom_objects: Optional[dict[Any, Any]] = None,
    ) -> ADCFunctionalModelType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[str, Any]): A dictionary.
            custom_objects (Optional[dict[str, Any]]): Not needed here, but see
                https://keras.io/guides/serialization_and_saving/ for yourself.

        Returns:
            ADCFunctionalModelType: An instance of the ADCFunctionalModel.

        """
        inputs_config = config.pop("inputs")
        inputs = tf.keras.saving.deserialize_keras_object(inputs_config)
        outputs_config = config.pop("outputs")
        outputs = tf.keras.saving.deserialize_keras_object(outputs_config)
        encoder_config = config.pop("encoder")
        encoder = tf.keras.saving.deserialize_keras_object(encoder_config)
        decoder_config = config.pop("decoder")
        decoder = tf.keras.saving.deserialize_keras_object(decoder_config)
        parameters = config.pop("parameters")

        if "cartesian_pwd_start" in parameters:
            parameters = ADCParameters(**parameters)
        else:
            parameters = Parameters(**parameters)

        # create a new functional model and apply the weights from the encoder and decoder
        input_shapes = tuple([tuple(i[1:]) for i in inputs])
        new_model = gen_functional_model(input_shapes, parameters, write_summary=False)
        new_model.encoder_model.set_weights(encoder.get_weights())
        new_model.decoder_model.set_weights(decoder.get_weights())
        return cls(
            parameters,
            new_model.inputs,
            new_model.outputs,
            new_model.encoder_model,
            new_model.decoder_model,
        )

    @property
    def encoder(self) -> tf.keras.Model:
        return self.encoder_model

    @property
    def decoder(self) -> tf.keras.Model:
        return self.decoder_model

    def compile(self, *args, **kwargs) -> None:
        super().compile(*args, **kwargs)
        self.unpacked_loss_fns = {fn.__name__: fn for fn in self.compiled_loss._losses}

    def get_loss(self, inp: ADCFunctionalModelInputType) -> tf.Tensor:
        # unpack the inputs
        if not self.p.reconstruct_sidechains:
            if self.p.use_sidechains or len(inp) == 5:
                (
                    inp_angles,
                    inp_dihedrals,
                    inp_cartesians,
                    inp_distances,
                    inp_side_dihedrals,
                ) = inp

            elif len(inp) == 4:
                (
                    inp_angles,
                    inp_dihedrals,
                    inp_cartesians,
                    inp_distances,
                ) = inp
            # call the model
            if not self.p.use_sidechains:
                out = self(
                    (
                        inp_angles,
                        inp_dihedrals,
                        inp_cartesians,
                        inp_distances,
                    ),
                    training=True,
                )
            else:
                out = self(inp, training=True)
        else:
            (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
                inp_side_angles,
                inp_side_dihedrals,
                inp_side_distances,
            ) = inp
            out = self(inp, training=True)

        # unpack the outputs
        if self.p.multimer_training is None:
            if self.p.reconstruct_sidechains:
                (
                    out_angles,
                    out_dihedrals,
                    out_side_angles,
                    out_side_dihedrals,
                    back_cartesians,
                    inp_pair,
                    out_pair,
                ) = out
            elif self.p.use_sidechains and not self.p.reconstruct_sidechains:
                (
                    out_angles,
                    out_dihedrals,
                    out_side_dihedrals,
                    back_cartesians,
                    inp_pair,
                    out_pair,
                ) = out
            else:
                (
                    out_angles,
                    out_dihedrals,
                    back_cartesians,
                    inp_pair,
                    out_pair,
                ) = out
        else:
            raise NotImplementedError

        # get the latent
        if self.p.multimer_training is None:
            if self.p.reconstruct_sidechains:
                latent = self.encoder_model(
                    (inp_angles, inp_dihedrals, inp_side_angles, inp_side_dihedrals),
                    training=True,
                )
            elif self.p.use_sidechains and not self.p.reconstruct_sidechains:
                latent = self.encoder_model(
                    (inp_angles, inp_dihedrals, inp_side_dihedrals),
                    training=True,
                )
            elif self.p.use_backbone_angles and not self.p.reconstruct_sidechains:
                latent = self.encoder_model(
                    (inp_angles, inp_dihedrals),
                    training=True,
                )
            else:
                latent = self.encoder_model(
                    inp_dihedrals,
                    training=True,
                )
        else:
            if self.p.multimer_training == "homogeneous_transformation":
                latent = self.encoder_model(
                    (inp_angles, inp_dihedrals, inp_side_dihedrals, inp_cartesians),
                    training=True,
                )
            else:
                raise NotImplementedError

        with tf.name_scope("Cost"):
            loss = 0.0
            # dihedral loss
            loss += self.unpacked_loss_fns["dihedral_loss_func"](
                inp_dihedrals, out_dihedrals
            )

            # angle loss
            # either uses trained angles or mean angles
            loss += self.unpacked_loss_fns["angle_loss_func"](inp_angles, out_angles)

            if self.p.reconstruct_sidechains:
                loss += self.unpacked_loss_fns["angle_loss_func"](
                    inp_side_angles, out_side_angles
                )

            # cartesian loss
            # compares the pairwise distances of the input cartesians
            # and the output cartesians
            # this cost function will slowly be added via a soft-start
            loss += self.unpacked_loss_fns["cartesian_loss_func"](inp_pair, out_pair)

            # distance loss
            # compares the input and the latent, thus needs to be adjusted
            # based on whether the encoder takes angles+dihedrals+side dihedrals,
            # angles+dihedrals, or just dihedrals.
            if self.p.multimer_training is None:
                if self.p.reconstruct_sidechains:
                    loss += self.unpacked_loss_fns["distance_loss_func"](
                        (inp_angles, inp_dihedrals, inp_side_angles, inp_side_dihedrals)
                    )
                elif self.p.use_sidechains and not self.p.reconstruct_sidechains:
                    loss += self.unpacked_loss_fns["distance_loss_func"](
                        (inp_angles, inp_dihedrals, inp_side_dihedrals)
                    )
                elif self.p.use_backbone_angles and not self.p.reconstruct_sidechains:
                    loss += self.unpacked_loss_fns["distance_loss_func"](
                        (inp_angles, inp_dihedrals)
                    )
                else:
                    loss += self.unpacked_loss_fns["distance_loss_func"](inp_dihedrals)
            else:
                if self.p.multimer_training == "homogeneous_transformation":
                    loss += self.unpacked_loss_fns["distance_loss_func"](
                        (inp_angles, inp_dihedrals, inp_side_dihedrals, inp_cartesians)
                    )
                else:
                    raise NotImplementedError

            # cartesian distance cost
            # Compares the input pairwise distances with the latent using a
            # second sigmoid function
            loss += self.unpacked_loss_fns["cartesian_distance_loss_func"](
                inp_pair, latent
            )

            # center loss
            # makes sure, that the latent is in the center and thus depends on
            # the input of the encoder
            if self.p.multimer_training is None:
                if self.p.reconstruct_sidechains:
                    loss += self.unpacked_loss_fns["center_loss_func"](
                        (inp_angles, inp_dihedrals, inp_side_angles, inp_side_dihedrals)
                    )
                elif self.p.use_sidechains and not self.p.reconstruct_sidechains:
                    loss += self.unpacked_loss_fns["center_loss_func"](
                        (inp_angles, inp_dihedrals, inp_side_dihedrals)
                    )
                elif self.p.use_backbone_angles and not self.p.reconstruct_sidechains:
                    loss += self.unpacked_loss_fns["center_loss_func"](
                        (inp_angles, inp_dihedrals)
                    )
                else:
                    loss += self.unpacked_loss_fns["center_loss_func"](inp_dihedrals)
            else:
                if self.p.multimer_training == "homogeneous_transformation":
                    loss += self.unpacked_loss_fns["center_loss_func"](
                        (inp_angles, inp_dihedrals, inp_side_dihedrals, inp_cartesians)
                    )
                else:
                    raise NotImplementedError

            # reg loss
            # just add the squared weights of all trainable layers
            loss += self.unpacked_loss_fns["regularization_loss_func"]()

            # side dihedral loss
            if self.p.use_sidechains:
                loss += self.unpacked_loss_fns["side_dihedral_loss_func"](
                    inp_side_dihedrals, out_side_dihedrals
                )
            tf.summary.scalar("Combined Cost", loss)

        return loss, inp_cartesians, back_cartesians

    def train_step(self, data: ADCFunctionalModelInputType) -> None:
        """Can receive two types of data.

        * use_backbone_angles = False, use_sidechains = False:
            Will receive a four-tuple in the order: angles, dihedrals, cartesians,
            distances. The angles will be used to construct mean angles.
        * use_backbone_angles = True, use_sidechains = False:
            Will receive the same four-tuple as above, but the angles will be
            fed through the autoencoder.
        * use_backbone_angles = True, use_sidechains = True:
            Will receive a five-tuple in the order: angles, dihedrals, cartesians,
            distances, side dihedrals. The angles, central dihedrals and side
            dihedrals will be fed through the autoencoder.

        """

        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._my_train_counter)
            loss, inp_cartesians, out_cartesians = self.get_loss(data)
            loggable_encoder_layers = [
                l for l in self.encoder_model.layers if l.__class__.__name__ == "Dense"
            ]
            loggable_decoder_layers = [
                l for l in self.decoder_model.layers if l.__class__.__name__ == "Dense"
            ]
            for l in loggable_encoder_layers + loggable_decoder_layers:
                add_layer_summaries(l, step=self._my_train_counter)

        # Compute Gradients
        if not self.p.trainable_dense_to_sparse:
            trainable_vars = (
                self.encoder_model.trainable_variables
                + self.decoder_model.trainable_variables
            )
        else:
            trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # if gradients are NaN, we skip training
        # tf.cond(
        #     tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients]),
        #     true_fn=lambda: self.fn_true_ignore_grad(gradients, trainable_vars),
        #     false_fn=lambda: self.fn_false_apply_grad(gradients, trainable_vars),
        # )

        # clip the gradients
        # raise Exception(f"{[tf.reduce_max(g).numpy() for g in gradients]=}")

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        for metric in self.compiled_metrics._metrics:
            metric.update_state(data, self(data))

        # update train counter because tensorflow seems to have deprecated it
        self._my_train_counter.assign_add(1)
        return {m.name: m.result() for m in self.compiled_metrics._metrics} | {
            "loss": loss,
        }


@testing
class ADCFunctionalModelTesting(tf.keras.Model):
    """A subclass of tf.keras.Model, that implements the logic for the
    AngleDihedralCartesianEncoderMap.

    """

    def __init__(
        self,
        parameters: ADCParameters,
        inputs: Iterable[tf.Tensor],
        outputs: Iterable[tf.Tensor],
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
    ) -> None:
        """Initialize the Model.

        Args:
            parameters (encodermap.parameters.ADCParameters): An instance of the ADCParameters class.
            inputs (Iterable[tf.Tensor]): The inputs of the model.
            outputs (Iterable[tf.Tensor]): The outputs of the model.
            encoder (tf.keras.Model): The encoder as its own model.
            decoder (tf.keras.Model): The decoder as its own model.

        """

        self.p = parameters
        super().__init__(inputs=inputs, outputs=outputs, name="ADCFunctionalModel")
        self.encoder_model = encoder
        self.decoder_model = decoder

        # loggable layers
        self.loggable_encoder_layers = [
            l for l in self.encoder_model.layers if l.__class__.__name__ == "Dense"
        ]
        self.loggable_decoder_layers = [
            l for l in self.decoder_model.layers if l.__class__.__name__ == "Dense"
        ]

    def get_config(self) -> dict[str, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[str, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "parameters": self.p.to_dict(),
                "inputs": [i.shape for i in self.inputs],
                "outputs": [o.shape for o in self.outputs],
                "encoder": tf.keras.saving.serialize_keras_object(self.encoder_model),
                "decoder": tf.keras.saving.serialize_keras_object(self.decoder_model),
            }
        )
        return config

    def train_step(self, data: ADCFunctionalModelInputType) -> Any:
        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._my_train_counter)
            if self.p.use_sidechains:
                y_pred = self(data, training=True)
            else:
                y_pred = self(data[:-1], training=True)

            loss = self.compute_loss(y=data, y_pred=y_pred)

        for l in self.loggable_encoder_layers + self.loggable_decoder_layers:
            add_layer_summaries(l, step=self._my_train_counter)

        trainable_vars = (
            self.encoder_model.trainable_variables
            + self.decoder_model.trainable_variables
        )
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # for metric in self.metrics:
        #     if metric.name == "loss":
        #         metric.update_state(loss)
        #     elif metric.__class__.__name__ == "Mean":
        #         continue
        #     else:
        #         metric.update_state(data, y_pred)
        # return {m.name: m.result for m in self.metrics}
        return {"loss": loss}

    @classmethod
    def from_config(
        cls: Type[ADCFunctionalModelTestingType],
        config: dict[str, Any],
        custom_objects: Optional[dict[Any, Any]] = None,
    ) -> ADCFunctionalModelTestingType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[str, Any]): A dictionary.
            custom_objects (Optional[dict[str, Any]]): Not needed here, but see
                https://keras.io/guides/serialization_and_saving/ for yourself.

        Returns:
            ADCFunctionalModelTestingType: An instance of the ADCFunctionalModelTesting.

        """
        inputs_config = config.pop("inputs")
        inputs = tf.keras.saving.deserialize_keras_object(inputs_config)
        outputs_config = config.pop("outputs")
        outputs = tf.keras.saving.deserialize_keras_object(outputs_config)
        encoder_config = config.pop("encoder")
        encoder = tf.keras.saving.deserialize_keras_object(encoder_config)
        decoder_config = config.pop("decoder")
        decoder = tf.keras.saving.deserialize_keras_object(decoder_config)
        parameters = config.pop("parameters")

        if "cartesian_pwd_start" in parameters:
            parameters = ADCParameters(**parameters)
        else:
            parameters = Parameters(**parameters)

        # create a new functional model and apply the weights from the encoder and decoder
        input_shapes = tuple([tuple(i[1:]) for i in inputs])
        new_model = gen_functional_model(input_shapes, parameters, write_summary=False)
        new_model.encoder_model.set_weights(encoder.get_weights())
        new_model.decoder_model.set_weights(decoder.get_weights())
        return cls(
            parameters,
            new_model.inputs,
            new_model.outputs,
            new_model.encoder_model,
            new_model.decoder_model,
        )

    @property
    def encoder(self) -> tf.keras.Model:
        return self.encoder_model

    @property
    def decoder(self) -> tf.keras.Model:
        return self.decoder_model


class ADCSparseFunctionalModel(ADCFunctionalModel):
    def __init__(
        self,
        parameters: ADCParameters,
        inputs: Iterable[tf.Tensor],
        outputs: Iterable[tf.Tensor],
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        get_dense_model_central_angles: tf.keras.Model,
        get_dense_model_central_dihedrals: tf.keras.Model,
        get_dense_model_cartesians: tf.keras.Model,
        get_dense_model_distances: tf.keras.Model,
        get_dense_model_side_dihedrals: Union[tf.keras.Model, None],
    ) -> None:
        """Instantiate the Model.

        Args:
            parameters (encodermap.parameters.ADCParameters): An instance of the ADCParameters class.
            inputs (Iterable[tf.Tensor]): The inputs of the model.
            outputs (Iterable[tf.Tensor]): The outputs of the model.
            encoder (tf.keras.Model): The encoder as its own model.
            decoder (tf.keras.Model): The decoder as its own model.
            get_dense_model_central_angles (tf.keras.Model): A model with a
                single dense layer that uses sparse matrix multiplication to
                transform the sparse tensor.
            get_dense_model_central_dihedrals (tf.keras.Model): A model with a
                single dense layer that uses sparse matrix multiplication to
                transform the sparse tensor.
            get_dense_model_cartesians (tf.keras.Model): A model with a
                single dense layer that uses sparse matrix multiplication to
                transform the sparse tensor.
            get_dense_model_distances (tf.keras.Model): A model with a
                single dense layer that uses sparse matrix multiplication to
                transform the sparse tensor.
            get_dense_model_side_dihedrals (Union[tf.keras.Model, None]):
                A model with a single dense layer that uses sparse matrix
                multiplication to transform the sparse tensor. Can be None,
                for when these angular inputs are not used for training.

        """
        super().__init__(parameters, inputs, outputs, encoder, decoder)
        self.get_dense_model_central_angles = get_dense_model_central_angles
        self.get_dense_model_central_dihedrals = get_dense_model_central_dihedrals
        self.get_dense_model_cartesians = get_dense_model_cartesians
        self.get_dense_model_distances = get_dense_model_distances
        self.get_dense_model_side_dihedrals = get_dense_model_side_dihedrals
        if self.get_dense_model_cartesians is not None:
            self.reshape_layer = tf.keras.layers.Reshape(
                target_shape=(inputs[2].shape[1] // 3, 3),
                input_shape=(inputs[2].shape[1],),
                name="reshape_sparse_to_dense_internally",
            )
            self.reshape_layer.build(
                input_shape=(self.p.batch_size, inputs[2].shape[1])
            )
        else:
            self.reshape_layer = None

    def get_config(self) -> dict[str, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[str, Any]: A dict with the serializable objects.

        """
        config = super().get_config().copy()
        config.update(
            {
                "get_dense_model_central_angles": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_central_angles
                ),
                "get_dense_model_central_dihedrals": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_central_dihedrals
                ),
                "get_dense_model_cartesians": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_cartesians
                ),
                "get_dense_model_distances": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_distances
                ),
                "get_dense_model_side_dihedrals": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_side_dihedrals
                ),
            }
        )
        return config

    @classmethod
    def from_config(
        cls: Type[ADCSparseFunctionalModelType],
        config: dict[str, Any],
        custom_objects: Optional[dict[Any, Any]] = None,
    ) -> ADCSparseFunctionalModelType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[str, Any]): A dictionary.
            custom_objects (Optional[dict[str, Any]]): Not needed here, but see
                https://keras.io/guides/serialization_and_saving/ for yourself.

        Returns:
            ADCSparseFunctionalModelType: An instance of the ADCSparseFunctionalModel.

        """
        inputs_config = config.pop("inputs")
        inputs = tf.keras.saving.deserialize_keras_object(inputs_config)
        outputs_config = config.pop("outputs")
        outputs = tf.keras.saving.deserialize_keras_object(outputs_config)
        encoder_config = config.pop("encoder")
        encoder = tf.keras.saving.deserialize_keras_object(encoder_config)
        decoder_config = config.pop("decoder")
        decoder = tf.keras.saving.deserialize_keras_object(decoder_config)

        # get the dense models
        get_dense_model_central_angles = config.pop("get_dense_model_central_angles")
        get_dense_model_central_angles = tf.keras.saving.deserialize_keras_object(
            get_dense_model_central_angles
        )

        get_dense_model_central_dihedrals = config.pop(
            "get_dense_model_central_dihedrals"
        )
        get_dense_model_central_dihedrals = tf.keras.saving.deserialize_keras_object(
            get_dense_model_central_dihedrals
        )

        get_dense_model_cartesians = config.pop("get_dense_model_cartesians")
        get_dense_model_cartesians = tf.keras.saving.deserialize_keras_object(
            get_dense_model_cartesians
        )

        get_dense_model_distances = config.pop("get_dense_model_distances")
        get_dense_model_distances = tf.keras.saving.deserialize_keras_object(
            get_dense_model_distances
        )

        get_dense_model_side_dihedrals = config.pop("get_dense_model_side_dihedrals")
        get_dense_model_side_dihedrals = tf.keras.saving.deserialize_keras_object(
            get_dense_model_side_dihedrals
        )

        parameters = config.pop("parameters")
        if "cartesian_pwd_start" in parameters:
            parameters = ADCParameters(**parameters)
        else:
            parameters = Parameters(**parameters)

        # create a new functional model and apply the weights from the encoder and decoder
        input_shapes = tuple([tuple(i[1:]) for i in inputs])
        new_model = gen_functional_model(
            input_shapes,
            parameters,
            write_summary=False,
            sparse=True,
        )
        if len(encoder.get_weights()) != len(new_model.encoder_model.get_weights()):
            # here, we can assume that the model was trained with
            # only sparse sidechains
            new_model = gen_functional_model(
                input_shapes,
                parameters,
                write_summary=False,
                sparse=False,
                sidechain_only_sparse=True,
            )

        # for l in new_model.encoder_model.layers:
        #     print(f"new_model.encoder layer {l=}")
        # for l in encoder.layers:
        #     print(f"encoder layer {l=}")

        new_model.encoder_model.set_weights(encoder.get_weights())
        new_model.decoder_model.set_weights(decoder.get_weights())

        if new_model.get_dense_model_central_angles is not None:
            new_model.get_dense_model_central_angles.set_weights(
                get_dense_model_central_angles.get_weights()
            )
        if new_model.get_dense_model_central_dihedrals is not None:
            new_model.get_dense_model_central_dihedrals.set_weights(
                get_dense_model_central_dihedrals.get_weights()
            )
        if new_model.get_dense_model_cartesians is not None:
            new_model.get_dense_model_cartesians.set_weights(
                get_dense_model_cartesians.get_weights()
            )
        if new_model.get_dense_model_distances is not None:
            new_model.get_dense_model_distances.set_weights(
                get_dense_model_distances.get_weights()
            )
        if new_model.get_dense_model_side_dihedrals is not None:
            new_model.get_dense_model_side_dihedrals.set_weights(
                get_dense_model_side_dihedrals.get_weights()
            )

        return cls(
            parameters,
            new_model.inputs,
            new_model.outputs,
            new_model.encoder_model,
            new_model.decoder_model,
            new_model.get_dense_model_central_angles,
            new_model.get_dense_model_central_dihedrals,
            new_model.get_dense_model_cartesians,
            new_model.get_dense_model_distances,
            new_model.get_dense_model_side_dihedrals,
        )

    def get_loss(self, inp):
        # unpack the inputs
        if self.p.use_sidechains and len(inp) == 5:
            (
                sparse_inp_angles,
                sparse_inp_dihedrals,
                sparse_inp_cartesians,
                sparse_inp_distances,
                sparse_side_dihedrals,
            ) = inp
        else:
            (
                sparse_inp_angles,
                sparse_inp_dihedrals,
                sparse_inp_cartesians,
                sparse_inp_distances,
            ) = inp

        if isinstance(sparse_inp_angles, tf.sparse.SparseTensor):
            inp_angles = self.get_dense_model_central_angles(
                sparse_inp_angles, training=True
            )
        else:
            inp_angles = sparse_inp_angles
        if isinstance(sparse_inp_dihedrals, tf.sparse.SparseTensor):
            inp_dihedrals = self.get_dense_model_central_dihedrals(
                sparse_inp_dihedrals, training=True
            )
        else:
            inp_dihedrals = sparse_inp_dihedrals
        if isinstance(sparse_inp_cartesians, tf.sparse.SparseTensor):

            inp_cartesians = self.get_dense_model_cartesians(
                sparse_inp_cartesians, training=True
            )
        else:
            inp_cartesians = sparse_inp_cartesians
        if isinstance(sparse_inp_distances, tf.sparse.SparseTensor):
            inp_distances = self.get_dense_model_distances(
                sparse_inp_distances, training=True
            )
        else:
            inp_distances = sparse_inp_distances

        if self.p.use_sidechains:
            if isinstance(sparse_side_dihedrals, tf.sparse.SparseTensor):
                inp_side_dihedrals = self.get_dense_model_side_dihedrals(
                    sparse_side_dihedrals, training=True
                )
            else:
                inp_side_dihedrals = sparse_side_dihedrals

        # make them into an Iterable again
        if self.p.use_sidechains:
            data = (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
                inp_side_dihedrals,
            )
        else:
            data = (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
            )

        # call the loss
        # when we are using sparse `inp_cartesians`, index `[1]` of the output
        # of `super().get_loss(data)` contains 'central_cartesians' in the flattened
        # rank 2 form and we need to transform it
        resulting_loss, _, out_cartesians = super().get_loss(data)
        if self.get_dense_model_cartesians is not None:
            inp_cartesians = self.reshape_layer(inp_cartesians)
        return resulting_loss, inp_cartesians, out_cartesians


class ADCFunctionalModelSidechainReconstruction(ADCSparseFunctionalModel):
    def __init__(
        self,
        parameters: ADCParameters,
        inputs: Iterable[tf.Tensor],
        outputs: Iterable[tf.Tensor],
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            encoder=encoder,
            decoder=decoder,
            get_dense_model_central_angles=kwargs["get_dense_model_central_angles"],
            get_dense_model_central_dihedrals=kwargs[
                "get_dense_model_central_dihedrals"
            ],
            get_dense_model_cartesians=kwargs["get_dense_model_cartesians"],
            get_dense_model_distances=kwargs["get_dense_model_central_distances"],
            get_dense_model_side_dihedrals=kwargs["get_dense_model_side_dihedrals"],
        )
        self.get_dense_model_side_angles = kwargs["get_dense_model_side_angles"]
        self.get_dense_model_side_distances = kwargs["get_dense_model_side_distances"]

    def get_loss(self, inp: tuple[tf.Tensor, ...]):
        # unpack the inputs
        (
            sparse_inp_central_angles,
            sparse_inp_central_dihedrals,
            sparse_inp_all_cartesians,
            sparse_inp_central_distances,
            sparse_inp_side_angles,
            sparse_inp_side_dihedrals,
            sparse_inp_side_distances,
        ) = inp

        # central angles
        if isinstance(sparse_inp_central_angles, tf.sparse.SparseTensor):
            inp_central_angles = self.get_dense_model_central_angles(
                sparse_inp_central_angles, training=True
            )
        else:
            inp_central_angles = sparse_inp_central_angles

        # central dihedrals
        if isinstance(sparse_inp_central_dihedrals, tf.sparse.SparseTensor):
            inp_central_dihedrals = self.get_dense_model_central_dihedrals(
                sparse_inp_central_dihedrals, training=True
            )
        else:
            inp_central_dihedrals = sparse_inp_central_dihedrals

        # all cartesians
        if isinstance(sparse_inp_all_cartesians, tf.sparse.SparseTensor):
            inp_all_cartesians = self.get_dense_model_cartesians(
                sparse_inp_all_cartesians, training=True
            )
        else:
            inp_all_cartesians = sparse_inp_all_cartesians

        # central distances
        if isinstance(sparse_inp_central_distances, tf.sparse.SparseTensor):
            inp_central_distances = self.get_dense_model_distances(
                sparse_inp_central_distances, training=True
            )
        else:
            inp_central_distances = sparse_inp_central_distances

        # side_angles
        if isinstance(sparse_inp_side_angles, tf.sparse.SparseTensor):
            inp_side_angles = self.get_dense_model_side_angles(
                sparse_inp_side_angles, training=True
            )
        else:
            inp_side_angles = sparse_inp_side_angles

        # side dihedrals
        if isinstance(sparse_inp_side_dihedrals, tf.sparse.SparseTensor):
            inp_side_dihedrals = self.get_dense_model_side_dihedrals(
                sparse_inp_side_dihedrals, training=True
            )
        else:
            inp_side_dihedrals = sparse_inp_side_dihedrals

        # side distances
        if isinstance(sparse_inp_side_distances, tf.sparse.SparseTensor):
            inp_side_distances = self.get_dense_model_side_distances(
                sparse_inp_side_distances, training=True
            )
        else:
            inp_side_distances = sparse_inp_side_distances

        data = (
            inp_central_angles,
            inp_central_dihedrals,
            inp_all_cartesians,
            inp_central_distances,
            inp_side_angles,
            inp_side_dihedrals,
            inp_side_distances,
        )

        # call the loss
        resulting_loss, _, out_cartesians = ADCFunctionalModel.get_loss(self, data)
        if self.get_dense_model_cartesians is not None:
            inp_all_cartesians = self.reshape_layer(inp_all_cartesians)
        return resulting_loss, inp_all_cartesians, out_cartesians

    def get_config(self) -> dict[str, Any]:
        sidechain_info = self.p.sidechain_info
        config = super().get_config().copy()
        config.update(
            {
                "sidechain_info": sidechain_info,
                "get_dense_model_side_angles": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_side_angles
                ),
                "get_dense_model_side_distances": tf.keras.saving.serialize_keras_object(
                    self.get_dense_model_side_distances
                ),
            }
        )
        return config

    @classmethod
    def from_config(
        cls: Type[ADCFunctionalModelSidechainReconstructionType],
        config: dict[Any, Any],
    ) -> ADCFunctionalModelSidechainReconstructionType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[Any, Any]): A dictionary.

        Returns:
            BackMapLayerType: An instance of the BackMapLayer.

        """
        raise Exception(f"Also put the sidechain_indices back into the parameters")
        return cls(parameters=p, **config)


class SequentialModel(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        parameters: Optional[Parameters] = None,
        sparse: bool = False,
        get_dense_model: Optional[tf.keras.Model] = None,
        # reload_layers: Optional[Sequence[tf.keras.Model]] = None,
    ) -> None:
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters
        super().__init__()
        self.sparse = sparse
        self.input_dim = input_dim
        self.get_dense_model = get_dense_model

        # tensors for using tf.cond inside self.train_step()
        self.update_step = tf.constant(self.p.summary_step, dtype="int64")
        self.debug_tensor = tf.constant(self.p.tensorboard, dtype="bool")

        # periodicity doubles the inputs and outputs
        if self.p.periodicity < float("inf"):
            self.input_dim *= 2

        # define regularizer
        regularizer = tf.keras.regularizers.l2(self.p.l2_reg_constant)

        # rename empty string in parameters to None
        activation_functions = list(
            map(lambda x: x if x != "" else None, self.p.activation_functions)
        )

        # define how layers are stacked
        layer_data = list(
            zip(
                self.p.n_neurons + self.p.n_neurons[-2::-1],
                activation_functions[1:] + activation_functions[-2::-1],
            )
        )
        # add a layer that reshapes the output
        layer_data.append([self.input_dim, None])

        # decide layer names
        names = []
        for i, (n_neurons, act_fun) in enumerate(layer_data):
            if i < len(self.p.n_neurons) - 1:
                name = f"Encoder_{i}"
            elif i > len(self.p.n_neurons) - 1:
                ind = i - len(self.p.n_neurons)
                name = f"Decoder_{ind}"
            else:
                name = "Latent"
            names.append(name)
        layer_data = list((*i, j) for i, j in zip(layer_data, names))

        # define encoder and decoder layers
        neurons = [i[0] for i in layer_data]
        bottleneck_index = neurons.index(min(neurons)) + 1
        self.encoder_layers = layer_data[:bottleneck_index]
        self.decoder_layers = layer_data[bottleneck_index:]

        # input
        # Instead of using InputLayer use Dense with kwarg input_shape
        # allows model to be reloaded better <- weird english... reloaded better
        if self.sparse:
            shape = self.input_dim
            if self.p.periodicity < float("inf"):
                shape /= 2
            _input_layer = Input(
                shape=(int(shape),),
                sparse=True,
            )
            x = Dense(shape)(_input_layer)
            self.get_dense_model = tf.keras.Model(
                inputs=_input_layer,
                outputs=x,
            )

        input_layer = tf.keras.layers.Dense(
            input_shape=(self.input_dim,),
            units=self.encoder_layers[0][0],
            activation=self.encoder_layers[0][1],
            name=self.encoder_layers[0][2],
            kernel_initializer=tf.initializers.VarianceScaling(),
            kernel_regularizer=regularizer,
            bias_initializer=tf.initializers.RandomNormal(0.1, 0.05),
        )

        # encoder
        self.encoder_model = tf.keras.Sequential(
            [input_layer]
            + [
                tf.keras.layers.Dense(
                    n_neurons,
                    activation=act_fun,
                    name=name,
                    kernel_initializer=tf.initializers.VarianceScaling(),
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.initializers.RandomNormal(0.1, 0.05),
                )
                for n_neurons, act_fun, name in self.encoder_layers[1:]
            ],
            name="Encoder",
        )

        # decoder
        self.decoder_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    n_neurons,
                    activation=act_fun,
                    name=name,
                    kernel_initializer=tf.initializers.VarianceScaling(),
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.initializers.RandomNormal(0.1, 0.05),
                )
                for n_neurons, act_fun, name in self.decoder_layers
            ],
            name="Decoder",
        )

        # build
        self.build(input_shape=(1, self.input_dim))

        # train counter
        self._my_train_counter = K.variable(0, "int64", name="train_counter")

    @classmethod
    def from_config(
        cls: Type[SequentialModelType],
        config: dict[str, Any],
        custom_objects: Optional[dict[Any, Any]] = None,
    ) -> SequentialModelType:
        """Reconstructs this keras serializable from a dict.

        Args:
            config (dict[str, Any]): A dictionary.
            custom_objects (Optional[dict[str, Any]]): Not needed here, but see
                https://keras.io/guides/serialization_and_saving/ for yourself.

        Returns:
            SequentialModelType: An instance of the SequentialModel.

        """
        input_dim = config.pop("input_dim")
        sparse = config.pop("sparse")
        parameters = config.pop("parameters")
        if "cartesian_pwd_start" in parameters:
            parameters = ADCParameters(**parameters)
        else:
            parameters = Parameters(**parameters)

        if parameters.periodicity < float("inf"):
            input_dim = input_dim // 2
        else:
            pass

        encoder_config = config.pop("encoder")
        encoder = tf.keras.saving.deserialize_keras_object(encoder_config)
        decoder_config = config.pop("decoder")
        decoder = tf.keras.saving.deserialize_keras_object(decoder_config)
        get_dense_model = config.pop("get_dense_model")
        if get_dense_model is not None:
            get_dense_model = tf.keras.saving.deserialize_keras_object(get_dense_model)

        new_class = cls(
            input_dim=input_dim,
            parameters=parameters,
            sparse=sparse,
            get_dense_model=get_dense_model,
        )

        new_class.encoder_model.set_weights(encoder.get_weights())
        new_class.decoder_model.set_weights(decoder.get_weights())

        if parameters.periodicity < float("inf"):
            new_class.compute_output_shape(input_shape=(1, input_dim))
        else:
            new_class.compute_output_shape(input_shape=(1, input_dim))

        return new_class

    def get_config(self) -> dict[str, Any]:
        """Serializes this keras serializable.

        Returns:
            dict[str, Any]: A dict with the serializable objects.

        """
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "parameters": self.p.to_dict(),
                "sparse": self.sparse,
                "encoder": tf.keras.saving.serialize_keras_object(self.encoder_model),
                "decoder": tf.keras.saving.serialize_keras_object(self.decoder_model),
            }
        )
        if self.get_dense_model is not None:
            config["get_dense_model"] = tf.keras.saving.serialize_keras_object(
                self.get_dense_model
            )
        else:
            config["get_dense_model"] = None
        return config

    def build(self, input_shape):
        input_shape = self.encoder_model.input_shape
        if isinstance(input_shape, tuple):
            input_shape = input_shape[1]

        if self.p.periodicity < float("inf"):
            super().build(input_shape=(1, input_shape // 2))
        else:
            super().build(input_shape=(1, input_shape))

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        try:
            self.unpacked_loss_fns = {
                fn.__name__: fn for fn in self.compiled_loss._losses
            }
        except AttributeError:
            for i in dir(self):
                if "loss" in i:
                    print(i)
            print(self._callable_losses)
            raise

    def encoder(self, x, training=False):
        """In the sequential model, the encoder is a method (as oppes to a model).

        This method handles the input, when the periodicity of the input data
        is greater than float('inf').

        Args:
            x (Union[np.ndarray, tf.Tensor): The input.
            training (bool): Whether we are training and compute gradients.

        Returns:
            Union[np.ndarray, tf.Tensor]: The output of the encoder.

        """
        if self.sparse:
            x = self.get_dense_model(x)
        if self.p.periodicity < float("inf"):
            if self.p.periodicity != 2 * pi:
                x = x / self.p.periodicity * 2 * pi
            x = tf.concat([tf.sin(x), tf.cos(x)], 1)
        return self.encoder_model(x, training=training)

    def decoder(self, x, training=False):
        x = self.decoder_model(x, training=training)
        if self.p.periodicity < float("inf"):
            x = tf.atan2(*tf.split(x, 2, 1))
            if self.p.periodicity != 2 * pi:
                x = x / (2 * pi) * self.p.periodicity
        return x

    def call(self, x, training=False):
        # encode and decode
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def train_step(self, data):
        """Overwrites the normal train_step. What is different?

        Not much. Even the provided data is expected to be a tuple of (data, classes) (x, y) in classification tasks.
        The data is unpacked, and y is discarded, because the Autoencoder Model is a regression task.

        Args:
            data (tuple): The (x, y) data of this train step.

        """
        x, _ = data
        if self.sparse:
            x = self.get_dense_model(x)

        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._my_train_counter)
            with tf.name_scope("Cost"):
                loss = 0.0
                for l in self.compiled_loss._losses:
                    loss += l(x, self(x, training=True))
                tf.summary.scalar("Combined Cost", loss)
            for l in self.encoder_model.layers + self.decoder_model.layers:
                add_layer_summaries(l, step=self._my_train_counter)

        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(x, self(x))
        # update train counter because tensorflow seems to have deprecated it
        self._my_train_counter.assign_add(1)
        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics}, **{"loss": loss}}


@testing
class ADCSequentialModel(SequentialModel):
    def __init__(self, input_dim, parameters=None, reload_layers=None, sparse=False):
        warnings.warn("check split")
        if parameters is None:
            self.p = ADCParameters()
        else:
            self.p = parameters
        self.multiples = tf.TensorShape((self.p.batch_size, 1))
        super(ADCSequentialModel, self).__init__(input_dim, self.p, reload_layers)

    def call(self, x, training=False):
        # encode and decode
        if isinstance(x, tf.Tensor):
            pass
        elif isinstance(x, tuple):
            (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
                inp_side_dihedrals,
            ) = x
            if not self.p.use_backbone_angles and not self.p.use_sidechains:
                x = inp_dihedrals
            elif self.p.use_backbone_angles and not self.p.use_sidechains:
                x = tf.concat([inp_angles, inp_dihedrals], 1)
            elif self.p.use_backbone_angles and self.p.use_sidechains:
                x = tf.concat([inp_angles, inp_dihedrals, inp_side_dihedrals], 1)
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def call_and_map_back(
        self, x, distances, angles, dihedrals, cartesians, splits, side_dihedrals=None
    ):
        # latent = self.encoder(x, training=False)
        out = self(x, training=True)
        latent = self.encoder(x, training=True)

        # unpack out
        if splits is None:
            out_dihedrals = out
            out_angles = tf.tile(
                tf.expand_dims(tf.reduce_mean(angles, 0), 0), multiples=self.multiples
            )
        elif len(splits) == 2:
            out_angles, out_dihedrals = tf.split(out, splits, 1)
        elif len(splits) == 3:
            out_angles, out_dihedrals, out_side_dihedrals = tf.split(out, splits, 1)

        # do back-mapping
        back_mean_lengths = tf.expand_dims(tf.reduce_mean(distances, 0), 0)
        back_chain_in_plane = chain_in_plane(back_mean_lengths, out_angles)
        back_cartesians = dihedrals_to_cartesian_tf(
            out_dihedrals + pi, back_chain_in_plane
        )

        # get pairwise distances of CA atoms
        inp_pair = pairwise_dist(
            cartesians[
                :,
                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
            ],
            flat=True,
        )
        out_pair = pairwise_dist(
            back_cartesians[
                :,
                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
            ],
            flat=True,
        )

        with tf.name_scope("Cost"):
            loss = 0.0
            # dihedral loss
            loss += self.unpacked_loss_fns["dihedral_loss_func"](
                dihedrals, out_dihedrals
            )
            # angle loss
            loss += self.unpacked_loss_fns["angle_loss_func"](angles, out_angles)
            # cartesian loss
            loss += self.unpacked_loss_fns["cartesian_loss_func"](inp_pair, out_pair)
            # distance loss
            loss += self.unpacked_loss_fns["distance_loss_func"](x)
            # cartesian distance cost
            loss += self.unpacked_loss_fns["cartesian_distance_loss_func"](
                inp_pair, latent
            )
            # center loss
            loss += self.unpacked_loss_fns["center_loss_func"](x)
            # reg loss
            loss += self.unpacked_loss_fns["regularization_loss_func"]()
            if self.p.use_sidechains:
                loss += self.unpacked_loss_fns["side_dihedral_loss_func"](
                    side_dihedrals, out_side_dihedrals
                )
            tf.summary.scalar("Combined Cost", loss)
        return loss

    def train_step(self, data):
        # unpack the data
        (
            inp_angles,
            inp_dihedrals,
            inp_cartesians,
            inp_distances,
            inp_side_dihedrals,
        ) = data
        if not self.p.use_backbone_angles and not self.p.use_sidechains:
            main_inputs = inp_dihedrals
            splits = None
        elif self.p.use_backbone_angles and not self.p.use_sidechains:
            main_inputs = tf.concat([inp_angles, inp_dihedrals], 1)
            splits = [inp_angles.shape[1], inp_dihedrals.shape[1]]
        elif self.p.use_backbone_angles and self.p.use_sidechains:
            # ToDo: make sure the splits work here. There seems to be different outputs from tf and np.
            main_inputs = tf.concat([inp_angles, inp_dihedrals, inp_side_dihedrals], 1)
            splits = [
                inp_angles.shape[1],
                inp_dihedrals.shape[1],
                inp_side_dihedrals.shape[1],
            ]

        # gradient tape to calculate loss for backmapping
        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._my_train_counter)
            if self.p.use_sidechains:
                loss = self.call_and_map_back(
                    main_inputs,
                    inp_distances,
                    inp_angles,
                    inp_dihedrals,
                    inp_cartesians,
                    splits,
                    inp_side_dihedrals,
                )
            else:
                loss = self.call_and_map_back(
                    main_inputs,
                    inp_distances,
                    inp_angles,
                    inp_dihedrals,
                    inp_cartesians,
                    splits,
                )
            for l in self.encoder_model.layers + self.decoder_model.layers:
                add_layer_summaries(l, step=self._my_train_counter)

        # optimization happens here
        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(data, self(data))
        # Return a dict mapping metric names to current value
        # Add loss to the dict so the ProgressBar callback can pick it up
        return {**{m.name: m.result() for m in self.metrics}, **{"loss": loss}}
