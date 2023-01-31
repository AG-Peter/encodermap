# -*- coding: utf-8 -*-
# encodermap/parameters/parameters.py
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
"""Parameter Classes for Encodermap.

This module contains parameter classes which are used to hold information for
the encodermap autoencoder. Parameters can be set from keyworded arguments, by
overwriting the class attributes or by reading them from .json, .yaml or ASCII files.

Features:
    * Setting and saving Parameters with the Parameter class.
    * Loading parameters from disk and continue where you left off.
    * The Parameter and ACDParamter class contains already good default values.

"""


################################################################################
# Imports
################################################################################


from __future__ import annotations

import datetime
import json
import os
from math import pi
from textwrap import wrap

from .._optional_imports import _optional_import
from ..misc.misc import _datetime_windows_and_linux_compatible, printTable

################################################################################
# Optional Imports
################################################################################


yaml = _optional_import("yaml")


################################################################################
# Typing
################################################################################


from typing import TYPE_CHECKING, Dict, Optional, TypeVar, Union

ParametersData = Union[
    float, int, str, bool, list[int], list[str], list[float], tuple[int, None], None
]
ParametersDict = Dict[str, ParametersData]
ParametersType = TypeVar("Parameters", bound="Parent")
ADCParametersType = TypeVar("Parameters", bound="Parent")


################################################################################
# Globals
################################################################################


__all__ = ["Parameters", "ADCParameters"]


################################################################################
# Functions
################################################################################


def search_and_replace(
    file_path: str,
    search_pattern: str,
    replacement: str,
    out_path: Optional[str] = None,
    backup: bool = True,
) -> None:
    """Searches for a pattern in a text file and replaces it with the replacement

    Args:
        file_path (str): File path of the file to replace the text pattern in.
        search_pattern (str): Pattern to search for.
        replacement (str): What to replace `search_pattern` with.
        out_path (str, optional): path where to write the output file.
            If no path is given the original file will be replaced. Defaults to ''.
        backup (bool, optional): If backup is true the original file is
            renamed to filename.bak before it is overwritten

    Examples:
        >>> with open('path/to/file', 'r') as f:
        ...     lines = f.readlines()
        >>> print(lines)
        This is a Test file.
        >>> search_and_replace('path/to/file', 'Test', 'new Test')
        >>> with open('path/to/file', 'r') as f:
        ...     lines = f.readlines()
        >>> print(lines)
        This is a new Test file.

    """
    with open(file_path, "r") as f:
        file_data = f.read()

    file_data = file_data.replace(search_pattern, replacement)

    if out_path is not None:
        out_path = file_path
        if backup:
            os.rename(file_path, file_path + ".bak")

    with open(out_path, "w") as file:
        file.write(file_data)


################################################################################
# Classes
################################################################################


class ParametersFramework:
    """Class to work with Parameters in the form of dict or attributes.

    Parameters can be set via keyword args in init, set as
    instance attributes or read from disk. Can write parameters
    to disk in .yaml or .json format.

    Attributes:
        main_path (str): The main path of the parameter class.
        defaults (dict): The defaults passed into the Parent Class by the child classes
            Parameters() and ACDParameters()

    Methods:
        save ():


    """

    n_neurons: list[int]
    activation_functions: list[str]

    def __init__(self, defaults: ParametersDict, **kwargs: ParametersData) -> None:
        """Instantiate the ParametersFramework class.

        This class is not meant to be used alone, but as a parent class for
        different parameters.

        Args:
            defaults (dict): A dictionary of default values.
            **kwargs: Arbitrary keyword arguments. If these arguments are not
                keys of the `defaults` dictionary, they will be ignored.
                Otherwise, they will overwrite the keys in the defaults dict.


        """
        self.main_path = os.getcwd()
        self.defaults = defaults

        # overwrite class defaults with user input **kwargs
        self._setattr(self.defaults)
        for key, value in kwargs.items():
            if key not in self.__dict__.keys():
                if key == "n_epochs":
                    print(
                        "Parameter `n_epochs` and `n_steps_per_epoch` is deprecated. Use `n_steps` instead."
                    )
                print(f"Dropping unknown dict entry for {{'{key}': {value}}}")
            else:
                setattr(self, key, value)
        if len(self.n_neurons) != len(self.activation_functions) - 1:
            raise Exception(
                f"Length of `n_neurons` and `activation_functions` (-1) does not match: {self.n_neurons}, {self.activation_functions}"
            )

    def save(self, path: Optional[str] = None) -> str:
        """Save parameters in json format or yaml format.

        Args:
            path (str, optional): Path where parameters should be saved. Possible extensions are '.json' and '.yaml'.
                If no path is given main_path/parameters.json is used. Defaults to ''.

        Returns:
            str: The path where the parameters were saved.

        """
        if path is None:
            path = os.path.join(self.main_path, f"parameters.json")
            fmt = "json"
        if os.path.isfile(path):
            filename, extension = os.path.splitext(path)
            time = _datetime_windows_and_linux_compatible()
            os.rename(path, filename + "_back_" + time + extension)
        fmt = path.split(".")[-1]
        if fmt not in ["json", "yaml"]:
            raise OSError(
                f"Unrecognized extension .{fmt}. Please provide either '.json' or '.yaml'"
            )
        with open(path, "w") as f:
            if fmt == "json":
                json.dump(self.__dict__, f, indent=4, sort_keys=True)
            else:
                yaml.dump(self.__dict__, f, default_flow_style=True)
        return path

    @property
    def parameters(self) -> str:
        """str: A string that contains tabulated parameter values."""
        doc_p = Parameters.__doc__.split("Attributes:")[1].split("Examples:")[0]
        doc_p = (
            "\n".join(map(lambda x: x.lstrip("        "), doc_p.splitlines()))
            .lstrip("\n")
            .rstrip("\n\n")
            .splitlines()
        )
        doc = ADCParameters.__doc__.split("Attributes:")[1].split("Examples:")[0]
        doc = (
            "\n".join(map(lambda x: x.lstrip("        "), doc.splitlines()))
            .lstrip("\n")
            .rstrip("\n\n")
            .splitlines()
        )
        doc = doc_p + doc
        descr_dict = {}
        key = doc[0].split("):")[0].split()[0]
        descr = doc[0].split("): ")[-1]
        for line in doc[1:]:
            if "):" not in line:
                descr = descr + " " + line
            else:
                descr_dict[key] = descr
                key = line.split("):")[0].split()[0]
                descr = line.split("): ")[-1]
        else:
            descr_dict[key] = descr
        out = []
        for key, value in self.__dict__.items():
            if key in self.defaults:
                out.append(
                    {
                        "Parameter": key,
                        "Value": value,
                        "Description": "\n".join(wrap(descr_dict[key], width=50)),
                    }
                )
        return printTable(out, sep="\n")

    @classmethod
    def from_file(cls, path: str) -> Union[ParametersType, ADCParametersType]:
        """Alternative constructor for ParameterFramework classes.

        Reads a file and sets the attributes based on that.

        Args:
            path (str): Path to the parameters.json or parameters.yaml file

        Returns:
            ParametersFramework: A new ParametersFramework class.

        """
        with open(path, "r") as f:
            if path.split(".")[-1] == "json":
                params = json.load(f)
            elif path.split(".")[-1] == "yaml":
                params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise ValueError(
                    f"The extension of the provided file should be `.json`, or `.yaml`. You provided {path.split('.')[1]}"
                )

        if "n_epochs" in params:
            print(
                "Detected old definition `n_epochs` and `n_steps_per_epoch`. I will change that to `n_steps` = `n_epochs` * `n_steps_per_epoch`."
            )
            params["n_steps"] = params["n_epochs"] * params["n_steps_per_epoch"]

            # also check soft start
            if "cartesian_cost_scale_soft_start" in params:
                if params["cartesian_cost_scale_soft_start"] != (None, None) and params[
                    "cartesian_cost_scale_soft_start"
                ] != [None, None]:
                    a, b = params["cartesian_cost_scale_soft_start"]
                    a *= params["n_steps_per_epoch"]
                    b *= params["n_steps_per_epoch"]
                    params["cartesian_cost_scale_soft_start"] = (a, b)

            # fix summary step and checkpoint_step
            params["summary_step"] *= params["n_steps_per_epoch"]
            params["checkpoint_step"] *= params["n_steps_per_epoch"]

            del params["n_epochs"]
            del params["n_steps_per_epoch"]

        return cls(**params)

    @classmethod
    def load(cls, path: str) -> Union[ParametersType, ADCParametersType]:
        """Loads the parameters saved in a .json or .yaml file into a new Parameter object.

        Args:
            path (str): Path to the parameters.json or parameters.yaml file

        Returns:
            ParametersFramework: A new ParametersFramework class.

        """
        with open(path, "r") as f:
            if path.split(".")[1] == "json":
                params = json.load(f)
            elif path.split(".")[1] == "yaml":
                params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise ValueError(
                    f"The extension of the provided file should be `.json`, or `.yaml`. You provided {path.split('.')[1]}"
                )

        if "n_epochs" in params:
            print(
                "Detected old definition `n_epochs` and `n_steps_per_epoch`. I will change that to `n_steps` = `n_epochs` * `n_steps_per_epoch`."
            )
            params["n_steps"] = params["n_epochs"] * params["n_steps_per_epoch"]
            del params["n_epochs"]
            del params["n_steps_per_epoch"]

        # check whether the parameters file has been moved and update it accordingly.
        if params["main_path"] != os.path.dirname(path):
            print(
                "seems like the parameter file was moved to another directory. Parameter file is updated ..."
            )
            search_and_replace(path, params["main_path"], os.path.dirname(path))
            with open(path, "r") as file:
                if path.split(".")[1] == "json":
                    params = json.load(f)
                elif path.split(".")[1] == "yaml":
                    params = yaml.load(f, Loader=yaml.FullLoader)
                else:
                    raise ValueError(
                        f"The extension of the provided file should be `.json`, or `.yaml`. You provided {path.split('.')[1]}"
                    )

        return cls(**params)

    def update(self, **kwargs: ParametersData) -> None:
        """Updates the values of `self`.

        Args:
            **kwargs: Arbitrary keyword arguments. If these arguments are not
                keys of the `self.defaults` dictionary, they will be ignored.
                Otherwise, they will overwrite the keys in the defaults dict.

        """
        for key, value in kwargs.items():
            if key not in self.__dict__.keys():
                print(f"Dropping unknown dict entry for {{'{key}': {value}}}")
            else:
                setattr(self, key, value)

    def _setattr(self, dictionary: ParametersDict) -> None:
        """Updates the values of `self.`

        Args:
            dictionary (dict):

        """
        if "cartesian_cost_scale_soft_start" in dictionary:
            if dictionary["cartesian_cost_scale_soft_start"] is not None or dictionary[
                "cartesian_cost_scale_soft_start"
            ] != (None, None):
                if len(dictionary["cartesian_cost_scale_soft_start"]) != 2:
                    raise Exception(
                        "Parameter cartesian_cost_scale_soft_start only takes a tuple of len 2."
                    )
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __setitiem__(self, key: str, value: ParametersData) -> None:
        """Implements the setitem method. Values can be set like so:

        Examples:
            >>> from encodermap import Parameters
            >>> p = Parameters()
            >>> p["center_cost_scale"] = 2.5
            >>> p["center_cost_scale"]
            2.5

        """
        if key == "cartesian_cost_scale_soft_start":
            if value is not None or value != (None, None):
                if len(value) != 2:
                    raise Exception(
                        "Parameter cartesian_cost_scale_soft_start only takes a tuple of len 2."
                    )
        setattr(self, key, value)

    def __getitem__(self, item: str) -> ParametersData:
        """Implements the getitem method. Get items with instance[key]."""
        return getattr(self, item)

    def _string_summary(self) -> str:
        """Creates a short summary of a parameter class. Additionally, adds info about non-standard values."""
        check_defaults = Parameters.defaults
        if self.__class__.__name__ == "ADCParameters":
            check_defaults.update(ADCParameters.defaults)
        diff_keys = list(
            filter(
                lambda x: not self.__dict__[x] == check_defaults[x],
                check_defaults.keys(),
            )
        )
        s = f"{self.__class__.__name__} class with Main path at {self.main_path}."
        for d in diff_keys:
            s += f"\nNon-standard value of {d}: {self.__dict__[d]}"
        if diff_keys == []:
            s += " All parameters are set to default values."
        return s

    def __str__(self) -> str:
        return self._string_summary()

    def __repr__(self) -> str:
        return f"<{self._string_summary()} Object at 0x{id(self):02x}>"


class Parameters(ParametersFramework):
    """Class to hold Parameters for the Autoencoder

    Parameters can be set via keyword args while instantiating the class, set as
    instance attributes or read from disk. This class can write parameters
    to disk in .yaml or .json format.

    Attributes:
        defaults (dict): Classvariable dict that holds the defaults
            even when the current values might have changed.
        main_path (str): Defines a main path where the parameters and other things might be stored.
        n_neurons (list of int): List containing number of neurons for each layer up to the bottleneck layer.
            For example [128, 128, 2] stands for an autoencoder with the following architecture
            {i, 128, 128, 2, 128, 128, i} where i is the number of dimensions of the input data.
            These are Input/Output Layers that are not trained.
        activation_functions (list of str): List of activation function names as implemented in TensorFlow.
            For example: "relu", "tanh", "sigmoid" or "" to use no activation function.
            The encoder part of the network takes the activation functions
            from the list starting with the second element. The decoder part of
            the network takes the activation functions in reversed order starting with
            the second element form the back. For example ["", "relu", "tanh", ""] would
            result in a autoencoder with {"relu", "tanh", "", "tanh", "relu", ""} as
            sequence of activation functions.
        periodicity (float): Defines the distance between periodic walls for the inputs.
            For example 2pi for angular values in radians.
            All periodic data processed by EncoderMap must be wrapped to one periodic window.
            E.g. data with 2pi periodicity may contain values from -pi to pi or from 0 to 2pi.
            Set the periodicity to float("inf") for non-periodic inputs.
        learning_rate (float): Learning rate used by the optimizer.
        n_steps (int): Number of training steps.
        batch_size (int): Number of training points used in each training step
        summary_step (int): A summary for TensorBoard is writen every summary_step steps.
        checkpoint_step (int): A checkpoint is writen every checkpoint_step steps.
        dist_sig_parameters (tuple of floats): Parameters for the sigmoid
            functions applied to the high- and low-dimensional distances
            in the following order (sig_h, a_h, b_h, sig_l, a_l, b_l)
        distance_cost_scale (int): Adjusts how much the distance based metric is weighted in the cost function.
        auto_cost_scale (int): Adjusts how much the autoencoding cost is weighted in the cost function.
        auto_cost_variant (str): defines how the auto cost is calculated. Must be one of:
            * `mean_square`
            * `mean_abs`
            * `mean_norm`
        center_cost_scale (float): Adjusts how much the centering cost is weighted in the cost function.
        l2_reg_constant (float): Adjusts how much the L2 regularisation is weighted in the cost function.
        gpu_memory_fraction (float): Specifies the fraction of gpu memory blocked.
            If set to 0, memory is allocated as needed.
        analysis_path (str): A path that can be used to store analysis
        id (str): Can be any name for the run. Might be useful for example for
            specific analysis for different data sets.
        model_api (str): A string defining the API to be used to build the keras model.
            Defaults to `sequntial`. Possible strings are:
            * `functional` will use keras' functional API.
            * `sequential` will define a keras Model, containing two other models with the Sequential API.
                These two models are encoder and decoder.
            * `custom` will create a custom Model where even the layers are custom.
        loss (str): A string defining the loss function.
                Defaults to `emap_cost`. Possible losses are:
                * `reconstruction_loss` will try to train output == input
                * `mse`: Returns a mean squared error loss.
                * `emap_cost` is the EncoderMap loss function. Depending on the class `Autoencoder`,
                    `Encodermap, `ACDAutoencoder`, different contributions are used for a combined loss.
                    Autoencoder uses atuo_cost, reg_cost, center_cost.
                    EncoderMap class adds sigmoid_loss.
        batched (bool): Whether the dataset is batched or not.
        training (str): A string defining what kind of training is performed when autoencoder.train() is callsed.
            * `auto` does a regular model.compile() and model.fit() procedure.
            * `custom` uses gradient tape and calculates losses and gradients manually.
        tensorboard (bool): Whether to print tensorboard information. Defaults to False.
        seed (Union[int, None]): Fixes the state of all operations using random numbers. Defaults to None.

    Examples:
        >>> import encodermap as em
        >>> paramters = em.Parameters()
        >>> parameters.auto_cost_variant
        mean_abs
        >>> parameters.save(path='/path/to/dir')
        /path/to/dir/parameters.json
        >>> # alternative constructor
        >>> new_params = em.Parameters.from_file('/path/to/dir/parameters.json')
        >>> new_params.main_path
        /path/to/dir/parameters.json

    """

    defaults = dict(
        n_neurons=[128, 128, 2],
        activation_functions=["", "tanh", "tanh", ""],
        periodicity=2 * pi,
        learning_rate=0.001,
        n_steps=100000,
        batch_size=256,
        summary_step=10,
        checkpoint_step=5000,
        dist_sig_parameters=(4.5, 12, 6, 1, 2, 6),
        distance_cost_scale=500,
        auto_cost_scale=1,
        auto_cost_variant="mean_abs",
        center_cost_scale=0.0001,
        l2_reg_constant=0.001,
        gpu_memory_fraction=0,
        analysis_path="",
        id="",
        model_api="sequential",
        loss="emap_cost",
        training="auto",
        batched=True,
        tensorboard=False,
        seed=None,
    )

    def __init__(self, **kwargs: ParametersData) -> None:
        """Instantiate the Parameters class

        Takes a dict as input and overwrites the class defaults. The dict is directly
        stored as an attribute and can be accessed via instance attributes.

        Args:
            **kwargs (dcit): Dict containing values. If unknown keys are passed they will be dropped.

        """
        # set class variable defaults to be instance variable
        if "defaults" in kwargs:
            kwargs.pop("defaults", None)
        super().__init__(self.defaults, **kwargs)

    @classmethod
    def defaults_description(cls) -> str:
        """str: A string that contains tabulated default parameter values."""
        doc = cls.__doc__.split("Attributes:")[1].split("Examples:")[0]
        doc = (
            "\n".join(map(lambda x: x.lstrip("        "), doc.splitlines()))
            .lstrip("\n")
            .rstrip("\n\n")
            .splitlines()
        )
        descr_dict = {}
        key = doc[0].split("):")[0].split()[0]
        descr = doc[0].split("): ")[-1]
        for line in doc[1:]:
            if "):" not in line:
                descr = descr + " " + line
            else:
                descr_dict[key] = descr
                key = line.split("):")[0].split()[0]
                descr = line.split("): ")[-1]
        else:
            descr_dict[key] = descr

        out = []
        for key, value in cls.defaults.items():
            out.append(
                {
                    "Parameter": key,
                    "Default Value": value,
                    "Description": "\n".join(wrap(descr_dict[key], width=50)),
                }
            )
        return printTable(out, sep="\n")


class ADCParameters(ParametersFramework):
    """This is the parameter object for the AngleDihedralCartesianEncoder.
    It holds all the parameters that the Parameters object includes, plus the following attributes:

    Attributes:
        cartesian_pwd_start (int): Index of the first atom to use for the pairwise distance calculation.
        cartesian_pwd_stop (int): Index of the last atom to use for the pairwise distance calculation.
        cartesian_pwd_step (int):  Step for the calculation of paiwise distances. E.g. for a chain of atoms
            N-C_a-C-N-C_a-C... cartesian_pwd_start=1 and cartesian_pwd_step=3 will result in using all C-alpha atoms for the
            pairwise distance calculation.
        use_backbone_angles (bool): Allows to define whether backbone bond angles should be learned (True) or if instead mean
            values should be used to generate conformations (False).
        use_sidechains (bool): Whether sidechain dihedrals should be passed through the autoencoder.
        angle_cost_scale (int): Adjusts how much the angle cost is weighted in the cost function.
        angle_cost_variant (str): Defines how the angle cost is calculated. Must be one of:
            * "mean_square"
            * "mean_abs"
            * "mean_norm".
        angle_cost_reference (int): Can be used to normalize the angle cost with the cost of same reference model (dummy).
        dihedral_cost_scale (int): Adjusts how much the dihedral cost is weighted in the cost function.
        dihedral_cost_variant (str): Defines how the dihedral cost is calculated. Must be one of:
            * "mean_square"
            * "mean_abs"
            * "mean_norm".
        dihedral_cost_reference (int): Can be used to normalize the dihedral cost with the cost of same reference model (dummy).
        side_dihedral_cost_scale (int): Adjusts how much the side dihedral cost is weighted in the cost function.
        side_dihedral_cost_variant (str): Defines how the side dihedral cost is calculated. Must be one of:
            * "mean_square"
            * "mean_abs"
            * "mean_norm".
        side_dihedral_cost_reference (int): Can be used to normalize the side dihedral cost with the cost of same reference model (dummy).
        cartesian_cost_scale (int): Adjusts how much the cartesian cost is weighted in the cost function.
        cartesian_cost_scale_soft_start (tuple): Allows to slowly turn on the cartesian cost. Must be a tuple with
            (start, end) or (None, None) If begin and end are given, cartesian_cost_scale will be increased linearly in the
            given range.
        cartesian_cost_variant (str): Defines how the cartesian cost is calculated. Must be one of:
            * "mean_square"
            * "mean_abs"
            * "mean_norm".
        cartesian_cost_reference (int): Can be used to normalize the cartesian cost with the cost of same reference model (dummy).
        cartesian_dist_sig_parameters (tuple of floats): Parameters for the sigmoid functions applied to the high- and low-dimensional
            distances in the following order (sig_h, a_h, b_h, sig_l, a_l, b_l).
        cartesian_distance_cost_scale (int): Adjusts how much the cartesian distance cost is weighted in the cost function.

    Examples:
        >>> import encodermap as em
        >>> parameters = em.ADCParameters()
        >>> parameters.auto_cost_variant
        mean_abs
        >>> parameters.save(path='/path/to/dir')
        /path/to/dir/parameters.json
        >>> # alternative constructor
        >>> new_params = em.Parameters.from_file('/path/to/dir/parameters.json')
        >>> new_params.main_path
        /path/to/dir/parameters.json

    """

    defaults = dict(
        Parameters.defaults,
        **dict(
            model_api="functional",  # overwrite main class. Functional allows multiple in and outputs.
            cartesian_pwd_start=None,
            cartesian_pwd_stop=None,
            cartesian_pwd_step=None,
            use_backbone_angles=False,
            use_sidechains=False,
            angle_cost_scale=0,
            angle_cost_variant="mean_abs",
            angle_cost_reference=1,
            dihedral_cost_scale=1,
            dihedral_cost_variant="mean_abs",
            dihedral_cost_reference=1,
            side_dihedral_cost_scale=0.5,
            side_dihedral_cost_variant="mean_abs",
            side_dihedral_cost_reference=1,
            cartesian_cost_scale=1,
            cartesian_cost_scale_soft_start=(None, None),  # begin, end
            cartesian_cost_variant="mean_abs",
            cartesian_cost_reference=1,
            cartesian_dist_sig_parameters=Parameters.defaults["dist_sig_parameters"],
            cartesian_distance_cost_scale=1,
            auto_cost_scale=None,
            distance_cost_scale=None,
        ),
    )

    def __init__(self, **kwargs: ParametersData) -> None:
        """Instantiate the ADCParameters class

        Takes a dict as input and overwrites the class defaults. The dict is directly
        stored as an attribute and can be accessed via instance attributes.

        Args:
            **kwargs (dict): Dict containing values. If unknown values are passed they will be dropped.

        """
        if "cartesian_cost_scale_soft_start" in kwargs:
            if kwargs["cartesian_cost_scale_soft_start"] is not None or kwargs[
                "cartesian_cost_scale_soft_start"
            ] != (None, None):
                if len(kwargs["cartesian_cost_scale_soft_start"]) != 2:
                    raise Exception(
                        "Parameter cartesian_cost_scale_soft_start only takes a tuple of len 2."
                    )
        # set class variable defaults to be instance variable
        if "defaults" in kwargs:
            kwargs.pop("defaults", None)
        super().__init__(self.defaults, **kwargs)

    @classmethod
    def defaults_description(cls) -> str:
        """str: A string that contains tabulated default parameter values."""
        doc_p = Parameters.__doc__.split("Attributes:")[1].split("Examples:")[0]
        doc_p = (
            "\n".join(map(lambda x: x.lstrip("        "), doc_p.splitlines()))
            .lstrip("\n")
            .rstrip("\n\n")
            .splitlines()
        )
        doc = cls.__doc__.split("Attributes:")[1].split("Examples:")[0]
        doc = (
            "\n".join(map(lambda x: x.lstrip("        "), doc.splitlines()))
            .lstrip("\n")
            .rstrip("\n\n")
            .splitlines()
        )
        doc = doc_p + doc
        descr_dict = {}
        key = doc[0].split("):")[0].split()[0]
        descr = doc[0].split("): ")[-1]
        for line in doc:
            if "):" not in line:
                descr = descr + " " + line
            else:
                descr_dict[key] = descr
                key = line.split("):")[0].split()[0]
                descr = line.split("): ")[-1]
        else:
            descr_dict[key] = descr

        out = []
        for key, value in cls.defaults.items():
            out.append(
                {
                    "Parameter": key,
                    "Default Value": value,
                    "Description": "\n".join(wrap(descr_dict[key], width=50)),
                }
            )
        return printTable(out, sep="\n")
