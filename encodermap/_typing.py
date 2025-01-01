# -*- coding: utf-8 -*-
# encodermap/_typing.py
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
"""Typing for the encodermap package"""


# Standard Library Imports
import typing
from collections.abc import Sequence
from typing import Literal, Union

# Third Party Imports
import numpy as np


################################################################################
# Type Defs
################################################################################


CanBeIndex = Union[int, Sequence[int], Sequence[np.ndarray], slice]


DihedralOrBondDict = dict[
    Literal[
        "bonds",
        "optional_bonds",
        "delete_bonds",
        "optional_delete_bonds",
        "PHI",
        "PSI",
        "OMEGA",
        "not_PHI",
        "not_PSI",
        "not_OMEGA",
        "CHI1",
        "CHI2",
        "CHI3",
        "CHI4",
        "CHI5",
    ],
    Union[list[str], list[tuple[Union[str, int], Union[str, int]]]],
]


CustomAAsDict = dict[
    Union[str, tuple[str, str]],
    Union[
        None,
        tuple[str, None],
        tuple[
            str,
            DihedralOrBondDict,
        ],
    ],
]
