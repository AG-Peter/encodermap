# -*- coding: utf-8 -*-
# encodermap/_typing.py
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
"""Typing for the encodermap package"""


# Standard Library Imports
import typing
from collections.abc import Sequence
from typing import Literal, Union

# Third Party Imports
import numpy as np


CanBeIndex = Union[int, Sequence[int], np.ndarray, slice]

CustomAAsDict = dict[
    Union[str, tuple[str, str]],
    tuple[
        str,
        dict[
            Literal[
                "bonds",
                "optional_bonds",
                "delete_bonds",
                "optional_delete_bonds",
                "PHI",
                "PSI",
                "OMEGA",
                "CHI1",
                "CHI2",
                "CHI3",
                "CHI4",
                "ChI5",
            ],
            Union[list[str], list[tuple[Union[str, int], Union[str, int]]]],
        ],
    ],
]


if typing.TYPE_CHECKING:
    # Local Folder Imports
    from .parameters import ADCParameters, Parameters

    AnyParameters = Union[Parameters, ADCParameters]

    # Local Folder Imports
    from .loading.features import (
        AllBondDistances,
        AllCartesians,
        CentralAngles,
        CentralBondDistances,
        CentralCartesians,
        CentralDihedrals,
        SideChainAngles,
        SideChainBondDistances,
        SideChainCartesians,
        SideChainDihedrals,
    )

    AnyFeature = Union[
        AllCartesians,
        AllBondDistances,
        CentralCartesians,
        CentralBondDistances,
        CentralAngles,
        CentralDihedrals,
        SideChainCartesians,
        SideChainBondDistances,
        SideChainAngles,
        SideChainDihedrals,
    ]

    # Local Folder Imports
    from .autoencoder.autoencoder import (
        AngleDihedralCartesianEncoderMap,
        Autoencoder,
        EncoderMap,
    )

    AutoencoderClass = Union[Autoencoder, EncoderMap, AngleDihedralCartesianEncoderMap]
