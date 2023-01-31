# -*- coding: utf-8 -*-
# encodermap/_typing.py
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
"""Typing for the encodermap package"""


import typing
from typing import Union

if typing.TYPE_CHECKING:
    from .parameters import ADCParameters, Parameters

    AnyParameters = Union[Parameters, ADCParameters]

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

    from .autoencoder.autoencoder import (
        AngleDihedralCartesianEncoderMap,
        Autoencoder,
        EncoderMap,
    )

    AutoencoderClass = Union[Autoencoder, EnocderMap, AngleDihedralCartesianEncoderMap]
