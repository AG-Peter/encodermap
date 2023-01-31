# -*- coding: utf-8 -*-
# encodermap/loading/pipeline.py
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
"""

ToDo:
    * Make a pyemma pipeline
    * Load feayures save as xarray in HDF5 and close file
    * This way everything will be easy on memory
    * Fix this TypeError, when inheriting from the same class twice. Make an option to change the metaclass of the second one.
    * TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
"""

import os

from .._optional_imports import _optional_import
from ..misc.errors import BadError

##############################################################################
# Optional Imports
##############################################################################


SerializableMixIn = _optional_import(
    "pyemma", "_base.serialization.serialization.SerializableMixIn"
)
StreamingEstimationTransformer = _optional_import(
    "pyemma", "coordinates.data._base.transformer.StreamingEstimationTransformer"
)

# class PipelineWriter(StreamingEstimationTransformer, SerializableMixIn):
#     """Uses pyemma pipelines to read CVs from files and write to disk without loading everything into memory."""
#     __serialize_version = 0
#     def __init__(self, directory):
#         """Instantiates the class."""
#         self.dir = directory
