# -*- coding: utf-8 -*-
# encodermap/misc/errors.py
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

__all__ = ["BadError"]

##############################################################################
# Errors
##############################################################################


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class BadError(Error):
    """Raised when the Error is really bad."""

    def __init__(self, message):
        self.message = "VERY BAD ERROR: " + message
        super().__init__(self.message)


class MixedUpInputs(Exception):
    pass


class NotImplementedError(Error):
    """Raised when I know, that there is a section missing,
    but I am too lazy to code that in."""

    pass
