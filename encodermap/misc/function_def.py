# -*- coding: utf-8 -*-
# encodermap/misc/function_def.py
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

# Third Party Imports
import tensorflow as tf


def function(debug=False):
    def decorator(f):
        def wrapper(*args, **kwargs):
            if debug:
                result = f(*args, **kwargs)
            else:
                compiled = tf.function(f)
                result = compiled(*args, **kwargs)
            return result

        return wrapper

    return decorator
