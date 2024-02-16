# -*- coding: utf-8 -*-
# tests/conftest.py
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
# Standard Library Imports
import os
from functools import wraps


def sort_tests(x):
    if x.name == "test_losses":
        print("test losses. Returning 1")
        return 1
    if x.name == "test_losses_periodic":
        return 2
    return 3


def pytest_collection_modifyitems(session, config, items):
    # Ignores tensorflow.test.TestCase.test_session methods
    # They will be marked as skipped tests wihtout this configuration.
    items[:] = [item for item in items if item.name != "test_session"]

    # For some reason the tests in test_losses.py succeed if tested on their own,
    # but fail, when pytest is called on complete directory. This is used to order
    # them correctly.
    items[:] = list(sorted(items, key=sort_tests))


def expensive_test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv("RUN_EXPENSIVE_TESTS", "False").lower() == "true":
            print(f"I will now run the expensive test " f"`{func.__name__}`.")
            return func(*args, **kwargs)
        else:
            print(
                "I will not run the expensive test "
                f"`{func.__name__}`. Set the "
                "environment variable `RUN_EXPENSIVE_TESTS=True` to run this test."
            )
            return

    return wrapper
