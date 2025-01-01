# -*- coding: utf-8 -*-
# tests/conftest.py
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
# Standard Library Imports
import inspect
import os
import unittest
from functools import wraps
from pathlib import Path


def sort_tests(x):
    if x.name == "test_losses":
        return 1
    if x.name == "test_losses_periodic":
        return 2
    return 3


def pytest_collection_modifyitems(session, config, items):
    # Ignores tensorflow.test.TestCase.test_session methods
    # They will be marked as skipped tests without this configuration.
    items[:] = [item for item in items if item.name != "test_session"]

    # For some reason, the tests in test_losses.py succeed if tested on their own,
    # but fail, when pytest is called on complete directory. This is used to order
    # them correctly.
    items[:] = list(sorted(items, key=sort_tests))


def expensive_test(func_or_class):
    """Marks a class or test_method as expensive.

    Expensive tests are only run, when the environment variable 'RUN_EXPENSIVE_TESTS'
    is set to 'True'. Otherwise, they are skipped.

    """
    if type(func_or_class) is type:
        for attr in func_or_class.__dict__.keys():
            if (
                callable(getattr(func_or_class, attr))
                and attr.startswith("test_")
                and attr != "test_session"
            ):
                setattr(
                    func_or_class, attr, expensive_test(getattr(func_or_class, attr))
                )
        return func_or_class
    else:

        @wraps(func_or_class)
        def wrapper(*args, **kwargs):
            self = args[0]
            if os.getenv("RUN_EXPENSIVE_TESTS", "False").lower() == "true":
                print(
                    f"I will now run the expensive test " f"`{func_or_class.__name__}`."
                )
                return func_or_class(*args, **kwargs)
            else:
                self.skipTest(
                    "I will not run the expensive test "
                    f"`{func_or_class.__name__}`. Set the "
                    "environment variable `RUN_EXPENSIVE_TESTS=True` to run this test."
                )

        return wrapper


def skip_all_tests_except_env_var_specified(decorator):
    """Skips all tests, except the one, that is set in the environment variable
    'ENCODERMAP_SKIP_TESTS_EXCEPT'.

    """

    def decorate(cls):
        for attr in cls.__dict__.keys():
            if (
                callable(getattr(cls, attr))
                and attr.startswith("test_")
                and attr != "test_session"
            ):
                env_tests = os.getenv("ENCODERMAP_SKIP_TESTS_EXCEPT", "false")
                if env_tests == "false":
                    break
                env_tests = env_tests.split(" ")
                this_test_name = (
                    f"{Path(inspect.getfile(cls)).stem}.{cls.__name__}.{attr}"
                )
                test_parts = this_test_name.split(".")
                for i in range(len(test_parts), 0, -1):
                    name = ".".join(test_parts[:-i])
                    if name in env_tests:
                        break
                else:
                    if this_test_name not in env_tests:
                        setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate
