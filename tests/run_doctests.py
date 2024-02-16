#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/run_doctests.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
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


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import doctest
import pkgutil
import unittest
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Optional, Union

# Third Party Imports
from click import command, option


################################################################################
# Fix doctests atrocities
################################################################################

OC = doctest.OutputChecker


class AEOutputChecker(OC):
    def check_output(self, want, got, optionflags):
        # Standard Library Imports
        from re import sub

        if optionflags & doctest.ELLIPSIS:
            want = sub(r"\[\.\.\.\]", "...", want)
        return OC.check_output(self, want, got, optionflags)


doctest.OutputChecker = AEOutputChecker


################################################################################
# Utils
################################################################################


def yield_modules(
    package: Union[str, ModuleType],
    pkg: Optional[str] = None,
) -> Iterable[str]:
    if pkg is None:
        pkg = deepcopy(package)
    if not isinstance(package, str):
        package = str(Path(package.module_finder.find_module(package.name).path).parent)
    for module in pkgutil.iter_modules([package]):
        name = f"{pkg}.{module.name}"
        if not module.ispkg:
            yield name
        else:
            yield name
            yield from yield_modules(module, name)


################################################################################
# Main
################################################################################


@command()
@option(
    "--module",
    default="encodermap",
    help=("The name of the module to run doctests from."),
)
@option(
    "--exclude",
    default="tf1",
    help=(f"Exclude this substring from recursing through modules."),
)
@option("--only", default=None, help=(f"Only run doctests matching this string."))
def main(
    module: str = "encodermap",
    exclude: str = "tf1",
    only: Optional[str] = None,
) -> None:
    suite = unittest.TestSuite()
    runner = unittest.TextTestRunner()
    tests = []
    if only is None:
        tests.append(doctest.DocTestSuite(module))
        for subpackage in yield_modules(module):
            if not exclude in subpackage:
                tests.append(doctest.DocTestSuite(subpackage))
    else:
        tests.append(doctest.DocTestSuite(only))
    suite.addTests(tests)
    runner.run(suite)


if __name__ == "__main__":
    main()
