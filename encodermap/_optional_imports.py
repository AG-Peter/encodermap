# -*- coding: utf-8 -*-
# encodermap/_optional_imports.py

# Copyright (c) 2021, Kevin Sawade (kevin.sawade@uni-konstanz.de)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# Find the GNU Lesser General Public License under <http://www.gnu.org/licenses/>.
"""Optional imports of python packages.

Allows you to postpone import exceptions. Basically makes the codebase of EncoderMap
leaner, so that users don't need to install packages for features they don't require.

Examples:
    >>> from encodermap._optional_imports import _optional_import
    >>> np = _optional_import('numpy')
    >>> np.array([1, 2, 3])
    array([1, 2, 3])
    >>> nonexistent = _optional_import('nonexistent_package')
    >>> try:
    ...     nonexistent.function()
    ... except ValueError as e:
    ...     print(e)
    Install the `nonexistent_package` package to make use of this feature.
    >>> try:
    ...     _ = nonexistent.variable
    ... except ValueError as e:
    ...     print(e)
    Install the `nonexistent_package` package to make use of this feature.
    >>> numpy_random = _optional_import('numpy', 'random.random')
    >>> np.random.seed(1)
    >>> np.round(numpy_random((5, 5)) * 20, 0)
    array([[ 8., 14.,  0.,  6.,  3.],
           [ 2.,  4.,  7.,  8., 11.],
           [ 8., 14.,  4., 18.,  1.],
           [13.,  8., 11.,  3.,  4.],
           [16., 19.,  6., 14., 18.]])

"""

from __future__ import annotations

from typing import Any


def _optional_import(
    module: str,
    name: str = None,
    version: str = None,
) -> Any:
    import importlib

    _module: str = module
    try:
        # try the import
        module: Any = importlib.import_module(module)
        if name is None:
            return module
        if "." in name:
            for i in name.split("."):
                module = getattr(module, i)
            return module
        return getattr(module, name)
    except ImportError as e:
        # import failed
        if version is not None:
            msg = f"Install the `{_module}` package with version `{version}` to make use of this feature."
        else:
            msg = f"Install the `{_module}` package to make use of this feature."
        import_error = e
    except AttributeError as e:
        # absolute import failed. Try relative import
        try:
            try:
                module_name = "." + name.split(".")[-2]
            except AttributeError as ae:
                raise ae from e
            object_name = name.split(".")[-1]
            path = _module + "." + ".".join(name.split(".")[:-2])
            path = path.rstrip(".")
            module = importlib.import_module(module_name, path)
            return getattr(module, object_name)
        except Exception as e2:
            msg = (
                f"Absolute and relative import of {name} from module "
                f"{_module} failed with Exception {e2}. Either install the "
                f"`{_module}` package or fix the optional_import."
            )
            module_name = "." + name.split(".")[-2]
            object_name = name.split(".")[-1]
            path = _module + "." + ".".join(name.split(".")[:-2])
            msg = (
                f"I was given these attrs: {_module=}, {name=}. After a "
                f"failed absolute import, I tried to mimic a relative "
                f"import of the object {object_name=} from the module "
                f"{module_name=}. The path of the object was determined to "
                f"{path=}."
            )

        import_error = e

    class _failed_import:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise ValueError(msg) from import_error

        def __getattribute__(self, name):
            # if class is parent class for some other class
            if name == "__mro_entries__":
                return object.__getattribute__(self, name)
            raise ValueError(msg) from import_error

        def __getattr__(self, name):
            # if class is parent class for some other class
            if name == "__mro_entries__":
                return object.__getattribute__(self, name)
            raise ValueError(msg) from import_error

    return _failed_import()
