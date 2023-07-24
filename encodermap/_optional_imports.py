# -*- coding: utf-8 -*-
# optional_imports.py

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

This module only contains one function. Look at the docstring of
`_optional_import` for more info.

"""
from __future__ import annotations

from typing import Any


def _optional_import(
    module: str,
    name: str = None,
    version: str = None,
    auto_install: bool = False,
    user_install: bool = False,
) -> Any:
    """Function that allows optional imports.

    This function can be provided with a str, denoting a module name (like 'numpy')
    and if numpy is available it will return the module `np = _optional_import('numpy')`.

    If the package is not availabe (`import some_package` would normally raise `ModuleNotFoundError`),
    a class is returned instead of the package. This class raises an Exception when it is called `()`,
    or an attribute of the class `getattr()` is accessed. This postpones the import error
    and allows to use optional packages for code libraries that would otherwise have
    long lists of dependencies.

    This also works with OOP class inheritance. So you can use _optional_import for
    constructing sub-classes of classes from other packages that might or
    might not be available.

    Args:
        module (str): The string of the module. The string is case sensitive. So for
            MDAnalysis, `module` should be 'MDAnalysis'. Another example is 'Biopython'.

    Keyword Args:
        name (str, optional): The name that is tried to be access from the module. If you
            want to use the function `random()` from numpy's random module, `name` should be
            'random.random'. This can also work for very long imports. For Dense layers
            from tensorflow import you would give `module` 'tensorflow' and `name`
            'keras.layers.Dense'. If None is provided this function will not
            be used and the base module will be returned. Defaults to None.
        version (str, optional): An optional version of the package. Will only be used, when
            the optional import fails and will inform the user, that a certain version needs
            to be installed. If None is provided, the version of the module
            is not included in the Exception. Defaults to None.
        auto_install (bool, optional): Whether to automatically install packages using subprocess and sys.
            Defaults to False.
        user_install (bool, optional): Prompts the user, whether they want to install the missing package.
            Defaults to False.

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
        >>> NonexistentClass = _optional_import('nonexistent_package', 'sub.module.NonexistentClass')
        >>> class MyClass(NonexistentClass):
        ...     def __init__(*args, **kwargs):
        ...         super().__init__()
        >>> try:
        ...     layer = MyClass(50)
        ... except ValueError as e:
        ...     print(e)
        Install the `nonexistent_package` package to make use of this feature.
        >>> pd = _optional_import('pandas', auto_install=True)
        >>> pd.isna('dog')
        False

    """
    import importlib

    _module = module
    try:
        module = importlib.import_module(module)
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
            module_name = "." + name.split(".")[-2]
            object_name = name.split(".")[-1]
            path = _module + "." + ".".join(name.split(".")[:-2])
            path = path.rstrip(".")
            module = importlib.import_module(module_name, path)
            return getattr(module, object_name)
        except Exception as e2:
            msg = f"Absolute and relative import of {name} from module {_module} failed with Exception {e2}. Either install the `{_module}` package or fix the optional_import."
        import_error = e

    # install packages
    if isinstance(import_error, Exception) and auto_install and not user_install:
        import subprocess
        import sys

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", _module])
            return _optional_import(_module, name, version)
        except subprocess.CalledProcessError as grepexc:
            msg = f"The `auto_install` option was set to True, but pip could not install the {_module} package and failed with ExitCode {grepexc}."

    # if user prompt
    if isinstance(import_error, Exception) and not auto_install and user_install:
        import subprocess
        import sys

        user_inp = input(
            f"Do you want to install the {_module} package to use this feature? (y/n)"
        )
        if user_inp == "y" or user_inp == "yes":
            subprocess.check_call([sys.executable, "-m", "pip", "install", _module])
            return _optional_import(_module, name, version)
        else:
            msg = f"User chose to not innstall the {_module} package"

    # failed import class closure
    class _failed_import:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise ValueError(msg) from import_error

        def __getattribute__(self, name):
            # if class is base class for some other class
            if name == "__mro_entries__":
                return object.__getattribute__(self, name)
            raise ValueError(msg) from import_error

        def __getattr__(self, name):
            # if class is base class for some other class
            if name == "__mro_entries__":
                return object.__getattribute__(self, name)
            raise ValueError(msg) from import_error

    return _failed_import()
