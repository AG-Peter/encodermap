# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build.
2. Update the README.md with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you
   do not have permission to do that, you may request the second reviewer to merge it for you.

## Code formatting

Use black and isort. Use google-style docstrings. Use vulture and pycodestyle.
Use section headers for files. A `*.py` file in EncoderMap should look like this:

```python
# -*- coding: utf-8 -*-
# encodermap/subpackage/filename.py
<<<LEGAL DISCLAIMER. COPY IT FORM OTHER FILES
AND ADD YOURSELF AS AN AUTHOR>>>
"""Summary of this modules functionality.

This module-level docstring will appear as the topmost documentation
of this module. It should introduce concepts and display how this
module ties in with the EncoderMap package. It can also contain examples

Example:
    >>> # This is example python code
    >>> # Examples like this will be picked up by EncoderMap's tests
    >>> # Lines that produce outputs should not start with the
    >>> # python prompt (>>>), but contain the output. Doctests
    >>> # will automatically run on them. Let's create a sum function
    >>> # and test it
    >>>
    >>> from __future__ import annotations
    >>>
    >>>
    >>> def my_sum(a: float, b: float) -> float:
    ...     return(a + b)
    >>> my_sum(1, 2)  # in-line comments have two spaces before the #-sign
    3
    >>> # the line above is the output, that doctests checks.

In the module-wide docstring variables, you can also use some reStructuredText
markup, which is initialized with two colons::

    This text will be in a monospace-styled font.

And restructured text links are added like so:

.. _Click Me:
   https://github.com/AG-Peter/encodermap

As a general rule, we use two spaces between elements. Elements are:
    * Module-level docstrings
    * import sections
    * global variables
    * headers
    * functions
    * classes

Docstrings should always close with an empty line.

"""


################################################################################
# Import
################################################################################


# Future Imports at the top
from __future__ import annotations


# Standard Library Imports
import os
import re

# Third Party Imports
import numpy as np


# Local Folder Imports
from ..misc.misc import _datetime_windows_and_linux_compatible


################################################################################
# Typing
################################################################################


# Standard Library Imports
from numbers import Number
from typing import Union, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from encodermap._typing import AnyParameters
    # import more expensive libraries here


################################################################################
# Globals
################################################################################


GLOBAL_VARIABLE: dict[str, str] = {"maybe": "some data"}
"""dict[str, str]: Module level variables are documented inline.


The docstring may span multiple lines. The type may optionally be specified
on the first line, separated by a colon.
"""


__all__: list[str] = ["public_function", "PublicClass"]
"""list[str]: The names, imported when * is imported."""


################################################################################
# Utils
################################################################################


def _private_function(*args: Union[float, str]) -> bool:
    """A private function. Use type hints and google-style docstrings.

    Since PEP 484, we don't need to specify the types of the arguments
    in the 'Args' section.

    .. _PEP 484:
       https://www.python.org/dev/peps/pep-0484/

    Args:
        *args: Any number of arguments can be float or str type.

    Returns:
        bool: True, when the number of args is even, False otherwise.

    """
    for i, a in enumerate(args):
       print(f"Argument {i} is {a=}")
    if len(args) % 2 == 0:
       return True
    return False


################################################################################
# Public
################################################################################


def public_function(*args: Number) -> Number:
    """A concise summary: This function returns the sum of *args.

    Public functions need to be well documented to make them usable.
    Maybe with some explanation and examples. This function adds
    all positional arguments. Because there are more number types than
    just int and float in python, this function is annotated with
    the `Numbers` abstract base class from the built-in numbers module.

    Python's Number Types:
        * int: Signed 64-bit integers. Unlimited precision.
        * float: 64-bit floating point number analogous to C's double.
        * complex: Two floats representing a complex.imag and complex.real
        * fractions.Fraction: Special type to deal with rational numbers.
            Can be a great tool to overcome the shortcomings of
            floating-point arithmetic as they have infinite precision.
        * decimal.Decimal: Similar to fractions. If you're working with
            financial data, use Decimal.

    Examples:
        >>> public_function(1, 2)
        3

    Args:
        *args (Number): Variable number of numbers.

    Returns:
        Number: The sum of *args.

    """
    return sum(args)


class PublicClass:
    """A concise docstring explaining this class.

    Below you can give more info about the class. Maybe list
    parent classes or important attributes:

    Attrs:
        name (str): The name of this class. Default is 'PublicClass'

    Examples:
        >>> c = PublicClass()
        >>> c.name
        PublicClass
        >>> c = PublicClass(name="MyClass")
        >>> c.name
        MyClass

    """
    def __init__(self, name: Optional[str] = None) -> None:
        """Instantiate the class.

        The class constructor should be documented in the class'
        __init__ function (https://peps.python.org/pep-0257/).

        Args:
            name (Optional[str]): A str representing the name of
                this class. If None is provided, the default is
                'PublicClass'.

        """
        self.name = name
        if self.name is None:
           self.name = "PublicClass"

```

## Jupyter Notebooks

Jupyter Notebooks are a great tool to help users with their first steps with EncoderMap. Notebooks are also included in the documentation using the `nbsphinx` tool. Writing EncoderMap notebooks should follow these design rules:

- One H1 heading. All notebooks should only have a single H1 heading at the start. That way they better integrate into the sphinx documentation.

- Include a primer. About what will be taught in this notebook. This can be aided with a clickable toc, which can be implemented as such:

  - Add these html at the start of the cell to be linked. Exactly like in this example. Replace spaces with hyphens. Make sure to use the same name in the id as in the section name (that way, the TOC becomes clickable in Sphinx docs).

    ```html
    <a id='section-name-with-hyphens'></a>

    ## Section Name with-hyphens
    ```


  - Add a link with `[This is the section summary](#section-name-with-hyphens')` in any markdown cell.

- Include a link to the notebook on Google Colab:

  ```markdown
  # Getting started: Basic Cube

  **Welcome**

  Welcome to your first EncoderMap tutorial. All EncoderMap tutorials are provided as jupyter notebooks, that you can run locally, on binderhub, or even on google colab.

  Run this notebook on Google Colab:

  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AG-Peter/encodermap/blob/main/tutorials/notebooks_starter/01_Basic_Usage-Cube_Example.ipynb)

  Find the documentation of EncoderMap:

  https://ag-peter.github.io/encodermap

  **Goals:**

  In this tutorial you will learn:
  - [How to set training parameters for EncoderMap.](#select_parameters)
  - [How to train EncoderMap.](#train_encodermap)
  - [How to use the decoder part of the network to create high-dimensional data.](#generate_highd_data)
  ```

- Tag cells accordingly:

  - Skip execution but show in doc:

    ```json
    "metadata": {"emap": "skip"}
    ```

  - Execute but hide in doc:

    ```json
    "metadata": {"emap": "hide"}
    ```

  - Runs the cell, but clears the output afterward. Can be useful, when tqdm is broken again and prints all progress in separate lines.

    ```json
    "metadata": {"emap": "clear_output"}
    ```

  - Allow Error makes a cell run and ignore errors.

    ```json
    "metadata": {"emap": "allow_error"}
    ```

When hiding a cell, make sure to include some info, so that users are not surprised when the raw ipynb file contains more cells.

```python
# This is a hidden cell to allow sphinx to render this notebook faster,
# by loading pre-trained models.
# These cells often follow `emap.train()` cells, which are often skipped
# when building documentation
# You, the user can choose to either run the train cell or this hidden cell,
# the output of the notebook should not be affected by this.
lowd_data, e_map = em.load_project("cube")
```

## Type hints, MyPy and Beartype

EncoderMap tries to implement type hints where possible to make the code more reliable by using the static-type checker MyPy. Although python is a dynamically typed language declaring types can be beneficial for code robustness and documentation reasons.

## Tests

Find the documentation about tests in the [Test README](`EncoderMap tests`_).

## Building the documentation and Sphinx

The documentation is built using sphinx. Find more

## Vulture, flake8, isoprt and black

### Linting

Linting can help identify potential errors and so, EncoderMap is set up to use flake8 as its go-to linter. However, the maintainer of flake8 does not allow it to be configured via a `pyproject.toml`. To save us from adding more .cfg files to EncoderMap, we use pyproject-flake8, which after installed, can be run from EncoderMap's main directory with:

```bash
pflake8 .
```

### Vulture

Vulture can be used to find unused code. Use it by calling

```bash
vulture .
```

in EncoderMap's home main directory.

### Isort

Isort automatically sorts imports. It is a great tool to make your code look better.
Use it with

```bash
isort .
```

### Black

Black is a code-formatter, that automatically formats your code. Use it with

```bash
black .
```

## Automate using pre-commit-hooks

The file `.pre-commit-config.yaml` contains some automated pre-commit hooks, that can help you with linting and formatting. Install `pre-commit` with

```bash
$ pip install pre-commit
```

* Set up the pre-commit hooks with

```bash
$ pre-commit install
```

* Run the hooks without commit

```bash
$ pre-commit run --all-files
```

These hooks will also run, when you try to `git commit` your changes. Some hooks might also prevent a commit.

Current hooks are:

* trailing-whitespace: Trims trailing whitespace from files
* check-added-large-files: Prevents commit of large files
* end-of-file-fixer: Fixes files to UNIX style newlines (\n).
* check-yaml: Can fix broken yaml files.
* check-private-key: Although EncoderMap does not do networking, this hook prevents you from publishing your keys.
* isort: Sorts the import statements in .py files.
* black: Runs black using the configurations in `pyproject.toml`
* vulture: Runs vulture using the configurations in `pyproject.toml`.
* run-pycodestyle: Runs pycodestyle (formerly PEP8).
* clear-ipynb-cells. Clears the cells of ipynbs.

## NektOS act

## Adding yourself as an author
