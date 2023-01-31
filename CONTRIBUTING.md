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
"""Module-wide docstring. Can be very short, as they generally don't contribute much."""


################################################################################
# Import
################################################################################


from __future__ import annotations
import numpy as np
# more imports


################################################################################
# Typing
################################################################################


import typing
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from encodermap._typing import AnyParameters
    # import more expensive libraries here


################################################################################
# Globals
################################################################################


GLOBAL_VARIABLE: dict = {"maybe": "some data"}
__all__: list[str] = ["public_function", "PublicClass"]


################################################################################
# Utils
################################################################################


def _private_function(*args: Union[float, str]) -> None:
    """A private function. Use type hints and google-style docstrings.

    Args:
        *args (Union[Number, str]): Any number of arguments can be
            Number or str types.

    """
    pass


################################################################################
# Public
################################################################################


def public_function(*args: float) -> None:
    """More docstrings. For all functions please."""
    print(args)
    return


class PublicClass:
    """Classes should have a general docstring, that explains everything."""
    def __init__(self, *args: float) -> None:
        """The docstring of the init can be shorter."""
        self.args_0  = args[0]

```

## Type hints and MyPy

EncoderMap tries to implement type hints where possible to make the code more reliable by using the static-type checker MyPy. Although python is a dynamically typed language declaring types can be beneficial for code robustness and documentation reasons.

## Unittests and Coverage

## Building the documentation and Sphinx

The documentation is built usi

## Vulture, pycodestyle, flake8 (linting) and black

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
