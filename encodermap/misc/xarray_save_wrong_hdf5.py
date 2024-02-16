# -*- coding: utf-8 -*-
# encodermap/misc/xarray_save_wrong_hdf5.py
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
"""Allows the combined storing of CVs and trajectories in single HDF5/NetCDF4 files.

These files represent collated and completed trajectory ensembles, which can be
lazy-loaded (memory efficient) and used as training input for EncoderMap's NNs.

"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os
import re
from io import BytesIO
from numbers import Number
from pathlib import Path

# Third Party Imports
import numpy as np
from optional_imports import _optional_import


################################################################################
# Optional Imports
################################################################################


xr = _optional_import("xarray")
AbstractDataStore = _optional_import("xarray", "backends.common.AbstractDataStore")
ArrayWriter = _optional_import("xarray", "backends.common.ArrayWriter")
Dataset = _optional_import("xarray", "core.dataset.Dataset")
backends = _optional_import("xarray", "backends")
_get_scheduler = _optional_import("xarray", "backends.locks._get_scheduler")
conventions = _optional_import("xarray", "conventions")
_get_default_engine = _optional_import("xarray", "backends.api._get_default_engine")


try:
    # Third Party Imports
    from dask.delayed import Delayed
except ImportError:
    Delayed = None


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Optional, Union


WritableStoresType = dict[str, Callable]

try:
    WRITEABLE_STORES: dict[str, Callable] = {
        "netcdf4": backends.NetCDF4DataStore.open,
        "scipy": backends.ScipyDataStore,
        "h5netcdf": backends.H5NetCDFStore.open,
    }
except (ImportError, ValueError, AttributeError):
    WRITEABLE_STORES: dict[str, Callable] = {}

if TYPE_CHECKING:
    # Third Party Imports
    from dask.delayed import Delayed
    from xarray import Dataset, backends, conventions
    from xarray.backends.api import _get_default_engine
    from xarray.backends.common import AbstractDataStore, ArrayWriter
    from xarray.backends.locks import _get_scheduler


################################################################################
# Globals
################################################################################


__all__ = ["save_netcdf_alongside_mdtraj"]


################################################################################
# Public functions
################################################################################


def save_netcdf_alongside_mdtraj(fname: str, dataset: Dataset) -> None:
    _to_netcdf(
        dataset,
        fname,
        mode="a",
        format="NETCDF4",
        group="CVs",
        engine="h5netcdf",
        invalid_netcdf=True,
    )


################################################################################
# xarray duplication to allow saving dataset alongside mdtraj
################################################################################


def dump_to_store(
    dataset: Dataset,
    store: WritableStoresType = WRITEABLE_STORES,
    writer: Optional[Callable] = None,
    encoder: Optional[Callable] = None,
    encoding: Optional[str] = None,
    unlimited_dims: Optional[Iterable[Hashable]] = None,
):
    """Store dataset contents to a backends.*DataStore object."""
    if writer is None:
        writer = ArrayWriter()

    if encoding is None:
        encoding = {}

    variables, attrs = conventions.encode_dataset_coordinates(dataset)

    check_encoding = set()
    for k, enc in encoding.items():
        # no need to shallow copy the variable again; that already happened
        # in encode_dataset_coordinates
        variables[k].encoding = enc
        check_encoding.add(k)

    if encoder:
        variables, attrs = encoder(variables, attrs)

    store.store(variables, attrs, check_encoding, writer, unlimited_dims=unlimited_dims)


def _normalize_path(path: str) -> str:
    if is_remote_uri(path):
        return path
    else:
        return os.path.abspath(os.path.expanduser(path))


def is_remote_uri(path: str) -> bool:
    return bool(re.search(r"^https?\://", path))


def _validate_dataset_names(dataset: Dataset) -> None:
    """DataArray.name and Dataset keys must be a string or None"""

    def check_name(name):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    "Invalid name for DataArray or Dataset key: "
                    "string must be length 1 or greater for "
                    "serialization to netCDF files"
                )
        elif name is not None:
            raise TypeError(
                "DataArray.name or Dataset key must be either a "
                "string or None for serialization to netCDF files"
            )

    for k in dataset.variables:
        check_name(k)


def _validate_attrs(dataset: Dataset) -> None:
    """`attrs` must have a string key and a value which is either: a number,
    a string, an ndarray or a list/tuple of numbers/strings.
    """

    def check_attr(name, value):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    "Invalid name for attr: string must be "
                    "length 1 or greater for serialization to "
                    "netCDF files"
                )
        else:
            raise TypeError(
                "Invalid name for attr: {} must be a string for "
                "serialization to netCDF files".format(name)
            )

        if not isinstance(value, (str, Number, np.ndarray, np.number, list, tuple)):
            raise TypeError(
                "Invalid value for attr: {} must be a number, "
                "a string, an ndarray or a list/tuple of "
                "numbers/strings for serialization to netCDF "
                "files".format(value)
            )

    # Check attrs on the dataset itself
    for k, v in dataset.attrs.items():
        check_attr(k, v)

    # Check attrs on each variable within the dataset
    for variable in dataset.variables.values():
        for k, v in variable.attrs.items():
            check_attr(k, v)


def _to_netcdf(
    dataset: Dataset,
    path_or_file: Optional[str] = None,
    mode: Optional[str] = "w",
    format: Optional[str] = None,
    group: Optional[str] = None,
    engine: Optional[str] = None,
    encoding: Optional[Mapping] = None,
    unlimited_dims: Optional[Iterable[Hashable]] = None,
    compute: bool = True,
    multifile: bool = False,
    invalid_netcdf: bool = False,
) -> Optional[Delayed]:
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    if isinstance(path_or_file, Path):
        path_or_file = str(path_or_file)

    if encoding is None:
        encoding = {}

    if path_or_file is None:
        if engine is None:
            engine = "scipy"
        elif engine != "scipy":
            raise ValueError(
                "invalid engine for creating bytes with "
                "to_netcdf: %r. Only the default engine "
                "or engine='scipy' is supported" % engine
            )
        if not compute:
            raise NotImplementedError(
                "to_netcdf() with compute=False is not yet implemented when "
                "returning bytes"
            )
    elif isinstance(path_or_file, str):
        if engine is None:
            engine = _get_default_engine(path_or_file)
        path_or_file = _normalize_path(path_or_file)
    else:  # file-like object
        engine = "scipy"

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError("unrecognized engine for to_netcdf: %r" % engine)

    if format is not None:
        format = format.upper()

    # handle scheduler specific logic
    scheduler = _get_scheduler()
    have_chunks = any(v.chunks for v in dataset.variables.values())

    autoclose = have_chunks and scheduler in ["distributed", "multiprocessing"]
    if autoclose and engine == "scipy":
        raise NotImplementedError(
            "Writing netCDF files with the %s backend "
            "is not currently supported with dask's %s "
            "scheduler" % (engine, scheduler)
        )

    target = path_or_file if path_or_file is not None else BytesIO()
    kwargs = dict(autoclose=True) if autoclose else {}
    # added phony dims support
    if engine == "h5netcdf":
        kwargs.update(dict(phony_dims="access"))
    if invalid_netcdf:
        if engine == "h5netcdf":
            kwargs["invalid_netcdf"] = invalid_netcdf
        else:
            raise ValueError(
                "unrecognized option 'invalid_netcdf' for engine %s" % engine
            )
    store = store_open(target, mode, format, group, **kwargs)

    if unlimited_dims is None:
        unlimited_dims = dataset.encoding.get("unlimited_dims", None)
    if unlimited_dims is not None:
        if isinstance(unlimited_dims, str) or not isinstance(unlimited_dims, Iterable):
            unlimited_dims = [unlimited_dims]
        else:
            unlimited_dims = list(unlimited_dims)

    writer = ArrayWriter()

    # TODO: figure out how to refactor this logic (here and in save_mfdataset)
    # to avoid this mess of conditionals
    try:
        # TODO: allow this work (setting up the file for writing array data)
        # to be parallelized with dask
        dump_to_store(
            dataset, store, writer, encoding=encoding, unlimited_dims=unlimited_dims
        )
        if autoclose:
            store.close()

        if multifile:
            return writer, store

        writes = writer.sync(compute=compute)

        if path_or_file is None:
            store.sync()
            return target.getvalue()
    finally:
        if not multifile and compute:
            store.close()

    if not compute:
        # Third Party Imports
        import dask

        return dask.delayed(_finalize_store)(writes, store)
    return None
