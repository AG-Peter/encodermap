# -*- coding: utf-8 -*-
# encodermap/misc/misc.py
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
"""Miscellaneous functions."""

##############################################################################
# Imports
##############################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile

# Third Party Imports
import numpy as np
import tensorflow as tf
from optional_imports import _optional_import


################################################################################
# Optional Imports
################################################################################


nx = _optional_import("networkx")
Image = _optional_import("PIL", "Image")
ImageDraw = _optional_import("PIL", "ImageDraw")
ImageFont = _optional_import("PIL", "ImageFont")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union, overload


if TYPE_CHECKING:
    # Standard Library Imports
    from pathlib import Path

    # Local Folder Imports
    from ..trajinfo.info_all import TrajEnsemble


################################################################################
# Globals
################################################################################


__all__: list[str] = [
    "create_n_cube",
    "plot_model",
    "run_path",
    "get_full_common_str_and_ref",
]


FEATURE_NAMES = {
    "AllCartesians": "all_cartesians",
    "AllBondDistances": "all_distances",
    "CentralCartesians": "central_cartesians",
    "CentralBondDistances": "central_distances",
    "CentralAngles": "central_angles",
    "CentralDihedrals": "central_dihedrals",
    "SideChainCartesians": "side_cartesians",
    "SideChainBondDistances": "side_distances",
    "SideChainAngles": "side_angles",
    "SideChainDihedrals": "side_dihedrals",
}


_TOPOLOGY_EXTS = [
    ".pdb",
    ".pdb.gz",
    ".h5",
    ".lh5",
    ".prmtop",
    ".parm7",
    ".prm7",
    ".psf",
    ".mol2",
    ".hoomdxml",
    ".gro",
    ".arc",
    ".hdf5",
    ".gsd",
]


################################################################################
# Utilities
################################################################################


def _is_notebook() -> bool:
    """Returns 'True', if we are currently in a notebook.

    Returns:
        bool: True if we are in a notebook, False otherwise.

    """
    try:
        # Third Party Imports
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def _can_be_feature(inp: Union[str, list[str]]) -> bool:
    """Function to decide whether the input can be interpreted by the Featurizer class.

    Outputs True, if inp == 'all' or inp is a list of strings contained in FEATURE_NAMES.

    Args:
        inp (Any): The input.

    Returns:
        bool: True, if inp can be interpreted by featurizer.

    Example:
        >>> from encodermap.misc.misc import _can_be_feature
        >>> _can_be_feature('all')
        True
        >>> _can_be_feature('no')
        False
        >>> _can_be_feature(['AllCartesians', 'central_dihedrals'])
        True

    """
    if isinstance(inp, str):
        if inp == "all":
            return True
    if isinstance(inp, list):
        if all([isinstance(i, str) for i in inp]):
            if all(
                [i in FEATURE_NAMES.keys() or i in FEATURE_NAMES.values() for i in inp]
            ):
                return True
    return False


def match_files(
    trajs: Union[list[str], list[Path]],
    tops: Union[list[str], list[Path]],
    common_str: list[str],
) -> tuple[list[str], list[str]]:
    """Matches trajectory and topology files with a list of common strings.

    The idea behind this function is to use one topology file with
    multiple trajectory files, that all use the same topology.

    Args:
        trajs (Union[list[str], list[Path]]): A list of str or list of Path objects.
            These are the trajectory files we want to match the subset of
            topology files with.
        tops (Union[list[str], list[Path]]): A list of str or list of Path objects.
            These are the topology files we will assign to the trajectory files
            using common substrings.
        common_str (list[str]): A list of common substrings.

    Returns:
        tuple[list[str], list[str]]: A tuple containing a two list of str.
            The first list of str are the topology files matched to the 'trajs'.
            The second list of str are the common_str matched to the 'trajs'.
            Both lists have the same length as `trajs`.

    """
    tops_out = []
    common_str_out = []

    trajs = list(map(str, trajs))
    tops = list(map(str, tops))

    if all([Path(p).suffix == ".h5" for p in trajs]) and len(trajs) == len(tops) == len(
        common_str
    ):
        return tops, common_str

    if len(trajs) == len(tops):
        try:
            first_cs = common_str[[cs in trajs[0] for cs in common_str].index(True)]
        except ValueError as e:
            raise Exception(
                f"Can't find any matching common_str in the input data:\n{trajs=}\n\n"
                f"{tops=}\n\n{common_str=}"
            ) from e
        if first_cs not in tops[0]:
            iterator = trajs
        else:
            iterator = zip(trajs, tops)
    else:
        iterator = trajs

    for t in iterator:
        if isinstance(t, tuple):
            t, top = t
            if not any([cs in t for cs in common_str]):
                raise Exception(
                    f"The traj file {t} does not match any of the common_str you provided."
                )
            else:
                t_lcut = max([t.rfind(cs) for cs in common_str])
                t_lcut = t[t_lcut:]
                cs = common_str[[cs in t_lcut for cs in common_str].index(True)]
                if t.split(".")[-1] == "h5":
                    tops_out.append(t)
                    assert cs in t, f"{cs=} {top=} {t=}"
                else:
                    tops_out.append(top)
                    assert cs in top and cs in t
                common_str_out.append(cs)
        else:
            if not any([cs in t for cs in common_str]):
                raise Exception(
                    f"The traj file {t} does not match any of the common_str you provided."
                )
            else:
                t_lcut = max([t.rfind(cs) for cs in common_str])
                t_lcut = t[t_lcut:]
                cs = common_str[[cs in t_lcut for cs in common_str].index(True)]
                if t.split(".")[-1] == "h5":
                    tops_out.append(trajs[[cs in r for r in trajs].index(True)])
                else:
                    tops_out.append(tops[[cs in r for r in tops].index(True)])
                assert cs in t
                common_str_out.append(cs)
    return tops_out, common_str_out


def get_full_common_str_and_ref(
    trajs: Union[list[str], list[Path]],
    tops: Union[list[str], list[Path]],
    common_str: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Matches traj_files, top_files and common string and returns lists with the
    same length matching the provided common str.

    Args:
        trajs (list[str]): A list of str pointing to trajectory files.
        tops (list[str]): A list of str pointing to topology files.
        common_str (list[str]): A list of strings that can be found in
            both trajs and tops (i.e. substrings).

    Args:
        tuple: A tuple containing the following:
            - list[str]: A list of str with the traj file names.
            - list[str]: A list of str with the top file names.
            - list[str]: A list of str with the common_str's.
            All lists have the same length.

    """
    assert isinstance(common_str, list)
    if len(trajs) != len(tops) and common_str == [] and len(tops) != 1:
        raise Exception(
            "When providing a list of trajs and a list of refs with different "
            "length you must provide a list of common_str to match them."
        )

    # if the length of all objects is the same we just return them
    # but also we check, whether the common_str appears in the
    # trajs and tops to make sure its correclty ordered
    if len(trajs) == len(tops) == len(common_str):
        if not all([i is None for i in common_str]):
            for traj, top, cs in zip(trajs, tops, common_str):
                if cs not in str(traj) or cs not in str(top):
                    return (trajs, *match_files(trajs, tops, common_str))
        return trajs, tops, common_str

    # if trajs and tops are the same length, they are expected to match
    elif len(trajs) == len(tops):
        if len(common_str) == 0:
            return trajs, tops, [None for i in trajs]

        elif len(common_str) == 1:
            return trajs, tops, [common_str[0] for i in trajs]

        else:
            tops_out, common_str_out = match_files(trajs, tops, common_str)
            return trajs, tops_out, common_str_out

    # if only one topology is provided, we hope the user passed a correct one and fill everything else up
    elif len(trajs) > 1 and len(tops) == 1:
        tops_out = [tops[0] for t in trajs]

        if common_str == []:
            common_str_out = [Path(traj).stem for traj in trajs]

        elif len(common_str) != len(trajs):
            tops_out, common_str_out = match_files(trajs, tops_out, common_str)

        elif len(common_str) == len(trajs):
            common_str_out = common_str

        return trajs, tops_out, common_str_out

    # in other cases, we need to do something similar
    else:
        if len(tops) > len(trajs):
            raise Exception(
                f"I was given more topologies {tops} than trajectories {trajs} . Something's not right."
            )
        if len(common_str) > len(trajs):
            raise Exception(
                "I was given more common strings than trajectories. Something's not right."
            )

        if common_str == []:
            common_str_out = [Path(traj).stem for traj in trajs]

        elif len(common_str) != len(trajs):
            tops_out, common_str_out = match_files(trajs, tops, common_str)

        elif len(common_str) == len(trajs):
            common_str_out = common_str

        tops_out, common_str_out = match_files(trajs, tops, common_str_out)
        return trajs, tops_out, common_str_out


def printTable(
    myDict: dict[str, dict[str, Any]],
    colList: Optional[list[str]] = None,
    sep: str = "\uFFFA",
) -> str:
    """Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    sep: row separator. Ex: sep='\n' on Linux. Default: dummy to not split line.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    out = []
    if not colList:
        colList = list(myDict[0].keys() if myDict else [])
    myList = [colList]  # 1st row = header
    for item in myDict:
        myList.append([str(item[col] or "") for col in colList])
    colSize = [max(map(len, (sep.join(col)).split(sep))) for col in zip(*myList)]
    formatStr = " | ".join(["{{:<{}}}".format(i) for i in colSize])
    line = formatStr.replace(" | ", "-+-").format(*["-" * i for i in colSize])
    item = myList.pop(0)
    lineDone = False
    while myList:
        if all(not i for i in item):
            item = myList.pop(0)
            if line and (sep != "\uFFFA" or not lineDone):
                out.append(line)
                lineDone = True
        row = [i.split(sep, 1) for i in item]
        out.append(formatStr.format(*[i[0] for i in row]))
        item = [i[1] if len(i) > 1 else "" for i in row]
    out = ["    " + i for i in out]
    points = "  \n".join(out)
    return points


##############################################################################
# Functions
##############################################################################


def _datetime_windows_and_linux_compatible() -> str:
    """Portable way to get `now` as either a linux or windows compatible string.

    For linux systems, strings in this manner will be returned:
        2022-07-13T16:04:04+02:00

    For Windows systems, strings in this manner will be returned:
        2022-07-13_16-04-46

    """
    # Standard Library Imports
    import datetime
    from sys import platform

    if platform == "linux" or platform == "linux2" or platform == "darwin":
        return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
    elif platform == "win32":
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def all_equal(iterable: Iterable[Any]) -> bool:
    """Returns True, when all elements in a list are equal

    Args:
        iterable (Iterable[Any]): An iterable of Any.

    Returns:
        bool: Whether all elements in `iterable` are equal.

    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _validate_uri(str_: Union[Path, str]) -> bool:
    """Checks whether the a str or Path is a valid uri.

    Args:
        str_ (Union[Path, str]): The str or Path to test.

    Returns:
        bool: True when `str_` is a valid URI, False otherwise.

    """
    # Standard Library Imports
    from urllib.parse import urlparse

    try:
        result = urlparse(str_)
        return all([result.scheme, result.path])
    except:
        return False


@overload
def _flatten_model(
    model_nested: tf.keras.Model,
    input_dim: Optional[Sequence[int]] = None,
    return_model: bool = True,
) -> tf.keras.Model: ...


@overload
def _flatten_model(
    model_nested: tf.keras.Model,
    input_dim: Optional[Sequence[int]] = None,
    return_model: bool = False,
) -> list[tf.keras.layers.Layer]: ...


def _flatten_model(
    model_nested: tf.keras.Model,
    input_dim: Optional[Sequence[int]] = None,
    return_model: bool = True,
) -> Union[tf.keras.Model, list[tf.keras.layers.Layer]]:
    """Flattens a nested tensorflow.keras.models.Model.

    Can be useful if a model consists of two sequential models and needs to
    be flattened to be plotted.

    Union[tf.keras.Model, list[tf.keras.layers.Layer]]:
        Either a tf.keras.Model or a list of layers.

    """
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    if return_model:
        model_flat = tf.keras.models.Sequential(layers_flat)
        model_flat.build(input_shape=(1, input_dim))
        return model_flat
    else:
        return layers_flat


def plot_model(
    model: tf.keras.Model,
    input_dim: Optional[Sequence[int]] = None,
) -> Optional[Image]:
    """Plots keras model using tf.keras.utils.plot_model"""
    # Encodermap imports
    from encodermap.models.models import SequentialModel

    if isinstance(model, SequentialModel) and input_dim is not None:
        model = _flatten_model(model, input_dim)
        with NamedTemporaryFile(suffix=".png") as f:
            try:
                _ = tf.keras.utils.plot_model(
                    model,
                    to_file=f.name,
                    show_shapes=True,
                    rankdir="TB",
                    expand_nested=True,
                )
            except:
                return
            return Image.open(f.name)
    with NamedTemporaryFile(suffix=".png") as f:
        try:
            _ = tf.keras.utils.plot_model(
                model,
                to_file=f.name,
                show_shapes=True,
                rankdir="TB",
                expand_nested=True,
            )
        except:
            return
        img = Image.open(f.name)
        # img.putalpha(alpha=255)
        cube_img = Image.open(Path(__file__).resolve().parent / "logo_cube_300.png")
        img.paste(cube_img, (800, 500), cube_img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf", size=40
            )
            ImageDraw.Draw(img).text(  # Image
                (700, 800),  # Coordinates
                "Made with EncoderMap",  # Text
                (0, 0, 0),  # Color
                font=font,
            )
        except:
            pass
        return img

        # plt.close("all")
        # plt.imshow(img)
        # plt.show()
        # if mpl.get_backend() == "module://ipykernel.pylab.backend_inline":
        #     fig = plt.gcf()
        #     fig.set_size_inches(fig.get_size_inches() * 4)
        # ax = plt.gca()
        # ax.axis("off")
        # plt.show()


def run_path(path: str) -> str:
    """Creates a directory at "path/run{i}" where the i is corresponding to the
    smallest not yet existing path.

    Args:
        path (str): Path to the run folder.

    Returns:
        str: The new output path.

    Exampples:
        >>> import encodermap as em
        >>> import tempfile
        >>> import os
        >>>
        >>> def sort_key(inp: str) -> int:
        ...     return int(inp[-1])
        >>>
        >>> with tempfile.TemporaryDirectory() as td:
        ...     # create some directories
        ...     os.makedirs(os.path.join(td, "run0"))
        ...     os.makedirs(os.path.join(td, "run1"))
        ...     # em.misc.run_path will automatically advance the counter to 'run2'
        ...     new_path = em.misc.run_path(td)
        ...     print(new_path)
        ...     print(sorted(os.listdir(td), key=sort_key))  # doctest: +ELLIPSIS
        /tmp/.../run2
        ['run0', 'run1', 'run2']

    """
    i = 0
    while True:
        current_path = os.path.join(path, "run{}".format(i))
        if not os.path.exists(current_path):
            os.makedirs(current_path)
            output_path = current_path
            break
        else:
            i += 1
    return output_path


def create_n_cube(
    n: int = 3,
    points_along_edge: int = 500,
    sigma: float = 0.05,
    same_colored_edges: int = 3,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates points along the edges of an n-dimensional unit hyper-cube.

    The cube is created using `networkx.hypercube_graph` and points are placed along
    the edges of the cube. By providing a `sigma` value, the points can be shifted
    by some Gaussian noise.

    Args:
        n (int): The dimension of the Hypercube (can also take 1 or 2).
            Defaults to 3.
        points_along_edge (int): How many points should be placed along any edge.
            By increasing the number of dimensions, the number of edges
            increases, which also increases the total number of points. Defaults to 500.
        sigma (float): The sigma value for np.random.normal which
            introduces Gaussian noise to the positions of the points. Defaults to 0.05.
        same_color_edges (int): How many edges of the Hypercube should
            be colored with the same color. This can be used to later
            better visualize the edges of the cube. Defaults to 3.
        seed (Optional[int]): If an int is provided, this will be used as a seed
            for `np.random` and fix the random state. Defaults to None which produces
            random results every time this function is called.

    Returns:
        tuple: A tuple containing the following:
            - np.ndarray: The coordinates of the points.
            - np.ndarray: Integers that can be used for coloration.

    Example:
        >>> from encodermap.misc.misc import create_n_cube
        >>> # A sigma value of zero means no noise at all.
        >>> coords, colors = create_n_cube(2, sigma=0)
        >>> coords[0]
        array([0., 0.])

    """
    if seed is not None:
        np.random.seed(seed=seed)
    # create networkx hypercube with given dimensions
    G = nx.hypercube_graph(n)

    # vertices are not really needed
    vertices = np.array([n for n in G.nodes])
    # get edges
    edges = np.array([e for e in G.edges])

    # fill this list with values
    coordinates = []

    # iterate over edges
    for i, edge in enumerate(edges):
        # some basic analytic geomerty
        A, B = edge
        AB = B - A
        # n points along edge
        lin = np.linspace(0, 1, points_along_edge)
        points = A + (AB[:, None] * lin).T
        if sigma:
            points += np.random.normal(scale=sigma, size=(len(points), n))
        # add label for colors
        points = np.hstack([points, np.full((len(points), 1), i)])
        coordinates.extend(points)

    # make big numpy array
    coordinates = np.array(coordinates)

    # color the specified number of same colored edges
    # choose a random edge
    found_edges = []
    edge_pairs = []

    # iterate over the number of same colore edges
    for _ in range(same_colored_edges):
        for i, edge in enumerate(edges):
            if i in found_edges:
                continue
            found_edges.append(i)
            vertex = edge[0]
            where = np.where(np.all(edges[:, 0] == vertex, axis=1))[0]
            for j in where:
                new_edge = edges[j]
                if j not in found_edges:
                    found_edges.append(j)
                    break
            if i != j:
                edge_pairs.append([i, j])
                break

    # replace the corresponding indices
    for i, j in edge_pairs:
        new = coordinates[coordinates[:, -1] == i]
        new[:, -1] = np.full(points_along_edge, j)
        coordinates[coordinates[:, -1] == i] = new

    return coordinates[:, :-1], coordinates[:, -1]
