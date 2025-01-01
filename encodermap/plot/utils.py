# -*- coding: utf-8 -*-
# encodermap/plot/utils.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
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

"""Utils for plotting. Ideas and Code taken from the matplotlib documentation.
Menu: https://matplotlib.org/3.1.0/gallery/widgets/menu.html
LassoSelector: https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html
PolygonSelector: https://matplotlib.org/3.1.3/gallery/widgets/polygon_selector_demo.html
Bezier: https://gist.github.com/gavincangan/b88a978e878e9bb1c0f8804e3af8de3c

"""

################################################################################
# Imports
################################################################################

# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy
import os
import shutil
import warnings

# Third Party Imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, PolygonSelector, RectangleSelector
from optional_imports import _optional_import
from packaging import version
from scipy.special import binom
from tqdm import tqdm

# Encodermap imports
from encodermap.misc.misc import _datetime_windows_and_linux_compatible


################################################################################
# Optional Imports
################################################################################


jinja2 = _optional_import("jinja2")
ngl = _optional_import("nglview")
mda = _optional_import("MDAnalysis")
md = _optional_import("mdtraj")
pd = _optional_import("pandas")
binom = _optional_import("scipy", "special.binom")
plotly_lasso = _optional_import("plotly", "callbacks.LassoSelector")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from typing import TYPE_CHECKING, Any, Optional, Union


if TYPE_CHECKING:
    # Local Folder Imports
    from ..trajinfo.info_all import TrajEnsemble


################################################################################
# Globals
################################################################################

_all__ = [
    "Props",
    "SelectFromCollection",
    "StatusMenu",
    "Menu",
    "ModeButton",
    "MenuItem",
]


################################################################################
# Functions (mainly for ternary plotting)
################################################################################


def abc_to_rgb(A=0.0, B=0.0, C=0.0):
    """Map values A, B, C (all in domain [0,1]) to
    suitable red, green, blue values."""
    return (min(B + C, 1.0), min(A + C, 1.0), min(A + B, 1.0))


def digitize_dssp(trajs, dssp, imshow=True, bins=100, progbar=None):
    """Digitizes the DSSP array"""
    if progbar is None:
        progbar = tqdm(total=all_dssp.shape[0] * 4, position=0, leave=True)
    if np.any(np.isin(dssp, ["H", "E", "C"], assume_unique=False)) and not "B" in dssp:
        sorted_ = np.array(["H", "E", "C"])
    else:
        sorted_ = np.array(["H", "B", "E", "G", "I", "T", "S", ""])
    num_aas = dssp.shape[1]
    uniques = correct_missing_uniques(
        [np.concatenate(np.unique(d, return_counts=True)) for d in dssp],
        sorted_=sorted_,
        progbar=progbar,
    )
    uniques, counts = uniques[:, :3], uniques[:, 3:]
    indices = np.vstack([u.argsort()[sorted_.argsort()] for u in uniques])
    counts = (
        np.vstack([c[i] for c, i in zip(counts, indices)]).astype(np.float) / num_aas
    )
    colors = np.array([abc_to_rgb(a, b, c) for a, b, c in zip(*counts.T)])
    if imshow:
        print("digitizing")
        digitized = np.full((bins, bins, 3), (1.0, 1.0, 1.0))
        x = trajs.lowd[:, 0]
        y = trajs.lowd[:, 1]
        H, xedges, yedges = np.histogram2d(x=x, y=y, bins=bins)
        for i in range(bins):
            for j in range(bins):
                where = np.where(
                    (
                        (trajs.lowd[:, 0] >= xedges[i])
                        & (trajs.lowd[:, 0] < xedges[i + 1])
                    )
                    & (
                        (trajs.lowd[:, 1] >= yedges[j])
                        & (trajs.lowd[:, 1] < yedges[j + 1])
                    )
                )[0]
                if len(where) != 0:
                    counts_ = np.mean(counts[where], axis=0)
                    try:
                        color = abc_to_rgb(*counts_)
                    except TypeError:
                        print(counts[where], counts_)
                        raise
                    digitized[i, j] = color
                progbar.update()
        return digitized
    else:
        return colors


def correct_missing_uniques(uniques, sorted_, progbar=None):
    """Takes a list of DSSP letters and counts and adds zeros for all missing letters in sorted_.

    Args:
        uniques (list of np.ndarray): A list of np.ndarrays. For every frame the DSSP letters
            and their respective counts should be provided. So for example
            ['H', 151] or ['H', 'E', 75, 76].
        sorted_ (np.ndarray): The sorted list of DSSP letters. This list will be used to update every
            frame. So The using ['H', 'E', 'C'], the two examples from above will become ['H', 'E', 'C', 151, 0, 0]
            or ['H', 'E', 'C', 75, 76, 0], respectively.
        progbar (tqdm): A tqdm progbar. Defaults to None

    Returns:
        np.ndarray: The corrected uniques.

    """
    if progbar is None:
        progbar = tqdm(total=len(uniques), position=0, leave=True)
    for i, u in enumerate(uniques):
        if len(u) != len(sorted_) * 2:
            letters, counts = np.split(u, 2)
            counts = counts.astype(int)
            newline = np.zeros(len(sorted_), dtype=int)
            for l, c in zip(letters, counts):
                newline[np.where(sorted_ == l)] = c
            newline = np.concatenate([sorted_, newline.astype(str)])
            uniques[i] = newline
        progbar.update()
    uniques = np.vstack(uniques)
    return uniques


def _get_system_info() -> dict[str, Any]:
    # Standard Library Imports
    import getpass
    import platform
    import re
    import socket
    import uuid

    # Third Party Imports
    import psutil

    info = {}
    try:
        info["platform"] = platform.system()
        info["system_user"] = getpass.getuser()
        info["platform_release"] = platform.release()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip_address"] = socket.gethostbyname(socket.gethostname())
        info["mac_address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
        info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        return info
    except Exception:
        return info


def _check_all_templates_defined(template, info_dict):
    # Standard Library Imports
    import re

    regex = r"\{(.*?)\}"
    matches = re.finditer(regex, template, re.MULTILINE | re.DOTALL)
    min_matches = []
    for matchNum, match in enumerate(matches):
        for groupNum in range(0, len(match.groups())):
            min_matches.append(match.group(groupNum))
    min_matches = list(
        set(map(lambda x: x.lstrip("{{").rstrip("}}"), [i for i in min_matches]))
    )
    if all(key in info_dict for key in min_matches):
        return True
    else:
        missing = set(min_matches).difference(info_dict)
        raise Exception(
            f"Not all expressions defined in template. Missing expressions: {missing}"
        )


def _create_readme(main_path, now, info_dict):  # pragma: no cover
    # Third Party Imports
    from pip._internal.operations import freeze

    # Local Folder Imports
    from .._version import get_versions

    __version__ = get_versions()["version"]
    # Local Folder Imports
    from .jinja_template import template

    # update info dict
    md_file = os.path.join(main_path, "README.md")
    pip_freeze = ""
    for i in freeze.freeze():
        pip_freeze += f"    {i}\n"
    info_dict.update({"pip_freeze": pip_freeze})
    info_dict.update({"filename": md_file.split(".")[0]})
    info_dict.update({"now": now})
    info_dict.update({"encodermap_version": __version__})
    info_dict.update(_get_system_info())

    # check
    assert _check_all_templates_defined(template, info_dict)

    # jinja2
    template = jinja2.Template(template)
    msg = template.render(
        info_dict,
    )

    # write
    with open(md_file, "w") as f:
        f.write(msg)


def _unpack_cluster_info(
    trajs: TrajEnsemble,
    main_path: Union[Path, str],
    selector: Any,
    dummy_traj: TrajEnsemble,
    align_string: str,
    col: str,
    display: Any,
    progbar: Any,
) -> tuple[int, Path]:
    # Standard Library Imports
    from pathlib import Path

    main_path = Path(main_path)
    max_ = trajs.CVs[col].max()
    where = np.where(trajs.CVs[col] == max_)[0]
    length = len(where)
    now = _datetime_windows_and_linux_compatible()

    # make dirs
    os.makedirs(os.path.join(main_path, "clusters"), exist_ok=True)
    main_path = os.path.join(main_path, f"clusters/{now}")
    os.makedirs(main_path, exist_ok=True)
    progbar.update()

    # define names
    h5_name = os.path.join(main_path, f"cluster_id_{max_}_stacked_{length}_structs.h5")
    pdb_start_name = os.path.join(main_path, f"cluster_id_{max_}_start.pdb")
    pdb_origin_names = os.path.join(main_path, f"cluster_id_{max_}_pdb_origins.txt")
    xtc_name = os.path.join(main_path, f"cluster_id_{max_}.xtc")
    csv_name = os.path.join(main_path, f"cluster_id_{max_}_selected_points.csv")
    png_name = os.path.join(main_path, f"cluster_id_{max_}_image.png")
    lowd_npy_name = os.path.join(
        main_path, f"cluster_id_{max_}_cluster_lowd_points.npy"
    )
    indices_npy_name = os.path.join(
        main_path, f"cluster_id_{max_}_cluster_lowd_points_indices.npy"
    )
    current_clustering = os.path.join(
        main_path,
        f"cluster_id_{max_}_cluster_current_clustering_%s.npy" % col,
    )
    selector_npy_name = os.path.join(
        main_path, f"cluster_id_{max_}_selector_points.npy"
    )
    parents_trajs = os.path.join(
        main_path, f"cluster_id_{max_}_all_plotted_trajs_in_correct_order.txt"
    )

    # save edges of selector
    try:
        verts = np.vstack([selector.xs, selector.ys]).T
        selector_npy_name = selector_npy_name.replace(
            "selector", f"{selector.__class__.__name__.lower()}"
        )
        np.save(selector_npy_name, verts)
    except Exception as e:
        display.outputs = []
        with display:
            print(f"Currently only plotly's LassoSelector is available. Exception: {e}")
        return
    progbar.update()

    # save the output as a h5 file, so we can also save CVs and lowd
    dummy_traj.save(h5_name)
    progbar.update()

    # render png
    # plot_cluster(trajs, h5_name, png_name, max_)

    # save all trajs
    with open(parents_trajs, "w") as f:
        for traj in trajs:
            f.write(
                f"{os.path.abspath(traj.traj_file)} {os.path.abspath(traj.top_file)} {traj.traj_num} {traj.common_str}\n"
            )

    # create df
    if trajs.lowd.shape[-1] == 2:
        lowd_coords = {"x": [], "y": []}
    elif trajs.lowd.shaoe[-1] == 3:
        lowd_coords = {"x": [], "y": [], "z": []}
    else:
        lowd_coords = {f"lowd_{i}": [] for i in range(trajs.lowd.shape[-1])}
    progbar.update()
    df = pd.DataFrame(
        {
            "trajectory file": [],
            "topology file": [],
            "frame number": [],
            "time": [],
            **lowd_coords,
            "cluster id": [],
            "trajectory number": [],
        }
    )
    # display.outputs = []
    # with display:
    #     print(f"Dataframe created {df.shape=}. {where=}")

    progbar.update()
    for frame_num, frame in dummy_traj.iterframes():
        if version.parse(pd.__version__) >= version.parse("2.0.0"):
            df.loc[len(df)] = pd.Series(
                {
                    "trajectory file": os.path.abspath(frame.traj_file),
                    "topology file": os.path.abspath(frame.top_file),
                    "frame number": frame_num,
                    "time": frame.time[0],
                    "cluster id": max_,
                    "trajectory number": frame.traj_num,
                    **{k: v for k, v in zip(lowd_coords.keys(), frame.lowd[0])},
                }
            )
        else:
            df = df.append(
                {
                    "trajectory file": os.path.abspath(frame.traj_file),
                    "topology file": os.path.abspath(frame.top_file),
                    "frame number": frame_num,
                    "time": frame.time[0],
                    "cluster id": max_,
                    "trajectory number": frame.traj_num,
                    **{k: v for k, v in zip(lowd_coords.keys(), frame.lowd[0])},
                },
                ignore_index=True,
            )
    display.outputs = []
    progbar.update()
    df = df.astype(
        dtype={
            "trajectory file": str,
            "topology file": str,
            "frame number": int,
            "time": float,
            **{k: float for k in lowd_coords},
            "cluster id": int,
            "trajectory number": int,
        }
    )
    df.to_csv(csv_name, index=False)
    progbar.update()

    # save npy
    np.save(lowd_npy_name, trajs.CVs[col][where])
    np.save(indices_npy_name, where)
    np.save(current_clustering, trajs.CVs[col])
    progbar.update()

    # save full traj
    progbar.update()
    with open(pdb_origin_names, "w") as f:
        for i, (top, value) in enumerate(dummy_traj.trajs_by_top.items()):
            _pdb_start_name = pdb_start_name.replace(
                ".pdb", f"_traj_{i}_from_{trajs.basename_fn(value.top_files[0])}.pdb"
            )
            _xtc_name = xtc_name.replace(".xtc", f"_traj_{i}.xtc")
            joined = value.join(progbar=False)[top]
            joined[0].save_pdb(_pdb_start_name)
            # shutil.copyfile(key, _pdb_start_name)
            f.write(
                f"{_pdb_start_name} is a copy (`shutil.copyfile`) of "
                f"{value.top_files[0]}. The corresponding trajectory files might "
                f"originate from other places. Refer to {parents_trajs} for info about xtcs.\n"
            )
            joined.save_xtc(_xtc_name)

        # create an info dict
        # Local Folder Imports
        from .jinja_template import h5_parents, h5_rebuild, xtc_parents, xtc_rebuild

        info_dict = {
            "h5_name": h5_name,
            "pdb_start_name": pdb_start_name,
            "pdb_origin_names": pdb_origin_names,
            "xtc_name": xtc_name,
            "csv_name": csv_name,
            "png_name": png_name,
            "lowd_npy_name": lowd_npy_name,
            "indices_npy_name": indices_npy_name,
            "current_clustering": current_clustering,
            "selector_npy_name": selector_npy_name,
            "parents_trajs": parents_trajs,
        }

        if all([t.extension == ".h5" for t in trajs]):
            template = jinja2.Template(h5_rebuild)
            rebuild_clustering_info = template.render(
                {"h5_file": trajs[0]._traj_file.resolve(), **info_dict},
            )
            template = jinja2.Template(h5_parents)
            parents_trajs = template.render(
                {"h5_file": trajs[0]._traj_file.resolve(), **info_dict},
            )
        else:
            template = jinja2.Template(xtc_rebuild)
            rebuild_clustering_info = template.render(
                info_dict,
            )
            template = jinja2.Template(xtc_parents)
            parents_trajs = template.render(
                info_dict,
            )

        info_dict["parents_trajs"] = parents_trajs
        info_dict["rebuild_clustering_info"] = rebuild_clustering_info
        info_dict = {k: os.path.basename(v) for k, v in info_dict.items()}
        info_dict.update({"cluster_id": max_})
        info_dict.update({"cluster_id": max_})
        info_dict.update({"n_trajs": trajs.n_trajs})
        info_dict.update({"cluster_n_points": len(where)})
        info_dict.update({"basename": trajs[0].basename})
        info_dict.update({"cluster_abspath": main_path})

        # create a readme
        _create_readme(main_path, now, info_dict)
        progbar.update()

    return max_, main_path


################################################################################
# Classes
################################################################################


class Props:
    """Class to contain the properties of MenuItems.

    Each MenuItem contains two copies of this class.
    One for props when the mouse cursor hovers over them.
    One for the rest of times.
    Can be used as class or as dict.

    Attributes:
        labelcolor (str): The color of the text.
        labelcolor_rgb (tuple): The RGBA value of labelcolor.
        bgcolor (str): The color of the background.
        bgcolor_rgb (tuple): The RGBA value of bgcolor.
        fontsize (int): The fontsize.
        alpha (float): The alpha value of the background.
        defaults (dict): Class Variable of default values.

    Examples:
        >>> props = Props()
        >>> props.labelcolor
        'black'

    """

    defaults = {"labelcolor": "black", "bgcolor": "blue", "fontsize": 15, "alpha": 0.2}

    def __init__(self, **kwargs):
        """Instantiate the Props class.

        Takes a dict as input and overwrites the class defaults. The dict is directly
        stored as an attribute and can be accessed via dot-notation.

        Args:
            **kwargs: dict containing values. If unknonwn values are passed they will be dropped.

        """
        self._setattr(self.defaults)
        for key, value in kwargs.items():
            if key not in self.__dict__.keys():
                print(f"Dropping unknown dict entry for {{'{key}': {value}}}")
            else:
                setattr(self, key, value)

    @property
    def bgcolor_rgb(self):
        return mpl.colors.to_rgba(self.bgcolor)

    @property
    def labelcolor_rgb(self):
        return mpl.colors.to_rgba(self.labelcolor)

    def _setattr(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __setitiem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def _string_summary(self):
        return "ep.plotting.interactive.Props class containing Info for the MenuItem class."

    def __str__(self):
        return self._string_summary()

    def __repr__(self):
        return f"<{self._string_summary()} Object at 0x{id(self):02x}>"


class BezierBuilder(object):
    """Bézier curve interactive builder."""

    def __init__(self, control_polygon, ax_main, ax_bernstein=None):
        """Constructor.
        Receives the initial control polygon of the curve.
        """
        self.control_polygon = control_polygon
        self.xp = list(control_polygon.get_xdata())
        self.yp = list(control_polygon.get_ydata())
        self.canvas = control_polygon.figure.canvas
        self.ax_main = ax_main
        self.ax_bernstein = ax_bernstein

        # Event handler for mouse clicking
        self.cid = self.canvas.mpl_connect("button_press_event", self)

        # Create Bézier curve
        line_bezier = Line2D([], [], c=control_polygon.get_markeredgecolor())
        self.bezier_curve = self.ax_main.add_line(line_bezier)

    def __call__(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.control_polygon.axes:
            return

        # Add point
        self.xp.append(event.xdata)
        self.yp.append(event.ydata)
        self.control_polygon.set_data(self.xp, self.yp)

        # Rebuild Bézier curve and update canvas
        self.bezier_curve.set_data(*self._build_bezier())
        self._update_bernstein()
        self._update_bezier()

    def _build_bezier(self):
        x, y = Bezier(list(zip(self.xp, self.yp))).T
        return x, y

    def _update_bezier(self):
        self.canvas.draw()

    def _update_bernstein(self):
        if self.ax_bernstein is not None:
            N = len(self.xp) - 1
            t = np.linspace(0, 1, num=200)
            ax = self.ax_bernstein
            ax.clear()
            for kk in range(N + 1):
                ax.plot(t, Bernstein(N, kk)(t))
            ax.set_title("Bernstein basis, N = {}".format(N))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    @property
    def ind(self):
        return np.vstack(self.bezier_curve.get_data()).T

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid)


def Bernstein(n, k):
    """Bernstein polynomial."""
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x**k * (1 - x) ** (n - k)

    return _bpoly


def Bezier(points, num=200):
    """Build Bézier curve from points."""
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for ii in range(N):
        curve += np.outer(Bernstein(N - 1, ii)(t), points[ii])
    return curve


class DummyTool:
    def disconnect(self):
        pass


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Args:
        ax (matplotlib.axes.Axes): Axes to interact with.
        collection (matplotlib.collections.Collection): Subclass of collection
            you want to select from.
        alpha_other (float): To highlight a selection, this tool sets all
            selected points to an alpha value of 1 and non-selected points to
            `alpha_other`. Needs to fulfill 0 <= alpha_other <= 1

    """

    def __init__(self, ax, collection, alpha_other=0.3, selector=LassoSelector):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = selector(ax, onselect=self.onselect, useblit=False)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def _string_summary(self):
        s = (
            f"encodermap.SelectFromCollection object. Selecting points "
            f"from a set of {self.Npts} points. The current selector tool "
            f"is matplotlotlib's {self.lasso} tool. The selected points are "
            f"{self.ind}"
        )
        return s

    def __str__(self):
        return self._string_summary()

    def __repr__(self):
        return self._string_summary()


class MenuItem(mpl.patches.Rectangle, mpl.artist.Artist):
    def __init__(
        self,
        fig,
        xy,
        width,
        height,
        labelstr,
        props={},
        hoverprops={},
        on_select=None,
        standalone=False,
    ):
        # define props and hoverprops
        self.labelstr = labelstr
        self.standalone = standalone
        props = {
            **{"labelcolor": "black", "bgcolor": "blue", "fontsize": 15, "alpha": 0.2},
            **props,
        }
        hoverprops = {
            **{"labelcolor": "C0", "bgcolor": "yellow", "fontsize": 15, "alpha": 0.2},
            **hoverprops,
        }
        self.props = Props(**props)
        self.hoverprops = Props(**hoverprops)

        # set the on_select method
        self.on_select = on_select

        # add a select event
        # hover event is only included if standalone is true
        # i.e. the MenuItem is not Instantiated from a Menu
        # In the case this is instantiated from a menu. The menu wil handle the hover
        self.cid_button = fig.canvas.mpl_connect(
            "button_release_event", self.check_select
        )
        if self.standalone:
            self.cid_move = fig.canvas.mpl_connect(
                "motion_notify_event", self.set_hover
            )

        # Instantiate text and Rectangle
        mpl.patches.Rectangle.__init__(
            self, xy, width, height, label=self.labelstr, zorder=1
        )
        self.text = mpl.text.Text(
            0,
            0,
            self.labelstr,
            color=self.props.labelcolor,
            fontproperties=dict(weight="bold", size=self.props.fontsize),
            zorder=2,
            verticalalignment="center",
            horizontalalignment="center",
        )

        # final thing to do is set the props of
        # the rectangle based whether a cursor hovers
        self.set_hover_props(False)

        # after artist has been placed get it ready
        # for accepting hover events
        self.hover = False

    def set_figure(self, figure):
        """Overwriting Base Class method to include labelstr"""
        mpl.patches.Rectangle.set_figure(self, figure)
        self.text.set_figure(figure)

    def set_axes(self, axes):
        """Overwriting Base Class method to include labelstr"""
        mpl.patches.Rectangle.set_axes(self, axes)
        self.text.set_axes(axes)

    def set_transform(self, transform):
        """Overwriting Base Class method to include labelstr"""
        mpl.patches.Rectangle.set_transform(self, transform)
        # set text to center of self(.rect)
        bbox = self.get_bbox()
        x = bbox.x0 + 0.5 * bbox.width
        y = bbox.y0 + 0.5 * bbox.height
        texttrans = mpl.transforms.Affine2D().translate(x, y) + self.axes.transData
        self.text.set_transform(texttrans)

    def set_data(self, x, y):
        """Overwriting Base Class method to include labelstr"""
        # if len(x):
        #     self.text.set_position((x[-1], y[-1]))
        mpl.patches.Rectangle.set_data(self, x, y)
        self.text.set_color(self.props.labelcolor_rgb)

    def check_select(self, event):
        over, _ = self.contains(event)
        if not over:
            return
        if self.on_select is not None:
            self.on_select(self)
        return True

    def draw(self, renderer):
        """Overwriting Base Class method to include labelstr"""
        # draw my label at the end of the line with 2 pixel offset
        mpl.patches.Rectangle.draw(self, renderer)
        self.text.draw(renderer)

    def set_hover_props(self, check):
        if check:
            props = self.hoverprops
        else:
            props = self.props

        self.set(facecolor=props.bgcolor_rgb, alpha=props.alpha)
        self.text.set_color(props.labelcolor_rgb)

    def set_hover(self, event):
        """Check the hover status of MenuItem"""
        check, _ = self.contains(event)
        changed = check != self.hover
        if changed:
            self.set_hover_props(check)
        self.hover = check
        if changed and self.standalone:
            self.figure.canvas.draw()
        return changed


class ModeButton(MenuItem):
    def __init__(self, *args, **kwargs):
        # overwite the labelstr with 'Idle'
        super(ModeButton, self).__init__(*args, **kwargs)

        # overwrite the on_select function with on_select_rotation
        self.pressed = False

    def check_select(self, event, overwrite=False):
        if overwrite:
            return
        over, _ = self.contains(event)
        if not over:
            return
        if self.on_select is not None:
            self.on_select(self)
            self.pressed = not self.pressed
            if self.pressed:
                self.set_hover_props(True)
        return self.pressed

    def set_hover(self, event):
        """Check the hover status of MenuItem"""
        check, _ = self.contains(event)
        changed = check != self.hover
        if changed and not self.pressed:
            self.set_hover_props(check)
        self.hover = check
        return changed

    def _on_select_rotation(self):
        """Old function to rotate labelstrings."""
        if self.status == len(self.labelrotation) - 1:
            self.status = 0
        else:
            self.status += 1
        self.labelstr = self.labelrotation[self.status]
        self.label.set_text(self.labelstr)
        # print(f"You pressed {self.labelstr}")


class Menu:
    def __init__(self, ax, items=["Reset", "Write", "Set Points"]):
        # suppresscomposite
        self.ax = ax
        self.fig = self.ax.get_figure()
        self.fig.suppressComposite = True

        # add the menu items
        self.menuitems = {}
        self.add_items(items=items)

        # mpl_connect move to set hover stuff
        self.cid = self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems.values():
            draw = item.set_hover(event)
            if draw:
                self.fig.canvas.draw()
                break

    def add_items(self, items):
        coords = self.get_coords(len(items))
        for i, (s, c) in enumerate(zip(items, coords)):
            # on_select = lambda item: print(f"You pressed {item.labelstr}.")
            on_select = lambda item: True
            item = MenuItem(
                self.fig, (0, c[0]), 1, c[1] - c[0], labelstr=s, on_select=on_select
            )
            self.menuitems[s] = item
            self.ax.add_artist(item)

    def get_coords(self, no_items, gap_space=0.05):
        no_gaps = no_items - 1
        total_length = 1 - no_gaps * gap_space
        length_per = total_length / no_items
        coords = []
        for i in range(no_items):
            if i == 0:
                coords.append([0, length_per])
            else:
                coords.append(
                    [
                        coords[i - 1][1] + gap_space,
                        coords[i - 1][1] + gap_space + length_per,
                    ]
                )
        return coords


class StatusMenu(Menu):
    def __init__(self, ax):
        items = [
            "Lasso",
            "Rectangle",
            "Ellipse",
            "Polygon",
            "Path",
            "Bezier",
            "Idle",
            "Mode",
        ]
        # call the parent class to use its get_coords() method
        # the add_items method is overwritten
        super(StatusMenu, self).__init__(ax, items=items)

        # set the label rotation and the current status
        self.set_idle()

        # click notify event to change self.status and switch ModeButtons off
        self.cid = self.fig.canvas.mpl_connect("button_release_event", self.on_click)

    def set_idle(self):
        """Sets the idle status. Called at __init__ and
        when nothing is pressed."""
        self.status = "Idle"
        self.menuitems[self.status].set_hover_props(True)
        self.menuitems[self.status].pressed = True

    def on_click(self, event):
        draw = False
        for key, item in self.menuitems.items():
            draw = item.pressed
            if draw and key != self.status:
                print(f"Changing Mode to {key}")
                self.menuitems[self.status].set_hover_props(False)
                self.menuitems[self.status].pressed = False
                self.menuitems[self.status].check_select(event, overwrite=True)
                self.status = key
        else:
            if all([not item.pressed for item in self.menuitems.values()]):
                print(f"Changing Mode to Idle")
                self.set_idle()

    def add_items(self, items):
        coords = self.get_coords(len(items))
        for i, (s, c) in enumerate(zip(items, coords)):
            if s == "Mode":
                on_select = lambda item: None  # print(f"You pressed {item.labelstr}.")
                # overwrite props and hoverprops
                props = {
                    "labelcolor": "black",
                    "bgcolor": "orange",
                    "fontsize": 15,
                    "alpha": 1,
                }
                hoverprops = {
                    "labelcolor": "black",
                    "bgcolor": "orange",
                    "fontsize": 15,
                    "alpha": 1,
                }
                item = MenuItem(
                    self.fig,
                    (0, c[0]),
                    1,
                    c[1] - c[0],
                    labelstr=s,
                    on_select=on_select,
                    props=props,
                    hoverprops=hoverprops,
                )
                item.pressed = False
            else:
                # on_select = lambda item: print(f"You pressed {item.labelstr}.")
                on_select = lambda item: True
                # item = ModeButton(self.fig, self.ax, s, on_select=on_select)
                item = ModeButton(
                    self.fig, (0, c[0]), 1, c[1] - c[0], labelstr=s, on_select=on_select
                )
            self.menuitems[s] = item
            self.ax.add_artist(item)
