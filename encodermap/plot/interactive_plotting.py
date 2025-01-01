# -*- coding: utf-8 -*-
# encodermap/plot/interactive_plotting.py
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

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import getpass
import os
import platform
import re
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

# Third Party Imports
import ipywidgets as widgets
import numpy as np
from optional_imports import _optional_import
from pip._internal.operations.freeze import freeze

# Encodermap imports
from encodermap.autoencoder.autoencoder import (
    AngleDihedralCartesianEncoderMap,
    Autoencoder,
    DihedralEncoderMap,
    EncoderMap,
)
from encodermap.misc.misc import _datetime_windows_and_linux_compatible, _is_notebook
from encodermap.plot.plotting import (
    _plot_free_energy,
    get_histogram,
    plot_trajs_by_parameter,
    to_density,
    to_free_energy,
)
from encodermap.trajinfo.info_all import TrajEnsemble
from encodermap.trajinfo.info_single import SingleTraj


################################################################################
# Optional Imports
################################################################################


sns = _optional_import("seaborn")
md = _optional_import("mdtraj")
jinja2 = _optional_import("jinja2")
make_subplots = _optional_import("plotly", "subplots.make_subplots")
px = _optional_import("plotly", "express")
go = _optional_import("plotly", "graph_objects")
Image = _optional_import("PIL", "Image")
Canvas = _optional_import("ipycanvas", "Canvas")
hold_canvas = _optional_import("ipycanvas", "hold_canvas")
nv = _optional_import("nglview")
psutil = _optional_import("psutil")
display = _optional_import("IPython", "display.display")
Image = _optional_import("PIL", "Image")


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    from mdtraj import Topology, Trajectory

    # Encodermap imports
    from encodermap.autoencoder.autoencoder import AutoencoderClass


################################################################################
# Globals
################################################################################


__all__: list[str] = ["InteractivePlotting"]


# fmt: off
BAD_MODEBAR_BUTTONS = [
    "autoScale2d", "autoscale", "editInChartStudio", "editinchartstudio",
    "hoverCompareCartesian", "hovercompare", "lasso", "lasso2d", "orbitRotation",
    "orbitrotation", "pan", "pan2d", "pan3d", "reset", "resetCameraDefault3d",
    "resetCameraLastSave3d", "resetGeo", "resetSankeyGroup", "resetScale2d",
    "resetViewMapbox", "resetViews", "resetcameradefault", "resetcameralastsave",
    "resetsankeygroup", "resetscale", "resetview", "resetviews", "select",
    "select2d", "sendDataToCloud", "senddatatocloud", "tableRotation",
    "tablerotation", "toImage", "toggleHover", "toggleSpikelines", "togglehover",
    "togglespikelines", "toimage", "zoom", "zoom2d", "zoom3d", "zoomIn2d",
    "zoomInGeo", "zoomInMapbox", "zoomOut2d", "zoomOutGeo", "zoomOutMapbox",
    "zoomin", "zoomout",
]
# fmt: on


H5_INFO = """\
## Loading a HDF5 file (.h5) with EncoderMap

EncoderMap introduces a way of storing multiple trajectories (a `TrajectorEnsemble`) in a
single file. These files can be loaded via:

```python
import encodermap as em
trajs = em.TrajEnsemble.from_dataset('{{ h5_file }}')
```
"""


PATH_TEMPLATE = """\
# README for EncoderMap.InteractivePlotting generate

You just used EncoderMap's `InteractivePlotting` and saved protein conformations generated from a path in a low-dimensional representation of a {{ ensemble_type}}. The conformations were generated using a trained neural network autoencoder (EncoderMap's {{ autoencoder_class }} class) from {{ n_points }} {{ lowd_dim }}-dimensional coordinates. The {{ ensemble_type }} contained {{ n_top }} distinct protein topologies. From these topologies, the {{ chosen_top }} was chosen to build this cluster. Find the topological information in the `.pdb` file in this directory. Look at EncoderMap's documentation at https://ag-peter.github.io/encodermap/ to learn more about Trajectory Ensembles.

### The complete Ensemble is also present

If you want to get more information about the clustering you carried out, you can refer to these files:

### lowd.csv

This `.csv` file contains info about the complete ensemble this cluster was selected from. The columns are as follows:

| traj_num  | The number of the trajectory in the full dataset. This number is 0-based. If only one trajectory is loaded, its `trajectory number` might also be `None`. |
| --------- | ------------------------------------------------------------ |
| frame_num | The frame number. The trajectory number and frame number can be used to unmistakably identify frames in a trajectory ensemble. Frame numbers are also 0-based. |
| traj_file | Contains the trajectory data (file formats such as .xtc, .dcd, .h5). |
| top_file  | Contains the topology of the file (i.e. atom types, masses, residues) (file formats such as .pdb, .gro, .h5). Some trajectory files (.h5) might also contain the topology. In that case `trajectory file` and `topology` file are identical. |
| time      | The time of the frame. This can be used for time-based indexing of trajectories. EncoderMap offers the `SingleTraj.tsel[time]` accessor to distinguish it from frame-based indexing via `SingleTraj[frame]`. |
| x         | The x coordinate of the low-dimensional projection.          |
| y         | The y-coordinate of the low-dimensional projection.          |

### path.npy

This numpy array contains the (x, y)-coordinates of the low-dimensional path, that was used to generate the conformations.

### path.png

A nice render of the selected cluster.

### generated.pdb and generated.xtc

These files contain the topological (`.pdb`) and trajectory (`.xtc`)information to rebuild this path. Check out the function `encodermap.plot.plottinginteractive_path_visualization`, which can be used to display a animation of that path:

```python
import encodermap as em
import numpy as np
import pandas as pd

path = em.load("{{ xtc_file }}", "{{ pdb_file }}")
lowd = pd.read_csv("{{ csv_file }}")
path = np.load("{{ npy_file }}")

em.plot.interactive_path_visualization(
	traj,
	lowd,
	path,
)
```



## Rendering this document

If you don't like to view plain markdown files with a text-viewer there are many viewers available, that are able to render markdown nicely. I am currently using ghostwriter:

https://ghostwriter.kde.org/

If you want to create a pdf from this document you can try a combination of pandoc, latex and groff.

### HTML

```bash
pandoc {{filename}}.md -o {{filename}}.html
```

### Latex

```bash
pandoc {{filename}}.md -o {{filename}}.pdf
```

### Groff

```bash
pandoc {{filename}}.md -t ms -o {{filename}}.pdf
```

## Debug Info

```
encodermap.__version__ = {{encodermap_version}}
system_user = {{system_user}}
platform = {{platform}}
platform_release = {{platform_release}}
platform_version = {{platform_version}}
architecture = {{architecture}}
hostname = {{hostname}}
ip_address = {{ip_address}}
mac_address = {{mac_address}}
processor = {{processor}}
ram = {{ram}}
pip freeze = {{pip_freeze}}
```

"""


CLUSTER_TEMPLATE = """\
# README for EncoderMap.InteractivePlotting cluster

You just used EncoderMap's `InteractivePlotting` and saved a cluster. Here's some information about this cluster. The cluster was selected from a `TrajectoryEnsemble` containing {{ n_trajs }} trajectories, {{ n_frames }} frames and {{ n_top }} unique topologies. This cluster was assigned the number {{ cluster_num }}. The file {{ h5_file }} contains only {{ n_points }} frames, chosen as representatives for this cluster. This file can be loaded with EncoderMap's `TrajEnsemble.from_dataset('{{ h5_file }}')` method. Look at EncoderMap's documentation at https://ag-peter.github.io/encodermap/ to learn more about Trajectory Ensembles.

### The complete Ensemble is also present

If you want to get more information about the clustering you carried out, you can refer to these files:

### cluster_{{ cluster_num }}.csv

This `.csv` file contains info about the complete ensemble this cluster was selected from. The columns are as follows:

| traj_num   | The number of the trajectory in the full dataset. This number is 0-based. If only one trajectory is loaded, its `trajectory number` might also be `None`. |
| ---------- | ------------------------------------------------------------ |
| frame_num  | The frame number. The trajectory number and frame number can be used to unmistakably identify frames in a trajectory ensemble. Frame numbers are also 0-based. |
| traj_file  | Contains the trajectory data (file formats such as .xtc, .dcd, .h5). |
| top_file   | Contains the topology of the file (i.e. atom types, masses, residues) (file formats such as .pdb, .gro, .h5). Some trajectory files (.h5) might also contain the topology. In that case `trajectory file` and `topology` file are identical. |
| time       | The time of the frame. This can be used for time-based indexing of trajectories. EncoderMap offers the `SingleTraj.tsel[time]` accessor to distinguish it from frame-based indexing via `SingleTraj[frame]`. |
| x          | The x coordinate of the low-dimensional projection.          |
| y          | The y-coordinate of the low-dimensional projection.          |
| cluster_id | This column contains -1, which are points not included in a cluster (outliers). Cluster 1 is denoted by a 0 in this column. If multiple clusters have been selected this column can contain multiple integer values. For every subsequent cluster, the `cluster_id` is advanced by 1. |

### cluster_{{ cluster_num }}_selector.npy

This numpy array contains the (x, y)-coordinates of the selector, that was used to highlight the cluster. Be careful, this shape might not be convex, so using convex algortihms to find points inside this Polygon might not work.

### cluster_{{ cluster_num }}.png

A nice render of the selected cluster.

{{ h5_info }}

## Rendering this document

If you don't like to view plain markdown files with a text-viewer there are many viewers available, that are able to render markdown nicely. I am currently using ghostwriter:

https://ghostwriter.kde.org/

If you want to create a pdf from this document you can try a combination of pandoc, latex and groff.

### HTML

```bash
pandoc {{filename}}.md -o {{filename}}.html
```

### Latex

```bash
pandoc {{filename}}.md -o {{filename}}.pdf
```

### Groff

```bash
pandoc {{filename}}.md -t ms -o {{filename}}.pdf
```

## Debug Info

```
encodermap.__version__ = {{encodermap_version}}
system_user = {{system_user}}
platform = {{platform}}
platform_release = {{platform_release}}
platform_version = {{platform_version}}
architecture = {{architecture}}
hostname = {{hostname}}
ip_address = {{ip_address}}
mac_address = {{mac_address}}
processor = {{processor}}
ram = {{ram}}
pip freeze = {{pip_freeze}}

```
"""


################################################################################
# Utils
################################################################################


@contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


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


def render_image(view: nv.NGLWidget, filename: str) -> None:
    """Renders a nv.NGLWidget inside a thread.

    Args:
        view (nv.NGLWidget): The widget to be rendered.
        filename (str): The file to render to.

    """
    view.clear()
    im = view.render_image()
    while not im.value:
        time.sleep(0.1)
    with open(filename, "wb") as fh:
        fh.write(im.value.tobytes())


def plotly_freeform_to_path(path, n_points=100):
    # Third Party Imports
    from scipy.interpolate import interp1d

    verts = path.lstrip("M").split("L")
    verts = np.array([list(map(float, v.split(","))) for v in verts])
    verts = np.array(verts)
    distance = np.cumsum(
        np.sqrt(
            np.ediff1d(verts[:, 0], to_begin=0) ** 2
            + np.ediff1d(verts[:, 1], to_begin=0) ** 2
        )
    )
    distance = distance / distance[-1]
    fx, fy = interp1d(distance, verts[:, 0]), interp1d(distance, verts[:, 1])
    alpha = np.linspace(0, 1, n_points)
    path = np.vstack([fx(alpha), fy(alpha)]).T
    return path


################################################################################
# Interactive Plotting
################################################################################


class ProgressWidgetTqdmCompatible:
    """A jupyter widgtes `IntProgress` wrapper, that is compatible with tqdm calls.

    Uses a contextmanager to open and close the progress bar.

    """

    def __init__(
        self,
        container: widgets.GridspecLayout,
        empty: widgets.Output,
        total: int,
        description: str,
    ) -> None:
        """Instantiate the progress bar.

        Args:
            container (widgets.GridSpecLayout): An instance of a widgets.GridSpecLayouts
                class. The progress bar will be placed in row 7 (index 6) at columns
                2 through to the end (index 1:).
            empty (widgtes.Output): After the progress bar closes, this object
                will be placed at the position of the progress bar to clear it.
            total (int): The initial total to count to.
            description (str): The description of the progress bar.

        """
        self.container = container
        self.total = total
        self.empty = empty
        self.description = description
        self._calls: dict[str, dict[str, int]] = {}
        self.print = os.getenv("ENCODERMAP_PRINT_PROG_UPDATES", "False") == "True"

    def __enter__(self):
        self.progbar = widgets.IntProgress(
            value=0,
            min=0,
            max=self.total,
            step=1,
            description=self.description,
            layout={"width": "90%"},
        )
        self.container[6, 1:] = self.progbar
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container[6, 1:] = self.empty
        if self.print:
            print(self._calls)
        del self.progbar

    def debug_print(self) -> None:
        """Prints debug info."""
        print(f"WidgetProgbar {id(self)}")
        for function, data in self._calls.items():
            print(
                f"{function:<15} total: {data['total']:>3} n: {data['update_calls']:>3}"
            )
        print("\n")

    def update(self, n: int = 1, **kwargs) -> None:
        """Advances the progress bar by n.

        Args:
            n (int): How far to advance. Defaults to 1.

        """
        function = kwargs.pop("function", None)
        if function is not None:
            if function not in self._calls:
                self._calls[function] = {
                    "update_calls": 0,
                    "total": 0,
                }
        if not isinstance(n, int):
            self.progbar.value += 1
        else:
            self.progbar.value += n
        if function is not None:
            self._calls[function]["update_calls"] += 1

    def reset(self, total: int, **kwargs) -> None:
        """Resets the progress bar with a new total.

        Args:
            total (int): New total. It should be greater than old total.

        """
        assert total > self.total
        function = kwargs.pop("function", None)
        if function is not None:
            if function not in self._calls:
                self._calls[function] = {
                    "update_calls": 0,
                    "total": total - self.total,
                }
            else:
                self._calls[function]["total"] += total - self.total
        self.total = total
        self.progbar = widgets.IntProgress(
            value=self.progbar.value,
            min=0,
            max=total,
            step=1,
            description=self.description,
            layout={"width": "90%"},
        )
        self.container[6, 1:] = self.progbar


class InteractivePlotting:
    """EncoderMap's interactive plotting for jupyter notebooks.

    Instantiating this class will display an interactive display in your notebook.
    The display will look like this::

        ┌─────────────────────┐ ┌───────────┐
        │Display              │ │Top        │
        └─────────────────────┘ └───────────┘
        ┌─────────────┐ ┌───┐ ┌─────────────┐
        │             │ │   │ │             │
        │             │ │ T │ │             │
        │  Main       │ │ R │ │  Molecular  │
        │  Plotting   │ │ A │ │  Conform.   │
        │  Area       │ │ C │ │  Area       │
        │             │ │ E │ │             │
        │             │ │   │ │             │
        └─────────────┘ └───┘ └─────────────┘
        ┌───┐ ┌─────────────────────────────┐
        │   │ │Progress Bar                 │
        └───┘ └─────────────────────────────┘
        ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌───────────────────┐
        │C│ │G│ │S│ │D│ │Slider             │
        └─┘ └─┘ └─┘ └─┘ └───────────────────┘
        ┌────────────────┐  ┌───────────────┐
        │                │  │               │
        │ Data           │  │               │
        │ Overview       │  │               │
        │                │  │               │
        │                │  │               │
        └────────────────┘  └───────────────┘

    The components do the following:
        * Display:
            This part will display debug information.
        * Top (Top selector):
            Select which topology to use when creating new
            molecular conformations from the autoencoder network.
        * Main plotting area:
            In this area, a scatter plot will be displayed. The coordinates of
            the scatter plot will be taken from the low-dimensional projection
            of the trajectories. The data for this plotting area can be
            taken from different sources. See the `_lowd_parser` docstring
            for information on how the lowd data is selected. Clicking
            on a point in the scatter plot displays the conformation of that
            point.
        * TRACE:
            Displays the high-dimensinal data of selected points or clusters.
        * Molecular conformation area:
            Displays molecular conformations.
        * Progress Bar:
            Displays progress.
        * C (Cluster button):
            After selecting point in the main plotting area
            with the lasso tool, hit this button to display the molecular
            conformations of the selected cluster.
        * G (Generate Button):
            Switch to density using the density button.
            Then, you can draw a freeform path into the Main plotting area.
            Pressing the generate button will generate the appropriate molecular
            conformations. If your data has multiple conformations, you can choose
            which conformation to use for decoding with the top selector.
        * S (Save button):
            Writes either a cluster or generated path to your disk. Uses the
            main_path of the autoencoder (the same directory as the training
            data will be stored).
        * D (Density button):
            Switch the main plotting area to Density.
        * Slider:
            In scatter mode this slider defines how many structures to select
            from a cluster for representation in the molecular conformations
            window. In density mode, this slider defines how many points along
            the user-drawn path should be sampled.

    """

    _max_filepath_len: int = 50
    stride: int = 10
    _max_slider_len: int = 200
    _cluster_col: str = "_user_selected_points"
    _nbins: int = 50
    _cluster_method: Literal["stack", "join"] = "join"
    _help_url: str = "https://github.com/AG-Peter/encodermap"

    @classmethod
    def from_project(cls, project_name: Literal["linear_dimers"]):
        # Encodermap imports
        from encodermap import load_project

        trajs, autoencoder = load_project(
            project_name,
            traj=-1,
            load_autoencoder=True,
        )
        return cls(autoencoder=autoencoder, trajs=trajs)

    def __init__(
        self,
        autoencoder: Optional[AutoencoderClass] = None,
        trajs: Optional[Union[str, list[str], TrajEnsemble, SingleTraj]] = None,
        lowd_data: Optional[np.ndarray] = None,
        highd_data: Optional[np.ndarray] = None,
        align_string: str = "name CA",
        top: Optional[Union[str, list[str], Topology]] = None,
        ball_and_stick: bool = False,
        histogram_type: Union[None, Literal["free_energy", "density"]] = "free_energy",
        superpose: bool = True,
        ref_align_string: str = "name CA",
        base_traj: Optional[Trajectory] = None,
    ):
        """Instantiate the InteractivePlotting class.

        Note:
            It is recommended to assign an instance of this class to a variable
            to safe variables from garbage collection::

                sess = em.InteractivePlotting()

        Args:
            autoencoder (Optional[AutoencoderClass]): An instance of any of
                EncoderMap's autoencoder classes (`Autoencoder`, `EncoderMap`,
                `DihedralEncoderMap`, `AngleDihedralCartesianEncoderMap`).
            trajs (Optional[Union[str, list[str], TrajEnsemble, SingleTraj]]): The
                trajectory data to use this session. Molecular conformations are
                selected from these trajectories. Can be one of EncoderMap's
                trajectory data containers (`SingleTraj`, `TrajEnsemble`). Can
                also be a str or a list of str, that point to trajectory files
                (.xtc, .dcd, .h5, .pdb, .gro). Can also be None. In this case
                the `autoencoder` argument is expected to be a
                `AngleDihedralCartesianEncoderMap`, that is expected to contain
                the trajs. Defaults to None.
            lowd_data (Optional[np.ndarray]): The low-dimensional data to use
                for this session. If not provided low-dimensional data will be
                inferred from either `trajs` or `autoencoder`. Defaults to None.
            highd_data (Optional[np.ndarray]): The high-dimensional data to use
                for this session. If not provided high-dimensional data will be
                inferred from either `trajs` or `autoencoder`. Defaults to None.
            align_string (str): The alignment string to superimpose the
                structures of selected clusters. See
                https://mdtraj.org/1.9.4/atom_selection.html for info on how
                this string affects the selected atoms. Defaults to 'name CA'.
            top (Optional[Union[str, list[str], Topology]]): If trajs is a str,
                and a trajectory file format that does not have topological
                information (.xtc, .dcd), this argument will be used for topology.
                Can be a str (file) or an instance of MDTraj's Topology. Can also
                be a list of str, that matches the list of str in `trajs` with
                the appropriate topology files. If None is provided, the trajs
                argument is expected to be either `SingleTraj` or `TrajEnsemble`.
                Defaults to None.
            ball_and_stick (bool): Whether to represent the structures in ball and
                stick representation (True) or in cartoon representation (False).
                Defaults to False and cartoon representation.
            histogram_type (Union[None, Literal["free_energy", "density"]]): Decide
                how to style your histogram. If None, a straight histogram (count
                per bin) will be plotted. If 'density' a density will be plotted.
                If 'free_energy', the negative natural logartihm of the density
                will be plotted. Defaults to 'free_energy'.
            superpose (bool): Whether to superpose the clustered structures.
                Defaults to True.
            ref_align_string (str): If a `base_traj` is provided, this string will
                be used to select the atoms to align the clustering results against.
            base_traj (Optional[Trajectory]): If not None, this traj will be
                used to align the clustered frames against. Can be used to make
                all clusterings be consistent in their placement in the 3d space.

        """
        self.total = 0
        self.cluster_output = None
        self.path_output = None
        self.align_string = align_string
        self.top = top
        self.ball_and_stick = ball_and_stick
        self.histogram_type = histogram_type
        self.superpose = superpose
        self.ref_align_string = ref_align_string
        self.base_traj = base_traj
        self._username = os.getlogin()

        # set the layout
        self.layout = go.Layout(
            {
                "modebar_add": ["drawline", "drawopenpath", "eraseshape"],
                "autosize": True,
                "margin": {
                    "l": 0,
                    "r": 0,
                    "t": 0,
                    "b": 0,
                },
                "shapedefaults": {"editable": False},
            }
        )

        # apply nest_asyncio for saving images
        if _is_notebook():
            # Third Party Imports
            import nest_asyncio

            nest_asyncio.apply()

        # parse the complex arrangement of args
        self.autoencoder = autoencoder
        self.main_path = Path(".").resolve()
        if self.autoencoder is not None:
            if not self.autoencoder.read_only:
                self.main_path = Path(self.autoencoder.p.main_path)
        self.trajs = self._trajs_parser(autoencoder, trajs, top)
        self.highd = self._highd_parser(autoencoder, highd_data, self.trajs)
        self.lowd = self._lowd_parser(autoencoder, lowd_data, self.trajs)
        self.file_arr = []
        self.frame_arr = []
        for t in self.trajs:
            self.file_arr.extend([t.traj_file for i in range(t.n_frames)])
            self.frame_arr.append(t.id[:, 1])
        self.file_arr = np.array(self.file_arr)
        self.frame_arr = np.hstack(self.frame_arr)

        # put the data into self.trajs
        if self.trajs is not None:
            if self.highd is not None:
                if "highd" not in self.trajs._CVs:
                    self.trajs.load_CVs(self.highd, "highd")
            if "lowd" not in self.trajs._CVs:
                self.trajs.load_CVs(self.lowd, "lowd")

        # debugging stuff
        self._debug_main_path = str(self.main_path)
        if len(self._debug_main_path) > self._max_filepath_len:
            self._debug_main_path = (
                "/"
                + "/".join([i[0] for i in self.main_path.parts[1:-2]])
                + "/"
                + "/".join(self.main_path.parts[-2:])
            )

        # set up base images
        self._setup_histogram()
        self._setup_graph()

    def _fake_progress(self) -> None:  # pragma: no cover
        with ProgressWidgetTqdmCompatible(
            container=self.container,
            empty=self.progbar_empty,
            total=10,
            description="Testing...",
        ) as self.progbar:
            for i in range(10):
                time.sleep(0.3)
                self.progbar.update()

    def _setup_histogram(self):
        (
            self.xcenters,
            self.ycenters,
            self.xedges,
            self.yedges,
            self.H,
        ) = get_histogram(
            self.lowd[:, 0],
            self.lowd[:, 1],
            bins=self._nbins,
            transpose=True,
            return_edges=True,
        )
        self.D = to_density(self.H)
        self.F = to_free_energy(self.D).astype(str)

    def _trajs_parser(
        self,
        autoencoder: AutoencoderClass,
        trajs: Optional[Union[str, TrajEnsemble]] = None,
        top: Optional[Union[str, Topology]] = None,
    ) -> TrajEnsemble:
        """Parses the input trajs and chooses what trajs to use.

        The order of priority follows:
            1. The input `trajs` parameter supersedes everything. If `trajs`
                1.1. If an `AutoencoderClass` has been provided, the trajs are
                    checked, whether they conform to the expected input shape.
                2.2. If trajs is a str, rather than a `TrajEnsemble`, the argument
                    `top` is used to build a `TrajEnsemble` from this topology
                    and the `trajs`. Thus, `top` can be either str or md.Topology.
            2. If trajs is None, the `top` argument is not used and the
                `TrajEnsemble` of the provided `AngleDihedralCartesianEncoderMap`
                is used.

        Args:
            autoencoder (AutoencoderClass): The autoencoder.
            trajs (Optional[Union[str, TrajEnsemble]]): The trajs.
            top (Optional[Union[str, Topology]]): The topology.

        Returns:
            TrajEnsemble: The trajectory ensemble to use in this session.

        """
        if isinstance(trajs, str):
            if not Path(trajs).is_file():
                # Standard Library Imports
                import errno

                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), trajs)
            if isinstance(top, str):
                if not Path(top).is_file():
                    # Standard Library Imports
                    import errno

                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), top
                    )
            trajs = TrajEnsemble([trajs], [top])
        elif isinstance(trajs, (list, tuple)):
            trajs = TrajEnsemble(trajs, top)

        if isinstance(autoencoder, AngleDihedralCartesianEncoderMap):
            if trajs is None:
                return autoencoder.trajs
            else:
                for key, d in autoencoder.inp_CV_data.items():
                    assert d.shape[1:] == trajs.CVs[key].shape[1:], (
                        f"The shape of the CV `{key}` of the provided `trajs` "
                        f"{trajs.CVs[key].shape[1:]} does not match the shape of "
                        f"the train data of the provided `autoencoder` {d.shape[1:]}."
                    )
                else:
                    if isinstance(trajs, SingleTraj):
                        return trajs._gen_ensemble()
                    return trajs
        assert (
            trajs is not None
        ), f"Please provide a `TrajEnsemble` for the argument `trajs`."
        if isinstance(trajs, SingleTraj):
            return trajs._gen_ensemble()
        return trajs

    def _highd_parser(
        self,
        autoencoder: AutoencoderClass,
        highd: Optional[np.ndarray] = None,
        trajs: Optional[TrajEnsemble] = None,
    ) -> np.ndarray:
        """Selects which source of high-dimensional data to use.

        The order of priority follows:
            1. The provided `highd` np.ndarray.
                1.1 If an autoencoder has been provided, the high-dimensional
                    input data will be checked with the autoencoder's input shape.
            2. If no high-dimensional data has been provided (`highd=None`), the
                high-dimensional data from the provided `trajs` will be used.
            3. If the autoencoder is a `AngleDihedralCartesianEncoderMap`, the
                trajs of this autoencoder will be used.
            4. As a last resort, the autoencoder's `train_data` attribute will
                be used if the other datasources are not provided.

        Args:
            autoencoder (AutoencoderClass): The autoencoder.
            highd (Optional[np.ndarray]): The high dimensional data.
            trajs (Optional[Union[str, TrajEnsemble]]): The trajs.

        Returns:
            np.ndarray: The high-dimensional data to use in this session.


        """
        if (
            isinstance(autoencoder, AngleDihedralCartesianEncoderMap)
            or autoencoder.__class__.__name__ == "AngleDihedralCartesianEncoderMap"
        ):
            if highd is not None:
                assert isinstance(highd, np.ndarray), (
                    f"The argument `highd_data` only supports None or np.ndarray. You "
                    f"supplied {type(highd)}."
                )
                raise Exception(
                    f"Confirming the shape of input highd and the input shape "
                    f"of the autoencoder model is currently not implemented."
                )
            else:
                if trajs is not None:
                    if "central_dihedrals" not in trajs._CVs:
                        print(
                            f"The provided `trajs`, don't have any CVs loaded. I will "
                            f"try to use the input data of the provided autoencoder."
                        )
                    sparse, highd_data, CV_dict = autoencoder.get_train_data_from_trajs(
                        trajs,
                        autoencoder.p,
                    )
                    if sparse:
                        highd_data = [trajs.central_dihedrals]
                        if autoencoder.p.use_backbone_angles:
                            highd_data.insert(0, trajs.central_angles)
                        if autoencoder.p.use_sidechains:
                            highd_data.append(trajs.side_dihedrals)
                        return np.hstack(highd_data)
                    return highd_data
                else:
                    return autoencoder.train_data
        elif isinstance(
            autoencoder, (Autoencoder, EncoderMap, DihedralEncoderMap)
        ) or autoencoder.__class__.__name__ in [
            "Autoencoder",
            "EncoderMap",
            "DihedralEncoderMap",
        ]:
            if highd is not None:
                assert isinstance(highd, np.ndarray), (
                    f"The argument `highd_data` only supports None or np.ndarray. You "
                    f"supplied {type(highd)}."
                )
                assert highd.shape[-1] == autoencoder.train_data.shape[-1], (
                    f"The provided np.array in argument `highd_data` has shape {highd.shape}, "
                    f"but the autoencoder's `train_data` has shape {autoencoder.train_data.shape}."
                )
                return highd
            else:
                if trajs is not None:
                    if "highd" in trajs.CVs:
                        return trajs.highd
                return autoencoder.train_data
        elif autoencoder is None:
            if "highd" in trajs.CVs:
                return trajs.highd
            assert highd is not None, (
                f"Please provide a numpy array containing high-dimensional data "
                f"or load high-dimensional data into your trajs with `trajs.load_CVs`."
            )
            return highd
        else:
            raise TypeError(f"Unknown type for autoencoder: {type(autoencoder)}.")

    def _lowd_parser(self, autoencoder, lowd, trajs):
        if (
            isinstance(autoencoder, AngleDihedralCartesianEncoderMap)
            or autoencoder.__class__.__name__ == "AngleDihedralCartesianEncoderMap"
        ):
            if lowd is not None:
                assert isinstance(lowd, np.ndarray), (
                    f"The argument `lowd_data` only supports None or np.ndarray. You "
                    f"supplied {type(lowd)}."
                )
                return lowd
            else:
                if trajs is not None:
                    if "lowd" in trajs.CVs:
                        return trajs.lowd
                return autoencoder.encode(trajs)
        elif isinstance(
            autoencoder, (Autoencoder, EncoderMap, DihedralEncoderMap)
        ) or autoencoder.__class__.__name__ in [
            "Autoencoder",
            "EncoderMap",
            "DihedralEncoderMap",
        ]:
            if lowd is not None:
                assert isinstance(lowd, np.ndarray), (
                    f"The argument `lowd_data` only supports None or np.ndarray. You "
                    f"supplied {type(lowd)}."
                )
                return lowd
            else:
                if trajs is not None:
                    if "lowd" in trajs.CVs:
                        return trajs.lowd
                return autoencoder.encode(self.highd)
        elif autoencoder is None:
            if "lowd" in trajs.CVs:
                return trajs.lowd
            assert lowd is not None, (
                f"Please provide a numpy array containing low-dimensional data "
                f"or load low-dimensional data into your trajs with `trajs.load_CVs`."
            )
            return lowd
        else:
            raise TypeError(f"Unknown type for autoencoder: {type(autoencoder)}.")

    @cached_property
    def density(self) -> Any:
        if self.histogram_type is None:
            H = self.H
        else:
            if self.histogram_type == "density":
                H = self.D
            elif self.histogram_type == "free_energy":
                H = self.F
            else:
                raise TypeError(
                    f"Argument `histogram_type` needs to be either of None, "
                    f"'density' or 'free_energy'. You supplied {self.histogram_type}."
                )
        return go.Contour(
            x=self.xcenters,
            y=self.ycenters,
            z=H,
            name="",
            showlegend=False,
            showscale=False,
            visible=True,
            colorscale="Viridis",
        )

    @cached_property
    def scatter(self) -> Any:
        """go.Scattergl: The scatter plot using the low-dimensional data."""
        # Third Party Imports
        from scipy.interpolate import interp1d

        if hasattr(self, "trajs"):
            basenames = np.array(
                [traj.basename for traj in self.trajs for i in range(traj.n_frames)]
            )
            traj_nums, frame_nums = self.trajs.id.T
            customdata = np.stack(
                (
                    basenames,
                    traj_nums,
                    frame_nums,
                ),
                axis=-1,
            )
            hovertemplate = (
                "%{customdata[0]} (Traj %{customdata[1]}, "
                "Frame %{customdata[2]}): (%{x:.2f}, %{y:.2f})"
            )
        else:
            customdata = None
            hovertemplate = None

        # map the values to the same range
        values = self.F.copy().astype("float32").T
        values_ma = np.ma.masked_invalid(values)
        interp = interp1d(
            (np.min(values_ma), np.max(values_ma)), (0.0, 1.0), bounds_error=False
        )
        values = interp(values)

        # fill an array with the default color
        self.marker_colors = np.full((len(self.lowd),), fill_value=1.0)

        # set the marker colors into the marker_colors array
        for i, x_ind in enumerate(zip(self.xedges[:-1], self.xedges[1:])):
            for j, y_ind in enumerate(zip(self.yedges[:-1], self.yedges[1:])):
                point_ind = (
                    (x_ind[0] <= self.lowd[:, 0])
                    & (self.lowd[:, 0] < x_ind[1])
                    & (y_ind[0] <= self.lowd[:, 1])
                    & (self.lowd[:, 1] < y_ind[1])
                )
                H_value = values[i, j]
                if np.isnan(H_value):
                    continue
                self.marker_colors[point_ind] = H_value

        return go.Scattergl(
            mode="markers",
            x=self.lowd[:, 0],
            y=self.lowd[:, 1],
            visible=True,
            marker={
                "color": self.marker_colors,
                "colorscale": "Viridis",
                "size": 1,
                "line": {
                    "width": 0,
                },
            },
            # opacity=0.8,
            name="",
            customdata=customdata,
            hovertemplate=hovertemplate,
        )

    def generate(self, b):
        # clear the display
        self.display.outputs = []

        self.progbar_description = "Backmapping: "

        # some error
        if len(self.canvas_path) == 0:
            with self.display:
                print(f"First Draw a line onto the Density map and the hit 'Generate'.")
            return

        # clear the pandas area
        self.pandas_info_area.outputs = []

        # instantiate the progbar
        # display a message
        n_points = self.slider.value
        with self.display:
            print(f"Generating {n_points} points. Please stand by.")

        # set up progbar
        with ProgressWidgetTqdmCompatible(
            container=self.container,
            empty=self.progbar_empty,
            total=0,
            description=self.progbar_description,
        ) as self.progbar:

            # get the path
            self.path = self._canvas_path_in_data_coords()

            # generate
            if (
                isinstance(self.autoencoder, AngleDihedralCartesianEncoderMap)
                or self.autoencoder.__class__.__name__
                == "AngleDihedralCartesianEncoderMap"
            ):
                self.path_output = self.autoencoder.generate(
                    self.path, top=self.top_selector.value, progbar=self.progbar
                )
            else:
                # Encodermap imports
                from encodermap.misc.backmapping import mdtraj_backmapping

                dihedrals = self.autoencoder.generate(self.path).numpy()
                self.path_output = mdtraj_backmapping(
                    top=self.trajs[0].top_file,
                    dihedrals=dihedrals,
                    progbar=self.progbar,
                    omega=False,
                )

            self.display.outputs = []
            with self.display:
                print(f"Conformations generated.")

        # clear progbar
        self.progbar_description = ""

        # create the media widget
        self.media_widget = widgets.Play(
            value=0,
            min=0,
            max=n_points,
            step=1,
            disabled=False,
        )
        self.media_slider = widgets.IntSlider()
        widgets.jslink((self.media_widget, "value"), (self.media_slider, "value"))
        self.container[7, 4:] = widgets.HBox(
            [self.media_widget, self.media_slider], layout={"align-content": "center"}
        )

        # create the view
        view = nv.show_mdtraj(self.path_output)
        self.ngl_area.children = [view]
        if self.ball_and_stick:
            view.clear_representations()
            view.add_representation("ball+stick")
        self.view = view

        # switch to plotly
        self.path_anim_widget.data[1].x = self.path[:, 0]
        self.path_anim_widget.data[1].y = self.path[:, 1]
        self.path_anim_widget.data[2].x = [self.path[0, 0]]
        self.path_anim_widget.data[2].y = [self.path[1, 1]]
        self.container[2:6, :3] = widgets.Box(
            [self.path_anim_widget],
            layout=widgets.Layout(
                height="auto",
                width="auto",
            ),
        )

        # make the slider responsive
        self.media_slider.observe(self.advance_path, names="value")

    def advance_path(self, n):
        n = n["new"]
        self.view.frame = n
        self.path_anim_widget.data[2].x = [self.path[n, 0]]
        self.path_anim_widget.data[2].y = [self.path[n, 1]]

    def cluster(self, b):
        # clear the display
        self.display.outputs = []
        self.progbar_description = "Clustering: "

        # some error
        if self.selected_point_ids.size == 0:
            with self.display:
                print(
                    f"First select some points using the Lasso or Polygon tool "
                    f"and then click 'cluster'."
                )
            return

        # clear the pandas area
        self.pandas_info_area.outputs = []

        # instantiate the progbar
        with ProgressWidgetTqdmCompatible(
            container=self.container,
            empty=self.progbar_empty,
            total=0,
            description=self.progbar_description,
        ) as self.progbar:
            # read the slider
            n_points = self.slider.value

            # display a message
            with self.display:
                print(f"Clustering {n_points} points. Please stand by.")

            # clustering
            if self._cluster_col not in self.trajs.CVs:
                _ = np.full(self.trajs.n_frames, -1)
                try:
                    _[self.selected_point_ids] = 0
                except IndexError as e:
                    raise SystemExit(f"{self.selected_point_ids=}") from e
                self.trajs.load_CVs(_, self._cluster_col, override=True)
            else:
                _ = self.trajs.CVs[self._cluster_col]
                max_ = _.max()
                _[self.selected_point_ids] = max_ + 1
                self.trajs.load_CVs(_, self._cluster_col, override=True)
            self.selected_point_ids = np.array([]).astype(int)

            self.cluster_output = self.trajs.cluster(
                cluster_id=max(_),
                col=self._cluster_col,
                n_points=n_points,
            )

            if self._cluster_method == "join":
                self._cluster = self.cluster_output.join(
                    align_string=self.align_string,
                    superpose=self.superpose,
                    ref_align_string=self.ref_align_string,
                    base_traj=self.base_traj,
                    progbar=self.progbar,
                )

                # nglview
                total = 0
                for i, val in enumerate(self._cluster.values()):
                    val.center_coordinates()
                    for j, frame in enumerate(val):
                        if i == j == 0:
                            view = nv.show_mdtraj(frame, gui=False)
                        else:
                            view.add_trajectory(frame)
                        total += 1
                self.ngl_area.children = [view]
                if self.ball_and_stick:
                    view.clear_representations()
                    for i in range(total):
                        view.add_representation("ball+stick", component=i)
            else:
                self._cluster = self.cluster_output.stack(
                    align_string=self.align_string,
                    superpose=self.superpose,
                    ref_align_string=self.ref_align_string,
                    base_traj=self.base_traj,
                    progbar=self.progbar,
                )

                # nglview
                self._cluster.center_coordinates()
                view = nv.show_mdtraj(self._cluster, gui=False)
                self.ngl_area.children = [view]
                if self.ball_and_stick:
                    view.clear_representations()
                    view.add_representation("ball+stick")

            # trace
            if self.highd is not None:
                d = self.highd[self.trajs.CVs[self._cluster_col] == max(_)]
                self.trace_widget.data[0].z = d.T

            # save the image, because threading is complicated in IPython
            filename = Path("/tmp/tmp.png")
            lock = threading.Lock()
            with lock:
                thread = threading.Thread(
                    target=render_image,
                    args=(view, filename),
                )
                thread.daemon = True
                thread.start()

        # clear progbar
        self.progbar_description = ""

        # clear display
        self.display.outputs = []
        with self.display:
            print(f"Finished clustering.")

        self.ngl_area.children = [view]
        self.view = view

        # pandas
        with self.pandas_info_area:
            display(self.cluster_output.dash_summary())

    def save(self, b):
        if self.cluster_output is None and self.path_output is None:
            self.display.outputs = []
            with self.display:
                print(
                    "Please select a cluster or a path and hit 'Generate' or "
                    "'Cluster', before 'Save'."
                )
        else:  # path save
            if self.path_output is not None:
                try:
                    # Third Party Imports
                    import imageio
                    import moviepy
                except (ModuleNotFoundError, NameError):
                    self.display.outputs = []
                    with self.display:
                        print("Please install moviepy, imageio and ffmpeg")
                    return
                with ProgressWidgetTqdmCompatible(
                    container=self.container,
                    empty=self.progbar_empty,
                    total=5,
                    description="Saving..",
                ) as self.progbar:
                    fname = self._save_path_on_disk()
                    self.display.outputs = []
                    with self.display:
                        print(f"Path saved at {fname}")
                return fname
            else:  # cluster save
                with ProgressWidgetTqdmCompatible(
                    container=self.container,
                    empty=self.progbar_empty,
                    total=5,
                    description="Saving...",
                ) as self.progbar:
                    fname = self._save_cluster_on_disk()
                    self.display.outputs = []
                    with self.display:
                        print(f"Cluster saved at {fname}")
                return fname

    def _save_path_on_disk(self) -> Path:
        # Third Party Imports
        from nglview.contrib.movie import MovieMaker

        now = _datetime_windows_and_linux_compatible()
        output = self.main_path / f"generated_paths/{now}"
        output.mkdir(parents=True, exist_ok=True)

        # define some files
        # fmt: off
        xtc_file = output / f"generated.xtc"
        pdb_file = output / f"generated.pdb"
        npy_file = output / f"path.npy"
        mp4_file = output / f"animated_path.mp4"  # save the cluster as h5 ensemble
        png_file = output / f"path.png"  # save the cluster as h5 ensemble
        csv_file = output / "lowd.csv"            # A csv file for later plotting the lowd
        md_file = output / "README.md"            # A readme filled by jinja
        # fmt: on

        # save the path
        self.path_output.save_pdb(str(pdb_file))
        self.path_output.save_xtc(str(xtc_file))
        self.progbar.update()

        # save the path
        np.save(npy_file, self.path)
        self.progbar.update()

        # create an animation
        # with tempfile.TemporaryDirectory() as td:
        #     td = Path(td)
        #     mov = MovieMaker(
        #         view=self.view,
        #         download_folder=str(td),
        #         # perframe_hook=self.update,
        #         output="my.gif",
        #     )
        #     mov.make()
        #     print(list(td.glob("*")))
        self.progbar.update()

        # save the lowd as csv
        df = self.trajs.to_dataframe(CV=["lowd"])
        df["x"] = df.pop("LOWD FEATURE 0")
        df["y"] = df.pop("LOWD FEATURE 1")
        df.to_csv(csv_file)
        self.progbar.update()

        # save a png similar to cluster
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            _plot_free_energy(
                x=self.trajs.lowd[:, 0],
                y=self.trajs.lod[:, 1],
                cbar=True,
                colorbar_x=0.45,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            plot_trajs_by_parameter(
                self.trajs,
                "traj_num",
                type="scatter",
                show=False,
            ).data[0],
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=self.path[:, 0],
                y=self.path[:, 1],
                mode="lines",
                name="",
            )
        )
        fig.data[1].marker.colorscale = px.colors.get_colorscale("Viridis")

        fig.update_layout(
            {
                "width": 800,
                "height": 300,
                "xaxis1": {"title": "x in a.u."},
                "xaxis2": {"title": "x in a.u."},
                "yaxis1": {"title": "y in a.u."},
                "yaxis2": {"title": "y in a.u."},
                "autosize": True,
                "margin": {
                    "l": 0,
                    "r": 0,
                    "t": 0,
                    "b": 0,
                },
            },
        )
        fig.write_image(png_file, engine="kaleido", width=1500, height=500, scale=2)
        self.progbar.update()

        # save a README
        # Local Folder Imports
        from .._version import get_versions

        _ensemble_type = "single traj"
        if self.trajs.__class__.__name__ == "TrajEnsemble":
            if self.trajs.n_trajs > 1:
                _ensemble_type = "trajectory ensemble"

        info_dict = {
            "platform": platform.system(),
            "system_user": getpass.getuser(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": socket.gethostname(),
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "mac_address": ":".join(re.findall("..", "%012x" % uuid.getnode())),
            "processor": platform.processor(),
            "ram": str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB",
            "pip_freeze": "\n".join(list(freeze())),
            "n_frames": self.trajs.n_frames,
            "n_trajs": self.trajs.n_trajs,
            "n_top": len(self.trajs.top),
            "encodermap_version": get_versions()["version"],
            "filename": str(md_file.resolve()),
            "n_points": len(self.path),
            "ensemble_type": _ensemble_type,
            "csv_file": str(csv_file.resolve()),
            "pdb_file": str(pdb_file.resolve()),
            "xtc_file": str(xtc_file.resolve()),
            "npy_file": str(npy_file.resolve()),
            "autoencoder_class": self.autoencoder.__class__.__name__,
            "chosen_top": self.top_selector.options[self.top_selector.value],
            "lowd_dim": self.lowd.shape[1],
        }
        # assert _check_all_templates_defined(PATH_TEMPLATE, info_dict)
        template = jinja2.Template(PATH_TEMPLATE)
        readme_text = template.render(info_dict)
        md_file.write_text(readme_text)
        self.progbar.update()

        return output

    def _save_cluster_on_disk(self) -> Path:
        """Saves the cluster in self.cluster_output to disk.

        Also writes a README.md and puts images into a directory.

        """
        now = _datetime_windows_and_linux_compatible()
        output = self.main_path / f"clusters/{now}"
        output.mkdir(parents=True, exist_ok=True)
        cluster_num = self.trajs.CVs[self._cluster_col].max()

        # define some files
        # fmt: off
        h5_file = output / f"cluster_{cluster_num}.h5"  # save the cluster as h5 ensemble
        csv_file = output / f"cluster_{cluster_num}.csv"  # the complete ensemble as a pandas array
        md_file = output / "README.md"  # A readme filled by jinja
        png_name = output / f"cluster_{cluster_num}.png"  # A render of the cluster
        npy_file = output / f"cluster_{cluster_num}_selector.npy"  # The xs and ys of the selector
        # fmt: on

        # save the cluster
        self.cluster_output.save(h5_file)
        self.progbar.update()

        # save the pandas
        CVs = ["lowd", self._cluster_col]
        if isinstance(self.autoencoder, AngleDihedralCartesianEncoderMap):
            CVs.append("central_dihedrals")
            if self.autoencoder.p.use_backbone_angles:
                CVs.append("central_angles")
            if self.autoencoder.p.use_sidechains:
                CVs.append("side_dihedrals")
        df = self.trajs.to_dataframe(CV=CVs)
        df["cluster_id"] = df.pop(self._cluster_col.upper() + " FEATURE")
        df["x"] = df.pop("LOWD FEATURE 0")
        df["y"] = df.pop("LOWD FEATURE 1")
        df.to_csv(csv_file)
        self.progbar.update()

        # save the selector
        verts = np.vstack([self.selector.xs, self.selector.ys]).T
        np.save(npy_file, verts)
        self.progbar.update()

        # create a png
        # the png is already saved in /tmp.json
        with Image.open("/tmp/tmp.png") as im:
            im = np.array(im).copy()

        fig = make_subplots(rows=1, cols=3)
        fig.add_trace(
            _plot_free_energy(
                x=self.trajs.lowd[:, 0],
                y=self.trajs.lowd[:, 1],
            ),
            row=1,
            col=1,
        )
        with set_env(ENCODERMAP_SKIP_SCATTER_SIZE_CHECK="True"):
            fig.add_trace(
                plot_trajs_by_parameter(
                    self.trajs,
                    self._cluster_col,
                    type="scatter",
                    z_name_overwrite="cluster id",
                    show=False,
                ).data[0],
                row=1,
                col=2,
            )
        fig.add_trace(
            px.imshow(im).data[0],
            row=1,
            col=3,
        )

        fig.data[1].marker.colorscale = px.colors.get_colorscale("Viridis")

        fig.update_layout(
            {
                "width": 1000,
                "height": 300,
                "xaxis1": {"title": "x in a.u."},
                "xaxis2": {"title": "x in a.u."},
                "xaxis3": {
                    "showticklabels": False,
                },
                "yaxis1": {"title": "y in a.u."},
                "yaxis2": {"title": "y in a.u."},
                "yaxis3": {
                    "showticklabels": False,
                },
                "coloraxis_showscale": False,
                "autosize": True,
                "margin": {
                    "l": 0,
                    "r": 0,
                    "t": 0,
                    "b": 0,
                },
            },
        )
        fig.update_traces(
            dict(
                showscale=False,
                coloraxis=None,
            ),
            selector={"type": "heatmap"},
        )
        fig.write_image(png_name, engine="kaleido", width=1500, height=500, scale=2)
        self.progbar.update()

        # save a README
        # Local Folder Imports
        from .._version import get_versions

        info_dict = {
            "platform": platform.system(),
            "system_user": getpass.getuser(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": socket.gethostname(),
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "mac_address": ":".join(re.findall("..", "%012x" % uuid.getnode())),
            "processor": platform.processor(),
            "ram": str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB",
            "pip_freeze": "\n".join(list(freeze())),
            "h5_file": str(h5_file.resolve()),
            "n_frames": self.trajs.n_frames,
            "n_trajs": self.trajs.n_trajs,
            "n_top": len(self.trajs.top),
            "cluster_num": cluster_num,
            "h5_info": jinja2.Template(H5_INFO).render(
                {"h5_file": str(h5_file.resolve())}
            ),
            "encodermap_version": get_versions()["version"],
            "filename": str(md_file.resolve()),
            "csv_file": str(csv_file.resolve()),
        }
        # assert _check_all_templates_defined(CLUSTER_TEMPLATE, info_dict)
        template = jinja2.Template(CLUSTER_TEMPLATE)
        readme_text = template.render(
            info_dict,
        )
        md_file.write_text(readme_text)
        self.progbar.update()

        # cleanup
        del self.selector
        del self.cluster_output
        self.progbar.update()
        return output

    def scatter_on_click(self, trace, points, selector):
        # clear the display
        self.display.outputs = []
        self.pandas_info_area.outputs = []

        # get the id of the clicked point
        point_id = points.point_inds[0]

        # display a message
        try:
            with self.display:
                print(
                    f"Displaying conformation {point_id} for "
                    f"{self.file_arr[point_id]} {self.frame_arr[point_id]}"
                )
        except Exception as e:
            with self.display:
                print(
                    f"Can't display point {point_id} due to error: {e}. The "
                    f"shapes of the file and frame arrays are "
                    f"{self.file_arr.shape}, {self.frame_arr.shape}"
                )

        # color the main plot
        # c = self.base_colors.copy()
        s = self.base_sizes.copy()
        # c[point_id] = "#ff7f0e"
        s[point_id] = 20
        with self.figure_widget.batch_update():
            # self.scatter_data.marker.color = c
            self.scatter_data.marker.size = s

        # plot the trace
        if self.highd is not None:
            d = np.expand_dims(self.highd[point_id], 0)
            self.trace_widget.data[0].z = d.T

        # nglview
        frame = self.trajs.get_single_frame(point_id)
        traj = deepcopy(frame.traj)
        traj = traj.center_coordinates()
        view = nv.show_mdtraj(traj, gui=False)
        if self.ball_and_stick:
            view.clear_representations()
            view.add_representation("ball+stick")
        self.ngl_area.children = [view]
        self.view = view

        # pandas
        with self.pandas_info_area:
            display(frame.dash_summary())

    def on_select(self, trace, points, selector):
        self.display.outputs = []
        self.selected_point_ids = np.concatenate(
            [self.selected_point_ids, np.asarray(points.point_inds)]
        )
        self.selector = deepcopy(selector)
        # c = self.base_colors.copy()
        s = self.base_sizes.copy()
        # c[self.selected_point_ids] = "#2ca02c"
        s[self.selected_point_ids] = 15
        with self.figure_widget.batch_update():
            # self.scatter_data.marker.color = c
            self.scatter_data.marker.size = s
        with self.display:
            print(
                f"Selected {len(self.selected_point_ids)} points. Hit 'cluster' to view."
            )

    def switch_between_density_and_scatter(self, b):
        if self._graph == "scatter":
            self.container[2:6, :3] = widgets.Box(
                [self.canvas],
                layout=widgets.Layout(
                    height="auto",
                    width="auto",
                ),
            )
            self.slider.value = 100
            self.container[7, 4:] = self.slider
            # self.figure_widget.data[0].visible = False
            # self.figure_widget.data[1].visible = True
            self._graph = "density"
        else:
            self.container[2:6, :3] = widgets.Box(
                [self.figure_widget],
                layout=widgets.Layout(
                    height="auto",
                    width="auto",
                ),
            )
            self.slider.value = 10
            self.container[7, 4:] = self.slider
            # self.figure_widget.data[0].visible = True
            # self.figure_widget.data[1].visible = False
            self._graph = "scatter"

    def help(self, n):
        # Third Party Imports
        from IPython.display import Javascript

        out = widgets.Output()
        with out:
            display(Javascript(f'window.open("{self._help_url.tooltip}");'))

    def on_canvas_mouse_down(self, x, y):
        self.canvas_drawing = True
        self.canvas_position = (x, y)
        self.canvas_path = [self.canvas_position]

    def on_canvas_mouse_up(self, x, y):
        self.canvas_drawing = False

        self.display.outputs = []
        with self.display:
            if len(self.trajs.top) > 1:
                print(
                    f"Select a topology from the Dropdown menu and hit "
                    f"'Gnerate' to generate new molecular conformations."
                )
            else:
                print("Click 'Generate' to generate new molecular conformations")

    def on_canvas_mouse_move(self, x, y):
        if not self.canvas_drawing:
            return

        with hold_canvas():
            self.canvas.stroke_line(
                self.canvas_position[0], self.canvas_position[1], x, y
            )
            self.canvas_position = (x, y)

        self.canvas_path.append(self.canvas_position)

    def _canvas_path_in_data_coords(self) -> np.ndarray:
        """Returns the path coordinates in data coordinates.

        Returns:
            np.ndarray: An array of shape (n_points, 2) containing the
                data coordinates. [:, 0] are the x-coordinates and
                [:, 1] are the y-coordinates.

        """
        # Third Party Imports
        from scipy.interpolate import interp1d

        path = np.array(self.canvas_path)
        x = interp1d([0, 500], [self.lowd[:, 0].min(), self.lowd[:, 0].max()])
        x = x(path[:, 0])
        y = self.figure_widget.layout.yaxis.range
        y = interp1d([500, 0], [self.lowd[:, 1].min(), self.lowd[:, 1].max()])
        y = y(path[:, 1])
        verts = np.vstack([x, y]).T
        distance = np.cumsum(
            np.sqrt(
                np.ediff1d(verts[:, 0], to_begin=0) ** 2
                + np.ediff1d(verts[:, 1], to_begin=0) ** 2
            )
        )
        distance = distance / distance[-1]
        fx, fy = interp1d(distance, verts[:, 0]), interp1d(distance, verts[:, 1])
        alpha = np.linspace(0, 1, self.slider.value)
        path = np.vstack([fx(alpha), fy(alpha)]).T
        return path

    def _setup_graph(self):
        # text areas
        self._graph = "scatter"
        self.container = widgets.GridspecLayout(n_rows=10, n_columns=7, height="1000px")
        self.header = widgets.HTML(
            value=f"<h2>EncoderMap Dashboard for {self._username} in {self._debug_main_path}</h2>",
            layout=widgets.Layout(height="auto", width="auto"),
        )
        self.display = widgets.Output(
            layout=widgets.Layout(height="auto", width="auto")
        )
        with self.display:
            print(
                "Interact with the Scatter Plot to view molecular conformations. "
                "Select points with the lasso tool and click 'cluster' "
                "to generate a cluster. Switch to 'Density' to draw a Path and "
                "generate new conformations."
            )

        # the traj options
        options = [
            (f"{i + 1}: " + str(top).lstrip("<mdtraj.Topology with ").rstrip(">"), i)
            for i, top in enumerate(self.trajs.top)
        ]
        if len(self.trajs.top) == len(self.trajs.common_str):
            for top, sub_trajs in self.trajs.trajs_by_top.items():
                if len(sub_trajs.common_str) != 1:
                    break
            else:
                options = [(cs, i) for i, cs in enumerate(self.trajs.common_str)]
        self.top_selector = widgets.Dropdown(options=options, description="Top:")

        # some placeholders
        self.ngl_area = widgets.Box(layout=widgets.Layout(height="auto", width="auto"))
        self.progbar_empty = widgets.Output(
            layout=widgets.Layout(height="auto", width="auto")
        )
        self.pandas_all_area = widgets.Output(
            layout=widgets.Layout(height="auto", width="auto")
        )
        self.pandas_info_area = widgets.Output(
            layout=widgets.Layout(height="auto", width="auto")
        )

        # slider
        self.slider = widgets.IntSlider(
            value=10,
            min=1,
            max=self._max_slider_len,
            description="Size",
            continuous_update=False,
            layout=widgets.Layout(height="auto", width="auto"),
        )

        # buttons
        self.help_button = widgets.HTML(
            value=(
                f'<a href={self._help_url}><div class="lm-Widget jupyter-widgets '
                f'jupyter-button widget-button mod-info" style="height: 50%; '
                f"width: 100%; grid-area: widget007; margin: auto; margin-top: 25px; display: "
                f'flex; align-items: center; justify-content: center;">'
                f'<i class="fa fa-info"></i>Help</div></a>'
            )
        )
        self.cluster_button = widgets.Button(
            description="Cluster",
            icon="th",
            button_style="info",
            layout=widgets.Layout(height="auto", width="auto"),
            tooltip=(
                "After selecting points with the Lasso Tool, this button will "
                "display a subset of the selected point in the display area. Use "
                "the 'Size' slider to choose how many representative structures of "
                "the selected cluster you want to have displayed."
            ),
        )
        self.generate_button = widgets.Button(
            description="Generate",
            icon="bezier-curve",
            button_style="info",
            tooltip=(
                "Use the decoder part of the autoencoder to create new molecular "
                "conformations from a path, that you have drawn with the 'Draw "
                "open freeform' Tool. The 'Size' slider will choose how many "
                "conformations to create along the path."
            ),
            layout=widgets.Layout(height="auto", width="auto"),
        )
        self.save_button = widgets.Button(
            description="Save",
            icon="floppy-o",
            button_style="info",
            layout=widgets.Layout(height="auto", width="auto"),
        )
        self.density_button = widgets.Button(
            description="Density",
            icon="bar-chart",
            button_style="info",
            layout=widgets.Layout(height="auto", width="auto"),
            tooltip=("This button toggles between a density and a scatter plot."),
        )

        # plots
        self.heatmap = go.Heatmap(
            z=[],
            showlegend=False,
            showscale=False,
            colorscale="Viridis",
            hoverinfo="skip",
            name="",
            hovertemplate="",
        )

        # this array prepares the selection
        self.selected_point_ids = np.array([]).astype(int)

        # set up the canvas for drawing
        img = go.Figure(
            data=[self.density],
            layout={
                "margin": {
                    "t": 0,
                    "b": 0,
                    "l": 0,
                    "r": 0,
                },
                "yaxis_visible": True,
                "xaxis_visible": True,
            },
        )
        stream = img.to_image(format="png", width=500, height=500)
        background_image = widgets.Image(
            value=stream,
            format="png",
            width=500,
            height=500,
        )
        self.canvas = Canvas(width=500, height=500)
        self.canvas.draw_image(background_image)
        self.canvas_drawing = False
        self.canvas_position = None
        self.canvas_path = []
        self.canvas.on_mouse_down(self.on_canvas_mouse_down)
        self.canvas.on_mouse_move(self.on_canvas_mouse_move)
        self.canvas.on_mouse_up(self.on_canvas_mouse_up)
        self.canvas.stroke_style = "#749cb8"

        # main figure widget
        self.figure_widget = go.FigureWidget(
            data=[self.scatter],
            layout=self.layout,
        )
        self.scatter_data = self.figure_widget.data[0]
        self.base_colors = self.marker_colors
        self.scatter_data.marker.color = self.marker_colors
        self.base_sizes = np.array([8] * len(self.lowd))
        self.scatter_data.marker.size = self.base_sizes

        # the animation widget
        self.path_anim_widget = go.FigureWidget(
            data=[
                self.density,
                go.Scatter(
                    x=[0, 0],
                    y=[0, 0],
                    mode="lines",
                    hovertemplate="Generation Path (%{x:.2f}, %{y:.2f})",
                    showlegend=False,
                ),
                go.Scatter(
                    x=[0, 0],
                    y=[0, 0],
                    mode="markers",
                    marker_size=12,
                    marker_line_width=2,
                    hovertemplate="Current Path (%{x:.2f}, %{y:.2f})",
                    showlegend=False,
                ),
            ],
            layout=self.layout,
        )

        # the trace widget
        if self.highd is not None:
            self.trace_widget = go.FigureWidget(
                data=[self.heatmap],
                layout=go.Layout(
                    {
                        "width": 50,
                        "modebar_remove": BAD_MODEBAR_BUTTONS,
                        "yaxis_visible": False,
                        "xaxis_visible": False,
                        "title": "Trace",
                        "height": 500,
                        "margin": {
                            "t": 25,
                            "b": 75,
                            "l": 10,
                            "r": 10,
                        },
                    }
                ),
            )

        # responsiveness
        self.scatter_data.on_click(self.scatter_on_click)
        self.scatter_data.on_selection(self.on_select)
        self.cluster_button.on_click(self.cluster)
        self.density_button.on_click(self.switch_between_density_and_scatter)
        self.save_button.on_click(self.save)
        self.generate_button.on_click(self.generate)

        # add the elements to the grid
        self.container[0, :] = self.header
        self.container[1, :-1] = self.display
        self.container[1, -1] = self.top_selector
        self.container[2:6, :3] = widgets.Box(
            [self.figure_widget],
            layout=widgets.Layout(
                height="auto",
                width="auto",
            ),
        )
        if self.highd is not None:
            self.container[2:6, 3] = widgets.Box(
                [self.trace_widget],
                layout=widgets.Layout(
                    height="auto",
                    width="auto",
                ),
            )
        self.container[2:6, 4:] = self.ngl_area
        self.container[6, 1:] = self.progbar_empty
        self.container[6, 0] = self.help_button
        self.container[7, 0] = self.cluster_button
        self.container[7, 1] = self.generate_button
        self.container[7, 2] = self.save_button
        self.container[7, 3] = self.density_button
        self.container[7, 4:] = self.slider
        self.container[8:, :3] = self.pandas_all_area
        self.container[8:, 4:] = self.pandas_info_area
        with self.pandas_all_area:
            display(self.trajs.dash_summary())

        # self.container = widgets.VBox([
        #     self.header,
        #     self.display,
        #     self.figure_widget,
        # ])

        display(self.container)


# class InteractivePlottingDep:
#     """Class to open up an interactive plotting window.
#
#     Contains subclasses to handle user-clickable menus and selectors.
#
#     Attributes:
#         trajs (encodermap.TrajEnsemble): The trajs passed into this class.
#         fig (matplotlib.figure): The figure plotted onto. If ax is passed when
#             this class is instantiated, the parent figure will be fetched with
#             self.fig = self.ax.get_figure()
#         ax (matplotlib.axes): The axes where the lowd data of the trajs
#             is plotted on.
#         menu_ax (matplotlib.axes): The axes where the normal menu is plotted on.
#         status_menu_ax (matplotlib.axes): The axes on which the status menu is plotted on.
#         pts (matplotlib.collections.Collection): The points which are plotted. Based on some
#             other class variables, the color of this collection is adjusted.
#         statusmenu (encodermap.plot.utils.StatusMenu): The menu containing the
#             status buttons.
#         menu (encodermap.plot.utils.Menu): The menu containing the remaining buttons.
#         tool (encodermap.plot.utils.SelectFromCollection): The current active
#             tool used to select points. This can be lasso, polygon, etc...
#         mode (str): Current mode of the statusmenu.
#
#     """
#
#     def __init__(
#         self,
#         autoencoder,
#         trajs=None,
#         data=None,
#         ax=None,
#         align_string="name CA",
#         top=None,
#         hist=False,
#         scatter_kws={"s": 5},
#         ball_and_stick=False,
#         top_index=0,
#     ):
#         """Instantiate the InteractivePlotting class.
#
#         Args:
#             trajs (encodermap.TrajEnsemble): The trajs of which the lowd info
#                 should be plotted.
#             ax (matplotlib.axes, optional): On what axes to plot. If no axis is provided
#                 a new figure and axes will be created, defaults to None.
#
#         """
#         # the align string for the cluster dummy method
#         self.align_string = align_string
#         self.top = top
#         self.hist = hist
#         self.autoencoder = autoencoder
#         self.ball_and_stick = ball_and_stick
#         self.top_index = top_index
#
#         # scatter kws
#         self.scatter_kws = {**{"s": 80, "alpha": 0.5}, **scatter_kws}
#
#         # close all plots
#         plt.close("all")
#
#         # decide on fate of data
#         if data is None:
#             if hasattr(trajs, "lowd"):
#                 print("Using the attribute `lowd` of provided `trajs`")
#                 data = trajs.lowd
#             elif isinstance(trajs, (TrajEnsemble, SingleTraj)) and (
#                 isinstance(autoencoder, AngleDihedralCartesianEncoderMap)
#                 or autoencoder.__class__.__name__ == "AngleDihedralCartesianEncoderMap"
#             ):
#                 print(
#                     "Using the provided `autoencoder` and `trajs` to create a projection."
#                 )
#                 data = autoencoder.encode(trajs)
#             elif isinstance(data, np.ndarray) and hasattr(autoencoder, "encode"):
#                 print("Using the `encode` method of `autoencoder` with provided data.")
#                 if np.any(np.isnan(data)):
#                     # Third Party Imports
#                     import tensorflow as tf
#
#                     indices = np.stack(np.where(~np.isnan(data))).T.astype("int64")
#                     dense_shape = data.shape
#                     values = data[~np.isnan(data)].flatten().astype("float32")
#                     data = tf.sparse.SparseTensor(indices, values, dense_shape)
#                 data = autoencoder.encode(data)
#             elif hasattr(autoencoder, "encode"):
#                 print("Using the `train_data` attribute of `autoencoder`.")
#                 data = autoencoder.encode()
#             else:
#                 print("Mocking data with np.random")
#                 np.random.seed(19680801)
#                 data = np.random.rand(100, 2)
#         if data.shape[1] != 2:
#             print("Using provided `data` to call encoder.")
#             data = autoencoder.encode(data)
#         self.data = data
#
#         # see what traj has been provided
#         if trajs is None:
#             self.trajs = autoencoder.trajs
#         else:
#             if isinstance(trajs, str):
#                 self.trajs = SingleTraj(trajs, self.top, traj_num=0)._gen_ensemble()
#             elif isinstance(trajs, list):
#                 self.trajs = TrajEnsemble(trajs, self.top)
#             else:
#                 self.trajs = trajs
#
#         if isinstance(trajs, SingleTraj):
#             if "lowd" not in self.trajs.CVs:
#                 self.trajs.load_CV(self.data, attr_name="lowd")
#         else:
#             if "lowd" not in self.trajs.CVs:
#                 self.trajs.load_CVs(self.data, attr_name="lowd")
#
#         # decide what function to use to build clusters
#         # Decided against gen_dummy traj as get_cluster_frames works better with jinja2
#         self.cluster_building_fn = get_cluster_frames
#
#         # create fig and ax
#         if ax is None:
#             # create fig and ax
#             subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=True)
#             self.fig, self.ax = plt.subplots(
#                 1, 1, figsize=(10, 8)
#             )  # subplot_kw=subplot_kw)
#         else:
#             self.ax = ax
#             self.fig = self.ax.get_figure()
#
#         # add the axes to create the menus on
#         self.fig.subplots_adjust(left=0.3)
#         self.menu_ax = plt.axes([0.05, 0.1, 0.15, 0.35], facecolor="lightblue")
#         self.status_menu_ax = plt.axes(
#             [0.05, 0.49, 0.15, 0.35], facecolor="lightyellow"
#         )
#
#         # remove everything in these axes
#         self.menu_ax.axis("off")
#         self.status_menu_ax.axis("off")
#         self.tool = DummyTool()
#
#         # plot
#         self.pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], **self.scatter_kws)
#
#         # hist
#         if self.hist:
#             self.ax.hist2d(*data.T, bins=400, norm=mpl.colors.LogNorm())
#
#         # Check whether mouse enters drawing area
#         # Upon entering drawing area tools are initialized based on current mode
#         # Leave event currently serves no purpose
#         self.cid_ax_enter = self.fig.canvas.mpl_connect(
#             "axes_enter_event", self.on_enter_ax
#         )
#         self.cid_ax_leave = self.fig.canvas.mpl_connect(
#             "axes_leave_event", self.on_leave_ax
#         )
#
#         # chech button presses and compare them with the status of the menuitems
#         self.cid_on_click = self.fig.canvas.mpl_connect(
#             "button_release_event", self.on_click
#         )
#
#         # Instantiate Menu
#         self.statusmenu = StatusMenu(self.status_menu_ax)
#         self.menu = Menu(self.menu_ax)
#
#         # Show
#         plt.show()
#
#     def on_click(self, event):
#         """Decides whether the release event happened in the drawing area or the menu.
#
#         Args:
#             event (matplotlib.backend_bases.Event): The event provided by figure.canvas.connect().
#
#         """
#         if event.inaxes == self.ax:
#             self.on_click_tool(event)
#         else:
#             self.on_click_menu(event)
#
#     def on_enter_ax(self, event):
#         """Chosses the tool to use when self.ax is entered, based on current mode.
#
#         Args:
#             event (matplotlib.backend_bases.Event): The event provided by figure.canvas.connect().
#
#         """
#         # print('Axis is entered')
#         if event.inaxes is self.ax and self.mode == "Idle":
#             # reset point coloration
#             self.pts.set_color("C0")
#             self.tool.disconnect()
#         if event.inaxes is self.ax and self.mode != "Idle":
#             # statusmenu
#             for key, item in self.statusmenu.menuitems.items():
#                 if self.mode == key:
#                     method = getattr(self, key.lower())
#                     method()
#
#     def on_leave_ax(self, event):
#         """Disconnect the current tool."""
#         pass
#
#     def on_click_tool(self, event):
#         """Left here for convenience if some tools need a button release event."""
#         pass
#
#
#     ick_menu(self, event):
#         """Chooses the function to call based on what MenuItem was clicked.
#
#         Args:
#             event (matplotlib.backend_bases.Event): The event provided by figure.canvas.connect().
#
#         """
#         for key, item in self.menu.menuitems.items():
#             if item.check_select(event):
#                 method = getattr(self, key.lower().replace(" ", "_"))
#                 method()
#
#     def reset(self):
#         """Called when 'Reset' is pressed."""
#         if "user_selected_points" in self.trajs.CVs:
#             self.trajs._CVs.drop(labels="user_selected_points")
#         self.__init__(
#             self.trajs,
#             self.autoencoder,
#             self.data,
#             None,
#             self.align_string,
#             self.top,
#             self.hist,
#             self.scatter_kws,
#             self.ball_and_stick,
#         )
#
#     def write(self):
#         """Called when 'Write' is pressed."""
#         if self.mode == "Idle":
#             return
#         time = _datetime_windows_and_linux_compatible()
#         if self.mode == "Bezier" or self.mode == "Path":
#             os.makedirs(
#                 f"{self.autoencoder.p.main_path}/generated_paths/", exist_ok=True
#             )
#             fname = (
#                 f"{self.autoencoder.p.main_path}/generated_paths/generated_{time}.pdb"
#             )
#             with mda.Writer(fname) as w:
#                 for step in self.uni.trajectory:
#                     w.write(self.uni.atoms)
#             self.ax.set_title(
#                 f"Generated Path with {len(generated)} points saved at {fname}"
#             )
#         else:
#             if "user_selected_points" not in self.trajs.CVs:
#                 self.ax.set_title("First set the points before writing them to disk.")
#                 return
#             max_, fname = _unpack_cluster_info(
#                 self.trajs,
#                 self.autoencoder.p.main_path,
#                 self.tool,
#                 self.dummy_traj,
#                 self.align_string,
#             )
#             self.ax.set_title(f"Cluster {max_} saved at {fname}")
#
#     def set_points(self):
#         """Called when 'Set Points' is pressed."""
#         if self.mode == "Idle":
#             return
#         if self.mode != "Idle":
#             if "tool" not in self.__dict__.keys():
#                 self.ax.set_title(f"Tool {self.mode} not yet implemented.")
#                 return
#             else:
#                 indices = self.accept()
#         if self.mode == "Bezier" or self.mode == "Path":
#             if np.unique(self.path_points, axis=0).shape[0] != 200:
#                 self.ax.set_title(
#                     f"Tool {self.mode} returned not the requested unique points."
#                 )
#                 return
#             self.dummy_traj = self.autoencoder.generate(
#                 self.path_points, backend="mdanalysis", top=self.top_index
#             )
#             self.view = ngl.show_mdanalysis(self.dummy_traj)
#             if self.ball_and_stick:
#                 self.view.clear_representations()
#                 self.view.add_ball_and_stick()
#
#             self.ax.set_title(
#                 f"Generated Path with {len(self.dummy_traj.trajectory)} points is accessible as InteractivePlotting.view."
#             )
#             return
#
#         if indices is not None and self.mode != "Bezier" and self.mode != "Path":
#             self.ax.set_title(
#                 f"Currently working on rendering the cluster. I'll let you know, when I'm finished."
#             )
#             indices = np.asarray(indices)
#
#             # update user defined clustering
#             col = "user_selected_points"
#             if col not in self.trajs.CVs:
#                 _ = np.full(self.trajs.n_frames, -1)
#                 try:
#                     _[indices] = 0
#                 except IndexError as e:
#                     print(indices)
#                     raise SystemExit from e
#                 self.trajs.load_CVs(_, col)
#             else:
#                 _ = self.trajs.CVs[col]
#                 max_ = _.max()
#                 _[indices] = max_ + 1
#                 self.trajs.load_CVs(_, col)
#
#             # change coloration of self.pts
#             color_palette = sns.color_palette("Paired", self.trajs.CVs[col].max() + 1)
#             cluster_colors = [
#                 (*color_palette[x], 1) if x >= 0 else (0.5, 0.5, 0.5, 0.01)
#                 for x in self.trajs.CVs[col]
#             ]
#             self.pts.set_color(cluster_colors)
#
#             max_ = np.max(self.trajs.CVs[col])
#             self.view, self.dummy_traj = self.cluster_building_fn(
#                 self.trajs,
#                 max_,
#                 nglview=True,
#                 shorten=True,
#                 stack_atoms=True,
#                 col=col,
#                 align_string=self.align_string,
#                 ball_and_stick=self.ball_and_stick,
#             )
#             if self.ball_and_stick:
#                 for i in range(len(self.dummy_traj)):
#                     self.view.clear_representations(component=i)
#                     self.view.add_ball_and_stick(component=i)
#             self.ax.set_title(
#                 f"Cluster {max_} is accessible as InteractivePlotting.view."
#             )
#
#     def render_move(self):
#         pass
#
#     def lasso(self):
#         self.tool = SelectFromCollection(self.ax, self.pts)
#
#     def rectangle(self):
#         self.tool = SelectFromCollection(self.ax, self.pts, selector=RectangleSelector)
#
#     def ellipse(self):
#         print("Ellipse not yet implemented")
#
#     def polygon(self):
#         textstr = "\n".join(
#             (
#                 "Select points in the figure by enclosing them within a polygon.",
#                 # Press the 'esc' key to start a new polygon.
#                 "Try holding the 'shift' key to move all of the vertices.",
#                 "Try holding the 'ctrl' key to move a single vertex.",
#             )
#         )
#
#         # these are matplotlib.patch.Patch properties
#         props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#
#         # place a text box in upper left in axes coords
#         self.manual_text = self.ax.text(
#             0.05,
#             0.95,
#             textstr,
#             transform=self.ax.transAxes,
#             fontsize=6,
#             verticalalignment="top",
#             bbox=props,
#         )
#         self.tool = SelectFromCollection(self.ax, self.pts, selector=PolygonSelector)
#
#     def path(self):
#         pass
#
#     def bezier(self):
#         line = Line2D([], [], ls="--", c="#666666", marker="x", mew=2, mec="#204a87")
#         self.ax.add_line(line)
#         self.tool = BezierBuilder(line, self.ax)
#
#     def accept(self):
#         if "manual_text" in self.__dict__.keys():
#             self.manual_text.set_visible(False)
#             del self.manual_text
#         if self.mode == "Bezier":
#             self.path_points = copy.deepcopy(self.tool.ind)
#         selected_indices = self.tool.ind
#         self.tool.disconnect()
#         return selected_indices
#
#     @property
#     def cluster_zoomed(self):
#         col = "user_selected_points"
#         if not col in self.trajs.df.keys():
#             return
#         max_ = np.max(self.trajs.df[col])
#         _ = plot_cluster_zoomed(self.trajs, max_, col=col)
#         return _
#
#     @property
#     def mode(self):
#         return self.statusmenu.status
