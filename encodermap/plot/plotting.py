# -*- coding: utf-8 -*-
# encodermap/plot/plotting.py
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
"""Convenience functions for Plotting.

Todo:
    * Find a way to use interactive plotting with less points but still cluster everything.

"""

##############################################################################
# Imports
##############################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os
import shutil
import subprocess
import time
from functools import partial
from typing import TYPE_CHECKING, Literal, Optional, Union

# Third Party Imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance
from IPython.display import display
from ipywidgets import widgets

# Local Folder Imports
from ..encodermap_tf1.misc import periodic_distance_np, sigmoid
from ..misc.clustering import gen_dummy_traj, rmsd_centroid_of_cluster


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    import plotly.express as px
    import plotly.graph_objs as go

    # Local Folder Imports
    from ..trajinfo.info_single import SingleTraj


################################################################################
# Optional Imports
################################################################################


# Third Party Imports
from optional_imports import _optional_import


md = _optional_import("mdtraj")
nv = _optional_import("nglview")
mda = _optional_import("MDAnalysis")
go = _optional_import("plotly", "graph_objects")
px = _optional_import("plotly", "express")


##############################################################################
# Globals
##############################################################################


__all__ = [
    "distance_histogram",
    "distance_histogram_interactive",
    "raw_data_plot",
    "interactive_path_visualization",
    "ramachandran_plot",
    "dssp_plot",
    "end2end_plot",
    "ball_and_stick_plot",
]


##############################################################################
# Utilities
##############################################################################


def euclidean_in_periodic(periodicity):
    def metric(u, v):
        return np.linalg.norm(u - v)

    return metric


def get_free_energy(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: int = 100,
    kT: float = 1.0,
    avoid_zero_count: bool = False,
    minener_zero: bool = False,
    transpose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, xedges, yedges = np.histogram2d(x=x, y=y, bins=bins, weights=weights)
    xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
    ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)

    if avoid_zero_count:
        H = np.maximum(H, np.min(H[H.nonzero()]))

    if transpose:
        H = H.T

    # to density
    H = H / float(H.sum())

    # to free energy
    F = np.inf * np.ones(shape=H.shape)
    nonzero = H.nonzero()
    F[nonzero] = -np.log(H[nonzero])
    if minener_zero:
        F[nonzero] -= np.min(F[nonzero])
    F = F * kT

    return xcenters, ycenters, F


def go_plot_free_energy(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: int = 100,
    kT: float = 1.0,
    avoid_zero_count: bool = False,
    minener_zero: bool = False,
    transpose: bool = True,
) -> go.Contour:
    X, Y, Z = get_free_energy(
        x=x,
        y=y,
        weights=weights,
        bins=bins,
        kT=kT,
        avoid_zero_count=avoid_zero_count,
        minener_zero=minener_zero,
        transpose=transpose,
    )
    trace = go.Contour(
        x=X,
        y=Y,
        z=Z,
        name="Lowd projection",
        showscale=False,
        hoverinfo="none",
        # histfunc="count",
    )
    return trace


def interactive_path_visualization(
    lowd: np.ndarray,
    path: np.ndarray,
    traj: Union["md.Trajectory", mda.Universe],
    representation: Literal["ball+stick", "cartoon"] = "cartoon",
) -> None:
    # define the widgets
    path_progression = widgets.IntSlider(
        value=0.0,
        min=0.0,
        max=len(path),
        step=1.0,
        description="Path position",
        continuous_update=False,
        tooltip="Slide to select generated structure along path",
    )
    playbutton = widgets.Button(
        description="Play",
        icon="play",
        tooltip="Click to play animation",
    )

    # define the traces
    trace1 = go_plot_free_energy(*lowd.T, transpose=True)
    trace2 = go.Scatter(
        mode="lines",
        x=path[:, 0],
        y=path[:, 1],
        name="Path",
    )
    trace3 = go.Scatter(
        mode="markers",
        marker={"size": 10},
        x=[path[0, 0]],
        y=[path[0, 1]],
        name="Current path pos.",
    )

    # create a figure widget
    g = go.FigureWidget(
        data=[trace1, trace2, trace3],
        layout=go.Layout(
            {
                "height": 700,
                "width": 700,
            }
        ),
    )

    # add the nglview object to the container
    if isinstance(traj, md.Trajectory):
        assert traj.n_frames == len(path)
        view = nv.show_mdtraj(traj)
    elif isinstance(traj, mda.Universe):
        assert len(traj.trajectory) == len(path)
        view = nv.show_mdanalysis(traj)
    if representation == "ball+stick":
        view.clear_representations()
        view.add_ball_and_stick()
    view.center()

    # combine all widgets into a container
    container = widgets.VBox(
        [
            widgets.HBox([path_progression, playbutton]),
            widgets.HBox([g, view]),
        ]
    )

    # display that container
    display(container)

    # some callbacks
    def response(path_pos):
        with g.batch_update():
            g.data[2].x = [path[path_pos["new"], 0]]
            g.data[2].y = [path[path_pos["new"], 1]]
        view.frame = path_pos["new"]

    def on_playbutton_clicked(val):
        if path_progression.value == len(path):
            for i in range(len(path), 0, -1):
                path_progression.value = i
                time.sleep(0.1)
        else:
            for i in range(path_progression.value, len(path)):
                path_progression.value = i
                time.sleep(0.1)

    path_progression.observe(response, names="value")
    playbutton.on_click(on_playbutton_clicked)


def distance_histogram_interactive(
    data: Union[np.ndarray, pd.DataFrame],
    periodicity: float,
    low_d_max: int = 5,
    n_values: int = 1000,
    bins: Union[Literal["auto"], int] = "auto",
    initial_guess: Optional[tuple[float, ...]] = None,
    renderer: Optional[Literal["colab", "plotly_mimetype+notebook"]] = None,
) -> None:  # pragma: no cover
    # Third Party Imports
    import numpy as np
    import plotly.graph_objects as go
    from IPython.display import display
    from ipywidgets import widgets
    from plotly.subplots import make_subplots

    # decide the renderer
    if renderer is None:
        try:
            # Third Party Imports
            from google.colab import data_table

            renderer = "colab"
        except ModuleNotFoundError:
            renderer = "plotly_mimetype+notebook"

    # some helper functions
    def my_ceil(a, precision=0):
        return np.round(a + 0.5 * 10 ** (-precision), precision)

    def sigmoid(r, sigma=1, a=1, b=1):
        return 1 - (1 + (2 ** (a / b) - 1) * (r / sigma) ** a) ** (-b / a)

    def add_shapes(edges_h, edges_l, high_d_max):
        shapes = []
        for i, (e_h, e_l) in enumerate(zip(edges_h, edges_l)):
            if e_l == edges_l[-1]:
                continue
            shape = {
                "type": "line",
                "xref": "x2",
                "yref": "y2",
                "x0": e_l,
                "y0": 0,
                "x1": e_h * high_d_max,
                "y1": 1,
                "line": {"color": "black", "width": 1},
            }
            shapes.append(shape)
        return shapes

    # get the distances while accounting for periodicity
    vecs = periodic_distance_np(
        np.expand_dims(data, axis=1), np.expand_dims(data, axis=0), periodicity
    )
    dists = np.linalg.norm(vecs, axis=2)
    while True:
        try:
            dists = np.linalg.norm(dists, axis=2)
        except np.AxisError:
            break
    dists = dists.reshape(-1)
    high_d_max = np.max(dists)

    # use the initial guess or default values
    if initial_guess is None:
        initial_guess = (4.5, 12, 6, 1, 2, 6)

    # instantiate the sliders
    lowd_sigma_slider = widgets.FloatSlider(
        value=initial_guess[3],
        min=0.1,
        max=my_ceil(low_d_max, 1),
        step=0.1,
        description="lowd sigma",
        continuous_udpate=True,
    )
    lowd_a_slider = widgets.FloatSlider(
        value=initial_guess[4],
        min=0.1,
        max=12.0,
        step=0.1,
        description="lowd a",
        continuous_udpate=True,
    )
    lowd_b_slider = widgets.FloatSlider(
        value=initial_guess[5],
        min=0.1,
        max=12.0,
        step=0.1,
        description="lowd b",
        continuous_udpate=True,
    )
    highd_sigma_slider = widgets.FloatSlider(
        value=initial_guess[0],
        min=0.1,
        max=my_ceil(np.max(dists), 1),
        step=0.1,
        description="highd sigma",
        continuous_udpate=True,
    )
    highd_a_slider = widgets.FloatSlider(
        value=initial_guess[1],
        min=0.1,
        max=12.0,
        step=0.1,
        description="highd a",
        continuous_udpate=True,
    )
    highd_b_slider = widgets.FloatSlider(
        value=initial_guess[2],
        min=0.1,
        max=12.0,
        step=0.1,
        description="highd b",
        continuous_udpate=True,
    )

    # histogram
    H, edges = np.histogram(dists, bins=bins, density=True)
    H *= 1 / max(H)
    centers = np.mean(np.vstack([edges[0:-1], edges[1:]]), axis=0)

    # highd sigmoid
    x_h = np.linspace(0, max(dists), n_values)
    highd_data = {
        "sigma": highd_sigma_slider.value,
        "a": highd_a_slider.value,
        "b": highd_b_slider.value,
    }
    y_h = sigmoid(x_h, **highd_data)
    edges_h = sigmoid(edges, **highd_data)

    # diff and norm
    dy = np.diff(y_h)
    dy_norm = dy / max(dy)

    # lowd sigmoid
    x_l = np.linspace(0, low_d_max, n_values)
    lowd_data = {
        "sigma": lowd_sigma_slider.value,
        "a": lowd_a_slider.value,
        "b": lowd_b_slider.value,
    }
    y_l = sigmoid(x_l, **lowd_data)
    lowd_idx_match = np.argmin(
        np.abs(np.expand_dims(edges_h, axis=1) - np.expand_dims(y_l, axis=0)), axis=1
    )
    edges_l = x_l[lowd_idx_match]

    # initial subplot with two traces
    fig = make_subplots(rows=3, cols=1, subplot_titles=["highd", "scaling", "lowd"])

    # add the bar
    fig.add_trace(
        go.Bar(
            x=centers,
            y=H,
            name="highd dists",
            xaxis="x1",
            yaxis="y1",
            marker_color="blue",
            marker_opacity=0.5,
        )
    )

    # add the lowd sigmoid
    fig.add_trace(
        go.Scatter(
            x=x_l,
            y=y_l,
            mode="lines",
            name="lowd sigmoid",
            line_color="orange",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # add connections lines
    shapes = add_shapes(edges, edges_l, high_d_max)
    fig.update_layout(shapes=shapes)

    # add the title
    fig.update_layout(
        height=800,
        width=600,
        title={
            "text": "Interact with the plot to select sigmoid parameters",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "middle",
        },
    )

    # add the highd sigmoids to a second axis
    fig.add_trace(
        go.Scatter(
            x=x_h,
            y=y_h,
            name="sigmoid",
            line_color="orange",
            mode="lines",
            xaxis="x4",
            yaxis="y4",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_h,
            y=dy_norm,
            name="diff sigmoid",
            line_color="green",
            mode="lines",
            xaxis="x4",
            yaxis="y4",
        ),
        row=1,
        col=1,
    )

    # some adjustmentns for xaxis3
    fig.update_layout(
        xaxis1=dict(
            title="highd distance",
            showgrid=True,
        ),
        xaxis2=dict(
            showticklabels=False,
        ),
        xaxis3=dict(
            title="lowd distance",
        ),
        xaxis4=dict(
            anchor="free",
            overlaying="x1",
            side="right",
            position=0.0,
            showticklabels=False,
            showgrid=False,
        ),
        yaxis2=dict(
            showticklabels=False,
        ),
        yaxis4=dict(
            anchor="free",
            overlaying="y1",
            side="right",
            position=0.0,
            showticklabels=False,
            showgrid=False,
            range=[0, 1],
            autorange=False,
        ),
        bargap=0,
    )

    # make the figure responsive

    # create a figure widget
    g = go.FigureWidget(fig)
    # print(g["layout"]["shapes"])
    lowd_sigmoid_trace_index = [
        trace["name"] == "lowd sigmoid" for trace in g["data"]
    ].index(True)
    highd_sigmoid_trace_index = [
        trace["name"] == "sigmoid" for trace in g["data"]
    ].index(True)
    diff_sigmoid_trace_index = [
        trace["name"] == "diff sigmoid" for trace in g["data"]
    ].index(True)
    object_mapping = {
        "lowd sigma": {"update_data": [lowd_sigmoid_trace_index], "keyword": "sigma"},
        "lowd a": {"update_data": [lowd_sigmoid_trace_index], "keyword": "a"},
        "lowd b": {"update_data": [lowd_sigmoid_trace_index], "keyword": "b"},
        "highd sigma": {
            "update_data": [highd_sigmoid_trace_index, diff_sigmoid_trace_index],
            "keyword": "sigma",
        },
        "highd a": {
            "update_data": [highd_sigmoid_trace_index, diff_sigmoid_trace_index],
            "keyword": "a",
        },
        "highd b": {
            "update_data": [highd_sigmoid_trace_index, diff_sigmoid_trace_index],
            "keyword": "b",
        },
    }

    # define the response function
    def response(change):
        nonlocal highd_data
        nonlocal lowd_data
        key = change["owner"].description
        indices = object_mapping[key]["update_data"]
        kwarg = object_mapping[key]["keyword"]
        if "highd" in key:
            highd_data |= {kwarg: change["new"]}
        else:
            lowd_data |= {kwarg: change["new"]}
        y_h = sigmoid(x_h, **highd_data)
        y_l = sigmoid(x_l, **lowd_data)
        dy = np.diff(y_h)
        dy_norm = dy / max(dy)
        edges_h = sigmoid(edges, **highd_data)
        lowd_idx_match = np.argmin(
            np.abs(np.expand_dims(edges_h, axis=1) - np.expand_dims(y_l, axis=0)),
            axis=1,
        )
        edges_l = x_l[lowd_idx_match]
        shapes = add_shapes(edges, edges_l, high_d_max)
        with g.batch_update():
            g.data[highd_sigmoid_trace_index].y = y_h
            g.data[diff_sigmoid_trace_index].y = dy_norm
            g.data[lowd_sigmoid_trace_index].y = y_l
            g.layout["shapes"] = shapes

    # observe the widgets
    lowd_sigma_slider.observe(response, names="value")
    lowd_a_slider.observe(response, names="value")
    lowd_b_slider.observe(response, names="value")
    highd_sigma_slider.observe(response, names="value")
    highd_a_slider.observe(response, names="value")
    highd_b_slider.observe(response, names="value")

    # create containers
    lowd_container = widgets.HBox(
        children=[lowd_sigma_slider, lowd_a_slider, lowd_b_slider]
    )
    highd_container = widgets.HBox(
        children=[highd_sigma_slider, highd_a_slider, highd_b_slider]
    )

    # start the app
    return widgets.VBox([lowd_container, highd_container, g])


def distance_histogram(
    data: np.ndarray,
    periodicity: float,
    sigmoid_parameters: tuple[float, float, float],
    axes: Optional[plt.Axes] = None,
    low_d_max: int = 5,
    bins: Union[Literal["auto"], int] = "auto",
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:  # pragma: no cover
    """Plots the histogram of all pairwise distances in the data.

    It also shows the sigmoid function and its normalized derivative.

    Args:
        data (np.ndarray): 2-dimensional numpy array. Columns should iterate
            over the dimensions of the datapoints, i.e. the dimensionality
            of the data. The rows should iterate over datapoints.
        periodicity (float): Periodicity of the data. Use float("inf")
            for non-periodic data.
        sigmoid_parameters (tuple): Tuple of sketchmap sigmoid parameters
            in shape (sigma, a, b).
        axes (Union[np.ndarray, None], optional): A numpy array of two
            matplotlib.axes objects or None. If None is provided, the axes will
            be created. Defaults to None.
        low_d_max (int, optional): Upper limit for plotting the low_d sigmoid.
            Defaults to 5.
        bins (Union[str, int], optional): Number of bins for histogram.
            Use 'auto' to let matplotlib decide how many bins to use. Defaults to 'auto'.

    Returns:
        tuple: A tuple containing the following:
            plt.axes: A matplotlib.pyplot axis used to plot the high-d distance
                sigmoid.
            plt.axes: A matplotlib.pyplot axis used to plot the high-d distance
                histogram (a twinx of the first axis).
            plt.axes: A matplotlib.pyplot axis used to plot the lowd sigmoid.

    """
    vecs = periodic_distance_np(
        np.expand_dims(data, axis=1), np.expand_dims(data, axis=0), periodicity
    )
    dists = np.linalg.norm(vecs, axis=2)
    while True:
        try:
            dists = np.linalg.norm(dists, axis=2)
        except np.AxisError:
            break
    dists = dists.reshape(-1)

    if axes is None:
        fig, axes = plt.subplots(2)
    axe2 = axes[0].twinx()
    counts, edges, patches = axe2.hist(
        dists, bins=bins, density=True, edgecolor="black"
    )
    x = np.linspace(0, max(dists), 1000)
    y = sigmoid(x, *sigmoid_parameters[:3])
    edges_sig = sigmoid(edges, *sigmoid_parameters[:3])
    dy = np.diff(y)
    dy_norm = dy / max(dy)
    axes[0].plot(x, y, color="C1", label="sigmoid")
    axes[0].plot(x[:-1], dy_norm, color="C2", label="diff sigmoid")

    axes[0].legend()
    axes[0].set_xlabel("distance")
    axes[0].set_ylim((0, 1))
    axes[0].set_zorder(axe2.get_zorder() + 1)
    axes[0].patch.set_visible(False)
    axes[0].set_title("high-d")

    x = np.linspace(0, low_d_max, 1000)
    y = sigmoid(x, *sigmoid_parameters[3:])
    dy = np.diff(y)
    dy_norm = dy / max(dy)
    idx = np.argmin(
        np.abs(np.expand_dims(edges_sig, axis=1) - np.expand_dims(y, axis=0)), axis=1
    )
    edges_x = x[idx]

    axes[1].plot(x, y, color="C1", label="sigmoid")

    axes[1].legend()
    axes[1].set_xlabel("distance")
    axes[1].set_ylim((0, 1))
    axes[1].set_title("low-d")
    for i in range(len(edges)):
        if edges_x[i] != edges_x[-1]:
            axes[1].annotate(
                "",
                xy=(edges[i], 0),
                xytext=(edges_x[i], 0),
                xycoords=axes[0].transData,
                textcoords=axes[1].transData,
                arrowprops=dict(faceyellowcolor="black", arrowstyle="-", clip_on=False),
            )
    axes[0].figure.tight_layout()
    return axes[0], axe2, axes[1]


def raw_data_plot(
    xyz: Union[np.ndarray, "SingleTraj"],
    frame_slice: slice = slice(0, 5),
    atom_slice: slice = slice(0, 50, 5),
) -> None:  # pragma: no cover
    """Plots the raw data of a trajectory as xyz slices in a 3D plot.

    Conventions:
        * x: The cartesian coordinates.
        * y: The atom.
        * z: The frame.

    Args:
        xyz (Union[np.ndarray], "SingleTraj"]): Can be either a numpy array with
            shape (n_frames, n_atoms, 3) or a SingleTraj object.
        frame_slice (slice): A slice to select the frames you want.
        atom_slice (slice): A slice to select the atoms you want.

    """
    if not isinstance(xyz, np.ndarray):
        data = xyz.xyz
        frame_extend, atom_extend = data.shape[:2]
        atoms_ind = []
        for i in np.array(xyz.top.select("all"))[atom_slice]:
            atoms_ind.append(str(xyz.top.atom(i)))
    else:
        data = xyz
        frame_extend, atom_extend = data.shape[:2]
        atoms_ind = np.arange(atom_extend)[atom_slice]
    frames_ind = np.arange(frame_extend)[frame_slice]

    # create the surfaces
    surfaces = []
    cmin = float("inf")
    cmax = -float("inf")
    for frame, xyz_slice in zip(frames_ind, data[frame_slice, atom_slice]):
        x = np.arange(3)
        y = np.arange(len(atoms_ind))
        x, y = np.meshgrid(x, y)
        z = np.full(x.shape, frame)
        cmin = min([cmin, xyz_slice.min()])
        cmax = min([cmax, xyz_slice.max()])
        customdata = np.stack(
            (
                np.full(xyz_slice.T.shape, fill_value=frame),
                np.tile(atoms_ind, [3, 1]),
                np.tile(["x", "y", "z"], [len(atoms_ind), 1]).T,
                xyz_slice.T,
            ),
            axis=-1,
        )
        text = (
            "Cartesian coordinate %{customdata[2]}<br>of atom %{customdata[1]}<br>at "
            "frame %{customdata[0]:.d}: %{customdata[3]:.3f}"
        )
        surfaces.append(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=xyz_slice,
                customdata=customdata,
                coloraxis="coloraxis",
                hovertemplate=text,
                name="",
            )
        )

    # create the figure
    fig = go.Figure(data=surfaces)
    fig.update_layout(
        title_text="Raw data plot",
        title_x=0.5,
        scene=dict(
            xaxis_title="xyz",
            yaxis_title="Atom",
            zaxis_title="Frame no.",
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["x", "y", "z"],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=np.arange(len(atoms_ind)),
                ticktext=atoms_ind,
            ),
            zaxis=dict(
                tickmode="array",
                tickvals=frames_ind,
                ticktext=frames_ind,
            ),
        ),
        legend_title="Cartesian coordinate value",
        width=700,
        height=700,
        coloraxis=dict(
            colorscale="viridis",
            colorbar_thickness=25,
            colorbar_len=0.75,
            cmin=cmin,
            cmax=cmax,
        ),
        coloraxis_colorbar=dict(
            title="value of coordinate",
        ),
    )
    fig.show()


def ball_and_stick_plot(
    traj: "SingleTraj",
    subsample: Union[int, slice] = slice(None, None, 100),
    animation: bool = False,
) -> None:  # pragma: no cover
    # data for plotting and annotation
    xyz = traj.xyz[subsample]
    times = traj.time[subsample]
    atom_names = np.array([str(a) for a in traj.top.atoms])
    bonds = [(a.index, b.index) for a, b in traj.top.bonds]
    sizes = np.array([24 if a.element.symbol != "H" else 10 for a in traj.top.atoms])
    elements = np.array([a.element.number for a in traj.top.atoms])
    coords = [f"x: {i:.3f}<br>y: {j:.3f}<br>z: {k:.3f}" for i, j, k in xyz[0]]

    colormap = {
        1: "rgb(255, 255, 255)",
        6: "rgb(126, 126, 126)",
        7: "rgb(0, 0, 255)",
        8: "rgb(255, 0, 0)",
        16: "rgb(255, 255, 0)",
    }

    color = [colormap[i] for i in elements]

    # create scatter trace
    scatter = go.Scatter3d(
        x=xyz[0, :, 0],
        y=xyz[0, :, 1],
        z=xyz[0, :, 2],
        customdata=np.stack(
            (
                atom_names,
                coords,
            ),
            axis=-1,
        ),
        mode="markers",
        hovertemplate="%{customdata[0]}:<br>%{customdata[1]}",
        name="",
        marker=dict(
            size=sizes,
            color=color,
            opacity=1.0,
        ),
    )

    # create line trace
    x_lines = []
    y_lines = []
    z_lines = []
    for p in bonds:
        for i in range(2):
            x_lines.append(xyz[0, p[i], 0])
            y_lines.append(xyz[0, p[i], 1])
            z_lines.append(xyz[0, p[i], 2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        name="",
        line=dict(
            color="black",
        ),
    )

    # create figure
    fig = go.Figure(
        data=[
            scatter,
            lines,
        ],
    )
    fig.update_layout(
        height=900,
        width=900,
        showlegend=False,
    )

    # create frames
    if animation:
        frames = [go.Frame(data=[scatter, lines])]
        for points in xyz:
            x_lines = []
            y_lines = []
            z_lines = []
            for p in bonds:
                for i in range(2):
                    x_lines.append(points[p[i], 0])
                    y_lines.append(points[p[i], 1])
                    z_lines.append(points[p[i], 2])
                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        customdata=atom_names,
                        mode="markers",
                        hovertemplate="%{customdata}",
                        name="",
                        marker=dict(
                            size=sizes,
                            color=color,
                            opacity=1.0,
                        ),
                    ),
                    go.Scatter3d(
                        x=x_lines,
                        y=y_lines,
                        z=z_lines,
                        mode="lines",
                        name="",
                        line=dict(
                            color="black",
                        ),
                    ),
                ],
            )
            frames.append(frame)
        fig.update(frames=frames)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        redraw=True, fromcurrent=True, mode="immediate"
                                    )
                                ),
                            ],
                        )
                    ],
                )
            ],
            # sliders=(
            #     [
            #         {
            #             "steps": [
            #                 {
            #                     "args": [
            #                         [f.name],
            #                         {
            #                             "frame": {"duration": 0, "redraw": True},
            #                             "mode": "immediate",
            #                         },
            #                     ],
            #                     "label": f.name,
            #                     "method": "animate",
            #                 }
            #                 for f in frames
            #             ],
            #         }
            #     ],
            # ),
            scene=dict(
                xaxis=dict(range=[np.min(xyz[..., 0]), np.max(xyz[..., 0])]),
                yaxis=dict(range=[np.min(xyz[..., 1]), np.max(xyz[..., 1])]),
                zaxis=dict(range=[np.min(xyz[..., 2]), np.max(xyz[..., 2])]),
            ),
        )
    fig.show()


def ramachandran_plot(
    angles: Union[tuple[np.ndarray, np.ndarray], np.ndarray, "SingleTraj"],
    subsample: Optional[Union[int, slice, np.ndarray]] = None,
) -> None:  # pragma: no cover
    """Plots a Ramachandran plot using plotly.

    Args:
        angles (Union[tuple[np.ndarray, np.ndarray], np.ndarray, "SingleTraj"]):
            Either a tuple of np.ndarray in which case it is assumed that the
            arrays are ordered like (psi, phi). Or an array of shape
            (2, n_frames, n_angles), in which case it is unpacked into psi and
            phi angles.

    """
    if isinstance(angles, tuple):
        psi, phi = angles
    elif isinstance(angles, np.ndarray):
        if angles.ndim == 3:
            psi, phi = angles
        else:
            psi, phi = angles[::2], angles[1::2]
    elif angles.__class__.__name__ == "SingleTraj":
        if not "central_dihedrals" in angles._CVs:
            angles.load_CV("central_dihedrals")
        _angles = angles._CVs.central_dihedrals
        psi = _angles[
            0, ..., _angles.CENTRAL_DIHEDRALS.str.lower().str.contains("psi")
        ].values
        phi = _angles[
            0, ..., _angles.CENTRAL_DIHEDRALS.str.lower().str.contains("phi")
        ].values
    else:
        raise ValueError("Wrong type for arg `angles`.")

    psi = psi.flatten()
    phi = phi.flatten()
    if np.all(psi < 4):
        mode = "rad"
        tickrange = np.linspace(-np.pi, np.pi, 5)
        ranges = [-np.pi, np.pi]
    else:
        mode = "deg"
        ranges = [-180, 180]
    tickrange = np.linspace(ranges[0], ranges[1], 5)
    ticklabels = [-180, -90, 0, 90, 180]
    if subsample is not None:
        psi = psi[::subsample]
        phi = phi[::subsample]

    fig = px.density_contour(
        x=phi,
        y=psi,
        marginal_x="violin",
        marginal_y="violin",
        labels={"x": "phi", "y": "psi"},
        range_x=ranges,
        range_y=ranges,
    )

    fig.data[0]["contours"].coloring = "fill"
    # fig.update_traces(contours_coloring="fill", contours_showlabels = True)

    fig.update_layout(
        width=700,
        height=700,
        title_text="Ramachandran plot",
        xaxis=dict(
            tickmode="array",
            tickvals=tickrange,
            ticktext=ticklabels,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tickrange,
            ticktext=ticklabels,
        ),
    )

    fig.show()


def dssp_plot(
    traj: SingleTraj,
    simplified: bool = True,
    subsample: Optional[Union[int, slice, np.ndarray]] = None,
    residue_subsample: int = 25,
) -> None:  # pragma: no cover
    # get the dssp and color values
    # Third Party Imports
    import mdtraj as md

    dssp = md.compute_dssp(traj, simplified=simplified)

    # the yticks and yticklabels are created here
    residues = np.arange(traj.top.n_residues)
    residue_names = np.array([str(r) for r in traj.top.residues])

    # subsample the dssp array
    if subsample is not None:
        if isinstance(subsample, int):
            subsample = slice(None, None, subsample)
        dssp = dssp[subsample]

    # vectorize the dssp-str -> rgb-value function and apply
    func = np.vectorize(partial(dssp_to_rgb, simplified=simplified))
    dssp_color = np.swapaxes(np.dstack(func(dssp)), 0, 1)
    func = np.vectorize(partial(dssp_to_text, simplified=simplified))
    dssp_text = func(dssp)

    # create a bar-chart that is hidden to use its legend
    # bar_data = pd.DataFrame(dssp, columns=residue_names)
    # bar_data = bar_data.apply(pd.Series.value_counts).fillna(0).astype(int).T
    # if not simplified:
    #     fig1 = px.bar(bar_data)
    # else:
    #     raise NotImplementedError
    # print(fig1)
    # fig1.show()
    # return

    # create fig
    fig = px.imshow(dssp_color)
    customdata_res_names = np.tile(residue_names, (dssp.shape[0], 1))
    customdata = np.stack((customdata_res_names.T, dssp_text.T), axis=-1)
    fig.update(
        data=[
            {
                "customdata": customdata,
                "hovertemplate": "Time: %{x}<br>Residue: %{customdata[0]}<br>DSSP: %{customdata[1]}",
                "name": "",
            },
        ],
    )

    # subsample the residues for label purposes
    if residue_subsample > -1:
        residues = residues[::residue_subsample]
        residue_names = residue_names[::residue_subsample]

    # combine and update layout
    # fig = go.Figure(data=fig1.data + fig2.data)
    fig.update_layout(
        width=1000,
        height=700,
        title="DSSP plot",
        xaxis=dict(title="time"),
        yaxis=dict(
            title="residue",
            tickmode="array",
            tickvals=residues,
            ticktext=residue_names,
        ),
        legend=dict(
            title="DSSP",
            orientation="h",
        ),
    )

    # add the legend
    simplified_legend = {
        "Coil": "rgb(1.0, 1.0, 1.0)",
        "Extended": "rgb(1.0, 0.0, 0.0)",
        "Helical": "rgb(0.0, 0.0, 1.0)",
    }
    legend = {
        "Coil": "rgb(1.0, 1.0, 1.0)",
        "Isolated beta-bridge": "rgb(0.0, 0.0, 0.0)",
        "Extended beta-ladder": "rgb(1.0, 0.0, 0.0)",
        "3/10-helix": "rgb(0.5, 0.5, 0.5)",
        "Alpha-helix": "rgb(0.0, 0.0, 1.0)",
        "Pi-helix": "rgb(0.0, 1.0, 1.0)",
        "Bend": "rgb(0.0, 1.0, 0.0)",
        "Hydrogen bonded Turn": "rgb(1.0, 1.0, 0.0)",
    }
    iterator = simplified_legend if simplified else legend
    for key, val in iterator.items():
        trace = go.Bar(
            x=[0],
            y=[0],
            name=key,
            legend="legend1",
            # visible="legendonly",
            showlegend=True,
        )
        trace.update(
            marker_color=val,
        )
        fig.add_trace(trace)
    # show
    fig.show()


def dssp_to_text(
    val: str,
    simplified: bool = False,
) -> str:  # pragma: no cover
    simplified_dssp = {"C": "Coil", "E": "Extended", "H": "Helical"}
    dssp = {
        " ": "Coil",
        "B": "Isolated beta-bridge",
        "E": "Extended beta-ladder",
        "G": "3/10-helix",
        "H": "Alpha-helix",
        "I": "Pi-helix",
        "S": "Bend",
        "T": "Hydrogen bonded Turn",
    }
    if simplified:
        return simplified_dssp[val]
    return dssp[val]


def dssp_to_rgb(
    val: str,
    simplified: bool = False,
) -> tuple[int, int, int]:  # pragma: no cover
    """Here are the values returned for simplified:

        * "C": coil, white, rgb(1, 1, 1)
        * "E": extended, red, rgb(1, 0, 0)
        * "H": helix, blue, rgb(0, 0, 1)

    And here for the full DSSP assignment:

        * " ": coil, white, rgb(1, 1, 1)
        * "B": b-bridge, black, rgb(0, 0, 0)
        * "E": b-sheet, red, rgb(1, 0, 0)
        * "G": 3_10 helix, grey, rgb(0.5, 0.5, 0.5)
        * "H": A-helix, blue, rgb(0, 0, 1)
        * "I": pi-helix, purple, rgb(0, 1, 1)
        * "S": bend, green, rgb(0, 1, 0)
        * "T": turn, yellow(1, 1, 0)

    Args:
        val (str): The dssp value.
        simplified (bool): Whether to use the simplified scheme.

    """
    simplified_dssp = {"C": (1.0, 1.0, 1.0), "E": (1.0, 0.0, 0.0), "H": (0.0, 0.0, 1.0)}
    dssp = {
        " ": (1.0, 1.0, 1.0),
        "B": (0.0, 0.0, 0.0),
        "E": (1.0, 0.0, 0.0),
        "G": (0.5, 0.5, 0.5),
        "H": (0.0, 0.0, 1.0),
        "I": (0.0, 1.0, 1.0),
        "S": (0.0, 1.0, 0.0),
        "T": (1.0, 1.0, 0.0),
    }
    if simplified:
        return simplified_dssp[val]
    return dssp[val]


def end2end_plot(
    traj: SingleTraj,
    selstr: str = "name CA",
    subsample: Optional[Union[int, slice, np.ndarray]] = None,
    rolling_avg_window: int = 5,
) -> None:  # pragma: no cover
    atoms = traj.top.select(selstr)[[0, -1]]
    dists = md.compute_distances(traj, [atoms])[:, 0]
    time = traj.time
    if subsample is not None:
        if isinstance(subsample, int):
            subsample = slice(None, None, subsample)
        dists = dists[subsample]
        time = time[subsample]

    fig = px.scatter(
        x=time,
        y=dists,
        labels=dict(
            x="time in ps",
            y="dist in nm",
        ),
        opacity=0.2,
        trendline="rolling",
        trendline_options=dict(
            window=rolling_avg_window,
        ),
        title="end to end distance",
        marginal_y="violin",
    )

    fig.show()


def _zoomingBoxManual(ax1, ax2, color="red", linewidth=2, roiKwargs={}, arrowKwargs={}):
    """Fakes a zoom effect between two mpl.axes.Axes.

    Uses mpl.patches.ConnectionPatch and mpl.patches.Rectangle
    to make it seem like ax2 is a zoomed in version of ax1.
    Instead of defining the coordinates of the zooming rectangle
    The axes limits of ax2 are used.

    Args:
        ax1 (plt.axes): The axes with the zoomed-out data.
        ax2 (plt.axes): The second axes with the zoomed-in data.
        color (str): The color of the zoom effect. Is passed into mpl,
            thus can be str, or tuple, ... Defaults to 'red'
        linewidth (int): The linewidth. Defaults to 2.
        roiKwargs (dict): Keyworded arguments for the rectangle.
            Defaults to {}.
        arrowKwargs (dict): Keyworded arguments for the arrow.
            Defaults to {}.

    """
    limits = np.array([*ax2.get_xlim(), *ax2.get_ylim()])
    roi = limits
    roiKwargs = dict(
        dict(
            [
                ("fill", False),
                ("linestyle", "dashed"),
                ("color", color),
                ("linewidth", linewidth),
            ]
        ),
        **roiKwargs,
    )
    ax1.add_patch(
        mpl.patches.Rectangle(
            [roi[0], roi[2]], roi[1] - roi[0], roi[3] - roi[2], **roiKwargs
        )
    )
    arrowKwargs = dict(
        dict([("arrowstyle", "-"), ("color", color), ("linewidth", linewidth)]),
        **arrowKwargs,
    )
    corners = np.vstack([limits[[0, 1, 1, 0]], limits[[2, 2, 3, 3]]]).T
    con1 = mpl.patches.ConnectionPatch(
        xyA=corners[0],
        xyB=corners[1],
        coordsA="data",
        coordsB="data",
        axesA=ax2,
        axesB=ax1,
    )
    con1.set_color([0, 0, 0])
    ax2.add_artist(con1)
    con1.set_linewidth(2)
    con2 = mpl.patches.ConnectionPatch(
        xyA=corners[3],
        xyB=corners[2],
        coordsA="data",
        coordsB="data",
        axesA=ax2,
        axesB=ax1,
    )
    con2.set_color([0, 0, 0])
    ax2.add_artist(con2)
    con2.set_linewidth(2)


def render_vmd(
    filepath,
    rotation=[0, 0, 0],
    scale=1,
    script_location="auto",
    image_location="auto",
    debug=False,
    image_name="",
    drawframes=False,
    ssupdate=True,
    renderer="tachyon",
    additional_spheres=[],
    additional_lines=[],
    surf=None,
    custom_script=None,
):
    """Render pdb file with combination of vmd, tachyon and image magick.

    This function creates a standardised vmd tcl/tk script and writes it
    to disk. Then vmd is called with the subprocess package and used to
    create a tachyon input file. Tachyon is then called to render the image
    with ambient occlusion and soft lighting. The output is a high quality
    targa (.tga) image, which is converted to png using image magick.

    Args:
        filepath (str): Location of the pdb file which should be rendered.
        rotation ([x_rot, y_rot, z_rot], optional): List of rotation values. Defaults to [0, 0, 0].
        scale (float, optional): By how much the structure should be scaled. Defaults to 1.
        script_location (str, optional): Where to save the script. Script will be removed
            after finish nonehteless. Defaults to 'auto' and writes into cwd.
        image_location (str, optional): Where to render the images file to. Will be
            deleted nonetheless. Don't give an extension for this. Defaults to 'auto' and
            writes into cwd.
        debug (bool, optional): Print debug info. Defaults to False.
        image_name (str, optional): This string will be used to save the image to after it has
            been rendered and converted to png. This will not be deleted. Defaults to ''.
        drawframes (bool, optional): If a trajectory is loaded, this will render all frames in it.
            Defaults to False.
        ssupdate (bool, optional): Updates the secondary structure for every frame. Normally
            vmd uses the secondary structure of the first frame. Setting this to True calcs
            the sec struct for every frame. Defaults to True.
        renderer (str, optional): Which renderer to use.
            * 'tachyon' uses the external Tachyon rendered. So vmd -> .dat -> .tga -> .png.
            * 'snapshot' uses the vmd internal snapshot renderer.
            Defaults to 'tachyon'.
        additional_spheres (list, optional): Draw spheres around two subunits to make
            them visually more distinct. Takes a list of lists. Each list in the main
            list should contain 4 values [x, y, z, r] (r for radius). Defaults to [].
        additional_lines (list, optional): A list of additional lines that should be added to the
            script. Please refert to the vmd manual for further info. Defaults to [].
        surf (Union[str, None], optional): A string defining the surface renderer. Can either be
            'quicksurf' or 'surf'. If None is provided, the surface won't be rendered (falls back
            to cartoon representation). Defaults to None.
        custom_script (Union[str, None], optional): Provide a completely custom script as this option.
            The render commands will still be appended to this script. If None is provided, the
            default script will be used.

    See also:
        See this nice webpage about rendering publication worthy images with vmd.
        https://www.ks.uiuc.edu/Research/vmd/minitutorials/tachyonao/

    Returns:
        image (np.ndarray): This array contains the raw pixel data.
            Can be used with matplotlib to have a quick view of the image.

    """
    if "." in image_location:
        raise Exception(
            "The argument image_location does not take a file extension, because the name is used for a .dat, .tga and .png file."
        )

    # add a shebang to the script
    # script = '#!/home/soft/bin/vmd\n\n'

    # print debug hello world
    script = 'puts "Hello World"\n'

    # if a list of files is provided we iterate over them
    if isinstance(filepath, list):
        for i, file in enumerate(filepath):
            # load molecule and change representation
            script += f"mol new {file}\n"
            if surf is None:
                script += f"mol modstyle 0 {i} newcartoon 0.3 50\n"
                script += f"mol modcolor 0 {i} structure\n"
            elif surf == "quicksurf":
                script += f"mol modstyle 0 {i} quicksurf 0.6 0.7 0.7 Medium\n"
            else:
                script += f"mol modstyle 0 {i} {surf}\n"
            script += f"mol modmaterial 0 {i} AOChalky\n"
            if drawframes and md.load(file).n_frames > 1:
                if renderer == "STL":
                    # Standard Library Imports
                    import warnings

                    warnings.warn(
                        "Rendering multiple frames with STL may lead to "
                        "undesired results. Instead of yielding the union of "
                        "all single-frame surfaces, you will get a mishmash of "
                        "all surfaces with intersection faces etc."
                    )
                script += f"mol drawframes 0 {i} 0:1:999\n"
    else:
        # load molecule and change representation
        script += f"mol new {filepath}\n"
        if surf is None:
            script += "mol modstyle 0 0 newcartoon 0.3 50\n"
            script += "mol modcolor 0 0 structure\n"
        elif surf == "quicksurf":
            script += "mol modstyle 0 0 quicksurf 0.6 0.7 0.7 Medium\n"
        else:
            script += f"mol modstyle 0 0 {surf}\n"
        script += "mol modmaterial 0 0 AOChalky\n"
        if drawframes:
            script += "mol drawframes 0 0 0:1:999\n"

    if ssupdate:
        print(
            "\033[93m"
            + "For the ssupdate function to work encodermap/vmd/sscache.tcl will be sourced within vmd. If no Error is thrown the file is present."
            + "\033[0m"
        )
        sscache_location = (
            os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
            + "/vmd/sscache.tcl"
        )
        if not os.path.isfile(sscache_location):
            raise FileNotFoundError(
                f"The sscache.tcl script is not here: {sscache_location}. Please put it there."
            )
        script += f"source {sscache_location}\n"
        script += "start_sscache 0\n"
    #         script += "proc update_secondary_structure_assigment { args } {"
    #         script += "  foreach molid [molinfo list] {"
    #         script += "    mol ssrecalc $molid"
    #         script += "  }"
    #         script += "}"
    #         script += "trace variable vmd_frame(0) w update_secondary_structure_assigment"

    # change some parameters to make a nice image
    script += "color Display Background white\n"
    script += "color Axes Labels black\n"
    script += "display depthcue off\n"
    script += "display ambientocclusion on\n"
    script += "display aoambient 1.0\n"
    script += "display aodirect 0.3\n"
    script += "display antialias on\n"
    # script += 'display resize 2000 2000\n'
    script += "axes location off\n"

    # scale and rotate
    script += f"rotate x by {rotation[0]}\n"
    script += f"rotate y by {rotation[1]}\n"
    script += f"rotate z by {rotation[2]}\n"
    script += f"scale by {scale}\n"

    # define image location
    if image_location == "auto":
        image_location = os.getcwd() + "/vmdscene"

    # add spheres
    if np.any(additional_spheres):
        for _, color in zip(additional_spheres, ["grey", "iceblue"]):
            x, y, z, r = np.round(_, 2)
            script += f"draw color {color}\n"
            script += f"draw sphere {{ {x} {y} {z} }} radius {r} resolution 25\n"
            script += "draw material Transparent\n"

    # add additional lines
    if additional_lines:
        for line in additional_lines:
            script += line + "\n"

    if custom_script is not None:
        script = custom_script

    # render command. Alternatively, I can use external Tachyon, which makes better images
    if renderer == "tachyon":
        script += f"render Tachyon {image_location}.dat\n"
    elif renderer == "snapshot":
        script += "render aasamples TachyonInternal 6\n"
        script += f"render TachyonInternal {image_location}.tga\n"
    elif renderer == "STL":
        script += "axes location off\n"
        script += f"render STL {image_location}.stl\n"
    elif renderer == "Wavefront":
        script += "axes location off\n"
        script += f"render Wavefront {image_location}.obj\n"
    else:
        raise NotImplementedError(
            "Other renderers than tachyon and snaphsot currently not supported."
        )

    # list molecules and quit
    script += "mol list\n"
    script += "quit"

    if debug:
        print(script)

    # write the script
    if script_location == "auto":
        script_location = os.getcwd() + "/vmd_script.tcl"
    with open(script_location, "w") as f:
        f.write(script)

    # call vmd -e script
    cmd = f"vmd -e {script_location} -dispdev none"
    if debug:
        print(cmd)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    (stdout, stderr) = proc.communicate()
    if debug:
        print(stdout.decode("utf-8"))
        print("\n")
        print(stderr.decode("utf-8"))

    # check if image has been written
    if renderer == "tachyon":
        assert os.path.isfile(
            f"{image_location}.dat"
        ), "Tachyon datafile not generated by renderer"
    else:
        assert os.path.isfile(
            f"{image_location}.tga"
        ), f"Snapshot image not created. {stderr.decode()} {stdout.decode()}"

    time.sleep(2)
    assert os.path.isfile(
        f"{image_location}.tga"
    ), f"Tachyon datafile not generated by renderer. Here's the script:\n\n{script}\n\n"

    if renderer == "tachyon":
        # call Tachyon and render
        cmd = f"/usr/bin/tachyon -aasamples 12 {image_location}.dat -res 2000 2000 -fullshade -format TARGA -o {image_location}.tga"
        if debug:
            print(cmd)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        (stdout, stderr) = proc.communicate()
        if debug:
            print(stdout.decode("utf-8"))
            print("\n")
            print(stderr.decode("utf-8"))

        # check if image has been written
        assert os.path.isfile(
            f"{image_location}.tga"
        ), "Tachyon renderer did not render image"

    if renderer == "STL":
        if image_name:
            shutil.copyfile(f"{image_location}.stl", image_name)
        # Third Party Imports
        import trimesh

        mesh = trimesh.load(f"{image_location}.stl")
        os.remove(f"{image_location}.stl")
        return mesh

    if renderer == "Wavefront":
        if image_name:
            shutil.copyfile(f"{image_location}.obj", image_name)
            shutil.copyfile(f"{image_location}.mtl", image_name.replace(".obj", ".mtl"))
        print(
            f"Find the rendered images at {image_name} and {image_name.replace('.obj', '.mtl')}."
        )
        return None

    # convert to png
    cmd = f"/usr/bin/convert {image_location}.tga {image_location}.png"
    if debug:
        print(cmd)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    (stdout, stderr) = proc.communicate()
    if debug:
        print(stdout.decode("utf-8"))
        print("\n")
        print(stderr.decode("utf-8"))

    # read image
    image = plt.imread(f"{image_location}.png")

    # write image if name has been provided
    if image_name:
        if os.path.isabs(image_name):
            shutil.copyfile(f"{image_location}.png", image_name)
        else:
            shutil.copyfile(f"{image_location}.png", os.getcwd() + f"/{image_name}")

    # remove temporary files
    if renderer == "tachyon":
        os.remove(f"{image_location}.dat")
    os.remove(f"{image_location}.tga")
    os.remove(f"{image_location}.png")
    # os.remove(f'{script_location}')

    # return matplotlib image object
    return image


def render_movie(path, scatter_data, dummy_traj):
    pass


def plot_cluster(
    trajs, pdb_path, png_path, cluster_no=None, col="user_selected_points"
):
    # Third Party Imports
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(20, 20)

    if cluster_no is None:
        cluster_no = trajs.CVs[col].max()

    # prepare ax1 to make the two side histograms
    divider = make_axes_locatable(ax4)
    axHistx = divider.append_axes("top", size=1.2, pad=0.1)  # , sharex=ax1)
    axHisty = divider.append_axes("right", size=1.2, pad=0.1)  # , sharey=ax1)

    # some data management
    data = trajs.lowd
    where = np.where(trajs.CVs[col] == cluster_no)
    not_where = np.where(trajs.CVs[col] != cluster_no)
    x = data[:, 0]
    y = data[:, 1]

    # scatter everything grey and cluster blue
    ax1.scatter(*data[where].T)
    ax1.scatter(*data[not_where].T, c="grey", s=5)
    ax1.set_xlabel("x in a.u.")
    ax1.set_ylabel("y in a.u.")
    ax1.set_title(f"Scatter of low-dimensional data")

    # density
    bin_density = 46
    log_density = True

    # ax2 gets hexbin density
    # x_bins = np.linspace(x.min(), x.max(), bin_density)
    # y_bins = np.linspace(y.min(), y.max(), bin_density)
    H, xedges, yedges = np.histogram2d(x=x, y=y, bins=bin_density)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
    ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
    X, Y = np.meshgrid(xcenters, ycenters)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if log_density:
        with np.errstate(divide="ignore"):  # ignore division by zero error
            F = np.log(H)
    else:
        F = H
    # mappable = ax2.hexbin(x=X.ravel(), y=Y.ravel(), C=F.T.ravel(), cmap=plt.cm.turbo_r, extent=extent,
    #                       norm=mpl.colors.PowerNorm(1), gridsize=bin_density +1)
    cmap = plt.get_cmap("turbo").with_extremes(under="w")
    mappable = ax2.contourf(
        X, Y, H.T, cmap=cmap, levels=np.linspace(0.001, H.max(), 20)
    )
    ax2.set_xlabel("x in a.u.")
    ax2.set_ylabel("y in a.u.")
    ax2.set_title("Log density of points")

    # colorbar for ax2
    # colorbar
    # use the axes divider method to add colorbar
    ax_divider = make_axes_locatable(ax2)
    # add colorbaraxis to work with ticks and whatnot
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    # define colorbar norm. I like to work with values between 0 and 1
    # initialize colormap
    cb = plt.colorbar(mappable, cax=cax)
    cax.set_ylabel("Number of points")

    # cluster on ax4
    # x hist
    spines = [k for k in axHistx.spines.values()]
    spines[1].set_linewidth(0)
    spines[3].set_linewidth(0)
    axHistx.set_xticks([])
    H, edges, patches = axHistx.hist(data[:, 0][where], bins=50)
    axHistx.set_ylabel("count")
    axHistx.set_title("Scatter of Cluster")

    # y hist
    spines = [k for k in axHisty.spines.values()]
    spines[1].set_linewidth(0)
    spines[3].set_linewidth(0)
    axHisty.set_yticks([])
    H, edges, patches = axHisty.hist(
        data[:, 1][where], bins=50, orientation="horizontal"
    )
    axHisty.set_xlabel("count")

    # scatter data
    ax4.scatter(x=data[where, 0], y=data[where, 1])
    spines = [k for k in ax4.spines.values()]
    spines[3].set_linewidth(0)
    spines[1].set_linewidth(0)
    ax4.set_xlabel("x in a.u.")
    ax4.set_ylabel("y in a.u.")

    # annotate rms
    rms = np.sqrt(
        (1 / len(data[where]))
        * np.sum(
            (data[where, 0] - np.mean(data[where, 0])) ** 2
            + (data[where, 1] - np.mean(data[where, 1])) ** 2
        )
    )
    text = f"RMS = {np.round(rms, decimals=5)}"
    ax4.text(0.05, 0.95, text, transform=ax1.transAxes)

    # annotate geometric center
    centroid = [np.mean(x[where]), np.mean(y[where])]
    ax4.scatter(*centroid, s=50, c="C1")
    ax4.annotate(
        "geom. center",
        xy=centroid,
        xycoords="data",
        xytext=(0.95, 0.95),
        textcoords="axes fraction",
        arrowprops=dict(facecolor="black", shrink=0.05, fc="w", ec="k", lw=2),
        horizontalalignment="right",
        verticalalignment="top",
        color="C1",
    )

    # annotate rmsd center
    # view, dummy_traj = gen_dummy_traj(trajs, cluster_no, max_frames=100, col=col)
    # index, distances, centroid = rmsd_centroid_of_cluster(dummy_traj, parallel=False)
    # idx = np.round(np.linspace(0, len(where) - 1, 100)).astype(int)
    # where = where[idx]
    # centroid = data[where[0][::5][index]]
    # ax4.scatter(*centroid, s=50, c='C2')
    # ax4.annotate('rmsd center',
    #             xy=centroid, xycoords='data',
    #             xytext=(0.95, 0.85), textcoords='axes fraction',
    #             arrowprops=dict(facecolor='black', shrink=0.05, fc="w", ec="k", lw=2),
    #             horizontalalignment='right', verticalalignment='top', color='C2')

    # make vmd snapshot
    try:
        image = render_vmd(
            pdb_path, drawframes=True, renderer="tachyon", debug=False, scale=1.5
        )
        ax3.imshow(image)
        [k.set_linewidth(0) for k in ax3.spines.values()]
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Image of cluster")
    except:
        ax3.annotate("VMD Rendering not possible", (0.5, 0.5))
        pass

    # # calculate distances between rmsd centroid and all other points
    # distances = scipy.spatial.distance.cdist(centroid.reshape(1, 2), np.stack([x, y]).T)
    # H, edges, patches = ax3.hist(distances.flatten(), color='C1')
    # ax3.set_title("Distances to rmsd centroid.")
    # ax3.set_xlabel("Distance in a.u.")
    # ax3.set_ylabel("Count")

    plt.suptitle(f"Cluster {cluster_no}")
    plt.savefig(png_path, transparent=False)
    plt.close(fig)
