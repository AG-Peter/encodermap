# -*- coding: utf-8 -*-
# encodermap/plot/interactive_plotting.py
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
"""ToDo:
    * Check if vmd rendering works
    * Animate the path and the torsions inside the Interactive Plotting class.
    * ToolTip when hovering over buttons.
    * Path Tool.
    * Allow a path that selects closest points for points on path. Maybe do both for any given bezier/path path.
    * Allow autoencoder to be None. Catch the exception that tries to get autoencoder.trajs.
    * Superpose option with mdtraj not working
    * Keep the lasso select.
    * Movie along paths.

"""

##############################################################################
# Imports
##############################################################################

# Standard Library Imports
import copy
import os
from itertools import groupby

# Third Party Imports
from matplotlib.path import Path
from optional_imports import _optional_import

# Local Folder Imports
from ..autoencoder.autoencoder import AngleDihedralCartesianEncoderMap
from ..misc.clustering import gen_dummy_traj, get_cluster_frames
from ..misc.misc import _datetime_windows_and_linux_compatible, all_equal
from ..plot.plotting import plot_cluster
from ..trajinfo.info_all import TrajEnsemble
from ..trajinfo.info_single import SingleTraj
from .utils import *
from .utils import _unpack_cluster_info, _unpack_path_info


##############################################################################
# Optional Imports
##############################################################################


sns = _optional_import("seaborn")


##############################################################################
# Classes
##############################################################################


class InteractivePlotting:
    """Class to open up an interactive plotting window.

    Contains subclasses to handle user-clickable menus and selectors.

    Attributes:
        trajs (encodermap.TrajEnsemble): The trajs passed into this class.
        fig (matplotlib.figure): The figure plotted onto. If ax is passed when
            this class is instantiated, the parent figure will be fetched with
            self.fig = self.ax.get_figure()
        ax (matplotlib.axes): The axes where the lowd data of the trajs
            is plotted on.
        menu_ax (matplotlib.axes): The axes where the normal menu is plotted on.
        status_menu_ax (matplotlib.axes): The axes on which the status menu is plotted on.
        pts (matplotlib.collections.Collection): The points which are plotted. Based on some
            other class variables, the color of this collection is adjusted.
        statusmenu (encodermap.plot.utils.StatusMenu): The menu containing the
            status buttons.
        menu (encodermap.plot.utils.Menu): The menu containing the remaining buttons.
        tool (encodermap.plot.utils.SelectFromCollection): The current active
            tool used to select points. This can be lasso, polygon, etc...
        mode (str): Current mode of the statusmenu.

    """

    def __init__(
        self,
        autoencoder,
        trajs=None,
        data=None,
        ax=None,
        align_string="name CA",
        top=None,
        hist=False,
        scatter_kws={"s": 5},
        ball_and_stick=False,
        top_index=0,
    ):
        """Instantiate the InteractivePlotting class.

        Args:
            trajs (encodermap.TrajEnsemble): The trajs of which the lowd info
                should be plotted.
            ax (matplotlib.axes, optional): On what axes to plot. If no axis is provided
                a new figure and axes will be created, defaults to None.

        """
        # the align string for the cluster dummy method
        self.align_string = align_string
        self.top = top
        self.hist = hist
        self.autoencoder = autoencoder
        self.ball_and_stick = ball_and_stick
        self.top_index = top_index

        # scatter kws
        self.scatter_kws = {**{"s": 80, "alpha": 0.5}, **scatter_kws}

        # close all plots
        plt.close("all")

        # decide on fate of data
        if data is None:
            if hasattr(trajs, "lowd"):
                print("Using the attribute `lowd` of provided `trajs`")
                data = trajs.lowd
            elif isinstance(trajs, (TrajEnsemble, SingleTraj)) and (
                isinstance(autoencoder, AngleDihedralCartesianEncoderMap)
                or autoencoder.__class__.__name__ == "AngleDihedralCartesianEncoderMap"
            ):
                print(
                    "Using the provided `autoencoder` and `trajs` to create a projection."
                )
                data = autoencoder.encode(trajs)
            elif isinstance(data, np.ndarray) and hasattr(autoencoder, "encode"):
                print("Using the `encode` method of `autoencoder` with provided data.")
                if np.any(np.isnan(data)):
                    # Third Party Imports
                    import tensorflow as tf

                    indices = np.stack(np.where(~np.isnan(data))).T.astype("int64")
                    dense_shape = data.shape
                    values = data[~np.isnan(data)].flatten().astype("float32")
                    data = tf.sparse.SparseTensor(indices, values, dense_shape)
                data = autoencoder.encode(data)
            elif hasattr(autoencoder, "encode"):
                print("Using the `train_data` attribute of `autoencoder`.")
                data = autoencoder.encode()
            else:
                print("Mocking data with np.random")
                np.random.seed(19680801)
                data = np.random.rand(100, 2)
        if data.shape[1] != 2:
            print("Using provided `data` to call encoder.")
            data = autoencoder.encode(data)
        self.data = data

        # see what traj has been provided
        if trajs is None:
            self.trajs = autoencoder.trajs
        else:
            if isinstance(trajs, str):
                self.trajs = SingleTraj(trajs, self.top, traj_num=0)._gen_ensemble()
            elif isinstance(trajs, list):
                self.trajs = TrajEnsemble(trajs, self.top)
            else:
                self.trajs = trajs

        if isinstance(trajs, SingleTraj):
            if "lowd" not in self.trajs.CVs:
                self.trajs.load_CV(self.data, attr_name="lowd")
        else:
            if "lowd" not in self.trajs.CVs:
                self.trajs.load_CVs(self.data, attr_name="lowd")

        # decide what function to use to build clusters
        # Decided against gen_dummy traj as get_cluster_frames works better with jinja2
        self.cluster_building_fn = get_cluster_frames

        # create fig and ax
        if ax is None:
            # create fig and ax
            subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=True)
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(10, 8)
            )  # subplot_kw=subplot_kw)
        else:
            self.ax = ax
            self.fig = self.ax.get_figure()

        # add the axes to create the menus on
        self.fig.subplots_adjust(left=0.3)
        self.menu_ax = plt.axes([0.05, 0.1, 0.15, 0.35], facecolor="lightblue")
        self.status_menu_ax = plt.axes(
            [0.05, 0.49, 0.15, 0.35], facecolor="lightyellow"
        )

        # remove everything in these axes
        self.menu_ax.axis("off")
        self.status_menu_ax.axis("off")
        self.tool = DummyTool()

        # plot
        self.pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], **self.scatter_kws)

        # hist
        if self.hist:
            self.ax.hist2d(*data.T, bins=400, norm=mpl.colors.LogNorm())

        # Check whether mouse enters drawing area
        # Upon entering drawing area tools are initialized based on current mode
        # Leave event currently serves no purpose
        self.cid_ax_enter = self.fig.canvas.mpl_connect(
            "axes_enter_event", self.on_enter_ax
        )
        self.cid_ax_leave = self.fig.canvas.mpl_connect(
            "axes_leave_event", self.on_leave_ax
        )

        # chech button presses and compare them with the status of the menuitems
        self.cid_on_click = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_click
        )

        # Instantiate Menu
        self.statusmenu = StatusMenu(self.status_menu_ax)
        self.menu = Menu(self.menu_ax)

        # Show
        plt.show()

    def on_click(self, event):
        """Decides whether the release event happened in the drawing area or the menu.

        Args:
            event (matplotlib.backend_bases.Event): The event provided by figure.canvas.connect().

        """
        if event.inaxes == self.ax:
            self.on_click_tool(event)
        else:
            self.on_click_menu(event)

    def on_enter_ax(self, event):
        """Chosses the tool to use when self.ax is entered, based on current mode.

        Args:
            event (matplotlib.backend_bases.Event): The event provided by figure.canvas.connect().

        """
        # print('Axis is entered')
        if event.inaxes is self.ax and self.mode == "Idle":
            # reset point coloration
            self.pts.set_color("C0")
            self.tool.disconnect()
        if event.inaxes is self.ax and self.mode != "Idle":
            # statusmenu
            for key, item in self.statusmenu.menuitems.items():
                if self.mode == key:
                    method = getattr(self, key.lower())
                    method()

    def on_leave_ax(self, event):
        """Disconnect the current tool."""
        pass

    def on_click_tool(self, event):
        """Left here for convenience if some tools need a button release event."""
        pass

    def on_click_menu(self, event):
        """Chooses the function to call based on what MenuItem was clicked.

        Args:
            event (matplotlib.backend_bases.Event): The event provided by figure.canvas.connect().

        """
        for key, item in self.menu.menuitems.items():
            if item.check_select(event):
                method = getattr(self, key.lower().replace(" ", "_"))
                method()

    def reset(self):
        """Called when 'Reset' is pressed."""
        if "user_selected_points" in self.trajs.CVs:
            self.trajs._CVs.drop(labels="user_selected_points")
        self.__init__(
            self.trajs,
            self.autoencoder,
            self.data,
            None,
            self.align_string,
            self.top,
            self.hist,
            self.scatter_kws,
            self.ball_and_stick,
        )

    def write(self):
        """Called when 'Write' is pressed."""
        if self.mode == "Idle":
            return
        time = _datetime_windows_and_linux_compatible()
        if self.mode == "Bezier" or self.mode == "Path":
            os.makedirs(
                f"{self.autoencoder.p.main_path}/generated_paths/", exist_ok=True
            )
            fname = (
                f"{self.autoencoder.p.main_path}/generated_paths/generated_{time}.pdb"
            )
            with mda.Writer(fname) as w:
                for step in self.uni.trajectory:
                    w.write(self.uni.atoms)
            self.ax.set_title(
                f"Generated Path with {len(generated)} points saved at {fname}"
            )
        else:
            if "user_selected_points" not in self.trajs.CVs:
                self.ax.set_title("First set the points before writing them to disk.")
                return
            max_, fname = _unpack_cluster_info(
                self.trajs,
                self.autoencoder.p.main_path,
                self.tool,
                self.dummy_traj,
                self.align_string,
            )
            self.ax.set_title(f"Cluster {max_} saved at {fname}")

    def set_points(self):
        """Called when 'Set Points' is pressed."""
        if self.mode == "Idle":
            return
        if self.mode != "Idle":
            if "tool" not in self.__dict__.keys():
                self.ax.set_title(f"Tool {self.mode} not yet implemented.")
                return
            else:
                indices = self.accept()
        if self.mode == "Bezier" or self.mode == "Path":
            if np.unique(self.path_points, axis=0).shape[0] != 200:
                self.ax.set_title(
                    f"Tool {self.mode} returned not the requested unique points."
                )
                return
            self.dummy_traj = self.autoencoder.generate(
                self.path_points, backend="mdanalysis", top=self.top_index
            )
            self.view = ngl.show_mdanalysis(self.dummy_traj)
            if self.ball_and_stick:
                self.view.clear_representations()
                self.view.add_ball_and_stick()

            self.ax.set_title(
                f"Generated Path with {len(self.dummy_traj.trajectory)} points is accessible as InteractivePlotting.view."
            )
            return

        if indices is not None and self.mode != "Bezier" and self.mode != "Path":
            self.ax.set_title(
                f"Currently working on rendering the cluster. I'll let you know, when I'm finished."
            )
            indices = np.asarray(indices)

            # update user defined clustering
            col = "user_selected_points"
            if col not in self.trajs.CVs:
                _ = np.full(self.trajs.n_frames, -1)
                try:
                    _[indices] = 0
                except IndexError as e:
                    print(indices)
                    raise SystemExit from e
                self.trajs.load_CVs(_, col)
            else:
                _ = self.trajs.CVs[col]
                max_ = _.max()
                _[indices] = max_ + 1
                self.trajs.load_CVs(_, col)

            # change coloration of self.pts
            color_palette = sns.color_palette("Paired", self.trajs.CVs[col].max() + 1)
            cluster_colors = [
                (*color_palette[x], 1) if x >= 0 else (0.5, 0.5, 0.5, 0.01)
                for x in self.trajs.CVs[col]
            ]
            self.pts.set_color(cluster_colors)

            max_ = np.max(self.trajs.CVs[col])
            self.view, self.dummy_traj = self.cluster_building_fn(
                self.trajs,
                max_,
                nglview=True,
                shorten=True,
                stack_atoms=True,
                col=col,
                align_string=self.align_string,
                ball_and_stick=self.ball_and_stick,
            )
            if self.ball_and_stick:
                for i in range(len(self.dummy_traj)):
                    self.view.clear_representations(component=i)
                    self.view.add_ball_and_stick(component=i)
            self.ax.set_title(
                f"Cluster {max_} is accessible as InteractivePlotting.view."
            )

    def render_move(self):
        pass

    def lasso(self):
        self.tool = SelectFromCollection(self.ax, self.pts)

    def rectangle(self):
        self.tool = SelectFromCollection(self.ax, self.pts, selector=RectangleSelector)

    def ellipse(self):
        print("Ellipse not yet implemented")

    def polygon(self):
        textstr = "\n".join(
            (
                "Select points in the figure by enclosing them within a polygon.",
                # Press the 'esc' key to start a new polygon.
                "Try holding the 'shift' key to move all of the vertices.",
                "Try holding the 'ctrl' key to move a single vertex.",
            )
        )

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        self.manual_text = self.ax.text(
            0.05,
            0.95,
            textstr,
            transform=self.ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox=props,
        )
        self.tool = SelectFromCollection(self.ax, self.pts, selector=PolygonSelector)

    def path(self):
        pass

    def bezier(self):
        line = Line2D([], [], ls="--", c="#666666", marker="x", mew=2, mec="#204a87")
        self.ax.add_line(line)
        self.tool = BezierBuilder(line, self.ax)

    def accept(self):
        if "manual_text" in self.__dict__.keys():
            self.manual_text.set_visible(False)
            del self.manual_text
        if self.mode == "Bezier":
            self.path_points = copy.deepcopy(self.tool.ind)
        selected_indices = self.tool.ind
        self.tool.disconnect()
        return selected_indices

    @property
    def cluster_zoomed(self):
        col = "user_selected_points"
        if not col in self.trajs.df.keys():
            return
        max_ = np.max(self.trajs.df[col])
        _ = plot_cluster_zoomed(self.trajs, max_, col=col)
        return _

    @property
    def mode(self):
        return self.statusmenu.status
