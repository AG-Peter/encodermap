import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from itertools import cycle
from matplotlib.widgets import Lasso
import os
from .misc import create_dir, periodic_distance_np, sigmoid
import MDAnalysis as md
import datetime
from .backmapping import dihedral_backmapping
import matplotlib.pyplot as plt
import subprocess


class ManualPath(object):
    """
    ManualPath is a tool to manually select a path in a matplotlib graph.
    It supports two modes: "interpolated line", and "free draw".
    Press "m" to switch modes.

    In interpolated line mode click in the graph to add an additional way point.
    Press "delete" to remove the last way point.
    Press "d" to remove all way points.
    Press "enter" once you have finished your path selection.

    In free draw mode press and hold the left mouse button while you draw a path.

    Once the path selection is completed, the use_points method is called with the points on the selected path.
    You can overwrite the use_points method to do what ever you want with the points on the path.
    """

    def __init__(self, axe, n_points=200):
        """

        :param axe: matplotlib axe object for example from: fig, axe = plt.subplots()
        :param n_points: Number of points distributed on the selected path.
        """
        self.axe = axe
        self.canvas = axe.figure.canvas
        self.fig = axe.figure
        self.lasso = None
        self.n_interpolation_points = n_points

        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        control_line = Line2D([], [], ls='--', c='#666666',
                              marker='x', mew=2, mec='#204a87')
        self.control_line = self.axe.add_line(control_line)
        self.x_control = list(self.control_line.get_xdata())
        self.y_control = list(self.control_line.get_ydata())

        interpolated_line = Line2D([], [], c=self.control_line.get_markeredgecolor())
        self.interpolated_line = self.axe.add_line(interpolated_line)

        self.modes = cycle([self._add_point_interp, self._free_draw])
        self.mode = next(self.modes)
        self.click_cid = self.canvas.mpl_connect('key_press_event', self._on_key)
        self.click_cid = self.canvas.mpl_connect('button_press_event', self._on_click)
        self.draw_cid = self.canvas.mpl_connect('draw_event', self._grab_background)

    def _on_click(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.axe:
            return
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        if event.button == 1:
            self.mode(event)

    def _on_key(self, event):
        if event.key == "m":
            self.mode = next(self.modes)
            return
        if event.key == "enter":
            points = np.array(self.interpolated_line.get_data()).T
            self._reset_lines()
            self.use_points(points)
            self._grab_background()

            return
        if event.key == "d":
            self._reset_lines()
            return
        if event.key == "delete":
            del self.x_control[-1]
            del self.y_control[-1]
            self._update_interp()

    def _free_draw(self, event):
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self._free_draw_callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def _free_draw_callback(self, verts):
        points = np.array(verts)
        self.use_points(points)
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def _add_point_interp(self, event):
        self.x_control.append(event.xdata)
        self.y_control.append(event.ydata)
        self._update_interp()

    def _update_interp(self):
        self.control_line.set_data(self.x_control, self.y_control)
        x_i, y_i = self._interpolate(self.x_control, self.y_control)
        x_i, y_i = self._interpolate(x_i, y_i)  # second iteration makes points more evenly spaced
        self.interpolated_line.set_data(x_i, y_i)
        self._update_lines()

    def _interpolate(self, x, y):
        cumulative_distances = [0]
        for i in range(1, len(x)):
            dist = ((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) ** 0.5
            cumulative_distances.append(cumulative_distances[-1] + dist)
        interp_i = np.linspace(0, max(cumulative_distances), self.n_interpolation_points)
        try:
            x_i = interp1d(cumulative_distances, x, kind='cubic')(interp_i)
            y_i = interp1d(cumulative_distances, y, kind='cubic')(interp_i)
        except ValueError:
            try:
                x_i = interp1d(cumulative_distances, x, kind='linear')(interp_i)
                y_i = interp1d(cumulative_distances, y, kind='linear')(interp_i)
            except ValueError:
                x_i = []
                y_i = []
        return x_i, y_i

    def use_points(self, points):
        """
        Overwrite this method to use the selected points in any way you like.

        For Example:

        >>> class MyManualPath(ManualPath):
        >>>     def use_points(self, points):
        >>>         print(points)

        :param points: numpy array with points from the manual path selection
        :return: None
        """
        self.axe.plot(points[:, 0], points[:, 1], linestyle="", marker=".")
        raise NotImplementedError("has to be implemented in subclass")

    def _grab_background(self, event=None):
        """
        When the figure is resized, hide the points, draw everything,
        and update the background.
        """
        # Thanks to: https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        self.canvas.mpl_disconnect(self.draw_cid)
        self.interpolated_line.set_visible(False)
        self.control_line.set_visible(False)
        self.canvas.draw()

        # With most backends (e.g. TkAgg), we could grab (and refresh, in
        # self.blit) self.ax.bbox instead of self.fig.bbox, but Qt4Agg, and
        # some others, requires us to update the _full_ canvas, instead.
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.interpolated_line.set_visible(True)
        self.control_line.set_visible(True)
        self.canvas.draw()
        self.draw_cid = self.canvas.mpl_connect('draw_event', self._grab_background)

    def _update_lines(self):
        """
        Efficiently update the figure, without needing to redraw the
        "background" artists.
        """
        self.fig.canvas.restore_region(self.background)
        self.axe.draw_artist(self.interpolated_line)
        self.axe.draw_artist(self.control_line)
        self.canvas.draw()  # Todo: canvas.draw() works in jupyter notebook but canvas.update() outside of notebook
        self.canvas.flush_events()

    def _reset_lines(self):
        self.interpolated_line.set_data([[], []])
        self.control_line.set_data([[], []])
        self._update_lines()

        self.x_control = []
        self.y_control = []


class PathGenerateDihedrals(ManualPath):
    """
    This class inherits from :py:class:`encodermap.plot.ManualPath`.
    The points from a manually selected path are fed into the decoder part of a given autoencoder.
    The output of the autoencoder is used as phi psi dihedral angles to reconstruct protein conformations
    based on the protein structure given with pdb_path.
    Three output files are written for each selected path:
    points.npy, generated.npy and generated.pdb which contain:
    the points on the selected path, the generated output of
    the autoencoder, and the generated protein conformations respectively.
    Keep in mind that backbone dihedrals are not sufficient to describe a protein conformation completely.
    Usually the backbone is reconstructed well but all side chains are messed up.
    """

    def __init__(self, axe, autoencoder, pdb_path, save_path=None, n_points=200):
        """

        :param axe: matplotlib axe object for example from: fig, axe = plt.subplots()
        :param autoencoder: :py:class:`encodermap.autoencoder.Autoencoder` which was trained on protein dihedral
            angles. The dihedrals have to be order starting from the amino end.
            First all phi angles then all psi angles.
        :param pdb_path: Path to a protein data bank (pdb) file of the protein
        :param save_path: Path where outputs should be written
        :param n_points: Number of points distributed on the selected path.
        """
        super(PathGenerateDihedrals, self).__init__(axe, n_points=n_points)

        self.autoencoder = autoencoder
        self.pdb_path = pdb_path

        if save_path:
            self.save_path = save_path
        else:
            self.save_path = autoencoder.p.main_path

    def use_points(self, points):
        current_save_path = create_dir(os.path.join(self.save_path, "generated_paths",
                                                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.axe.plot(points[:, 0], points[:, 1], linestyle="", marker=".")
        np.save(os.path.join(current_save_path, "points"), points)
        generated = self.autoencoder.generate(points)
        np.save(os.path.join(current_save_path, "generated.npy"), generated)

        universe = dihedral_backmapping(self.pdb_path, generated)
        output_pdb_path = os.path.join(current_save_path, "generated.pdb")
        with md.Writer(output_pdb_path) as w:
            for step in universe.trajectory:
                w.write(universe.atoms)


class PathGenerateCartesians(ManualPath):
    """
    This class inherits from :py:class:`encodermap.plot.ManualPath`.
    The points from a manually selected path are fed into the decoder part of a given autoencoder.
    The output of the autoencoder is used as phi psi dihedral angles to reconstruct protein conformations
    based on the protein structure given with pdb_path.
    Three output files are written for each selected path:
    points.npy, generated.npy and generated.pdb which contain:
    the points on the selected path, the generated output of
    the autoencoder, and the generated protein conformations respectively.
    Keep in mind that backbone dihedrals are not sufficient to describe a protein conformation completely.
    Usually the backbone is reconstructed well but all side chains are messed up.
    """

    def __init__(self, axe, autoencoder, mol_data, save_path=None, n_points=200, vmd_path=""):
        """

        :param axe: matplotlib axe object for example from: fig, axe = plt.subplots()
        :param autoencoder: :py:class:`encodermap.autoencoder.Autoencoder` which was trained on protein dihedral
            angles. The dihedrals have to be order starting from the amino end.
            First all phi angles then all psi angles.
        :param pdb_path: Path to a protein data bank (pdb) file of the protein
        :param save_path: Path where outputs should be written
        :param n_points: Number of points distributed on the selected path.
        """
        super().__init__(axe, n_points=n_points)

        self.autoencoder = autoencoder
        self.mol_data = mol_data
        self.vmd_path = vmd_path

        if save_path:
            self.save_path = save_path
        else:
            self.save_path = autoencoder.p.main_path

    def use_points(self, points):
        current_save_path = create_dir(os.path.join(self.save_path, "generated_paths",
                                                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.axe.plot(points[:, 0], points[:, 1], linestyle="", marker=".")
        np.save(os.path.join(current_save_path, "points"), points)
        try:
            dihedrals, cartesians = self.autoencoder.generate(points)
        except ValueError:
            angles, dihedrals, cartesians = self.autoencoder.generate(points)
        np.save(os.path.join(current_save_path, "generated_dihedrals.npy"), dihedrals)
        np.save(os.path.join(current_save_path, "generated_cartesians.npy"), cartesians)

        self.mol_data.write(current_save_path, cartesians, only_central=False)
        if self.vmd_path:
            cmd = "{} {} {}".format(self.vmd_path,
                                    "generated.pdb",
                                    "generated.xtc")
            print(cmd)
            process = subprocess.Popen(cmd, cwd=current_save_path, shell=True, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE)
            process.stdin.write(b"animate delete beg 0 end 0 skip 0 0\n")
            process.stdin.flush()


def distance_histogram(data, periodicity, sigmoid_parameters, axes=None, low_d_max=5):
    """
    Plots the histogram of all pairwise distances in the data.
    If sigmoid parameters are given it also shows the sigmoid function and its normalized derivative.

    :param data: each row should contain a point in a number_of _columns dimensional space.
    :param periodicity: Periodicity of the data. use float("inf") for non periodic data
    :param sigmoid_parameters: tuple (sigma, a, b)
    :param axe: matplotlib axe object ore None. If None a new figure is generated.
    :return:
    """
    vecs = periodic_distance_np(np.expand_dims(data, axis=1), np.expand_dims(data, axis=0), periodicity)
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
    counts, edges, patches = axe2.hist(dists, bins="auto", density=True, edgecolor='black')
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

    x = np.linspace(0, low_d_max, 1000)
    y = sigmoid(x, *sigmoid_parameters[3:])
    dy = np.diff(y)
    dy_norm = dy / max(dy)
    idx = np.argmin(np.abs(np.expand_dims(edges_sig, axis=1) - np.expand_dims(y, axis=0)), axis=1)
    edges_x = x[idx]

    axes[1].plot(x, y, color="C1", label="sigmoid")

    axes[1].legend()
    axes[1].set_xlabel("distance")
    axes[1].set_ylim((0, 1))
    for i in range(len(edges)):
        if edges_x[i] != edges_x[-1]:
            axes[1].annotate('', xy=(edges[i], 0), xytext=(edges_x[i], 0), xycoords=axes[0].transData,
                             textcoords=axes[1].transData,
                             arrowprops=dict(facecolor='black', arrowstyle='-', clip_on=False))

    return axes
