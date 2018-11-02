"""
EncoderMap
Copyright (C) 2018  Tobias Lemke

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from itertools import cycle
from matplotlib.widgets import Lasso
import os
from .misc import create_dir
import MDAnalysis as md
import datetime
from .dihedral_backmapping import dihedral_backmapping


class ManualPath(object):
    def __init__(self, axe, n_points=200):
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

        self.modes = cycle([self.add_point_interp, self.free_draw])
        self.mode = next(self.modes)
        self.click_cid = self.canvas.mpl_connect('key_press_event', self.on_key)
        self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.draw_cid = self.canvas.mpl_connect('draw_event', self.grab_background)

    def on_click(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.axe:
            return
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        if event.button == 1:
            self.mode(event)

    def on_key(self, event):
        if event.key == "m":
            self.mode = next(self.modes)
            return
        if event.key == "enter":
            points = np.array(self.interpolated_line.get_data()).T
            self.reset_lines()
            self.use_points(points)
            self.grab_background()

            return
        if event.key == "d":
            self.reset_lines()
            return
        if event.key == "delete":
            del self.x_control[-1]
            del self.y_control[-1]
            self.update_interp()

    def free_draw(self, event):
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.free_draw_callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def free_draw_callback(self, verts):
        points = np.array(verts)
        self.use_points(points)
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def add_point_interp(self, event):
        self.x_control.append(event.xdata)
        self.y_control.append(event.ydata)
        self.update_interp()

    def update_interp(self):
        self.control_line.set_data(self.x_control, self.y_control)
        x_i, y_i = self.interpolate(self.x_control, self.y_control)
        x_i, y_i = self.interpolate(x_i, y_i)  # second iteration makes points more evenly spaced
        self.interpolated_line.set_data(x_i, y_i)
        self.update_lines()

    def interpolate(self, x, y):
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
        self.axe.plot(points[:, 0], points[:, 1], linestyle="", marker=".")
        raise NotImplementedError("has to be implemeted in subclass")

    def grab_background(self, event=None):
        """
        When the figure is resized, hide the points, draw everything,
        and update the background.
        Thanks to: https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        """
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
        self.draw_cid = self.canvas.mpl_connect('draw_event', self.grab_background)

    def update_lines(self):
        """
        Efficiently update the figure, without needing to redraw the
        "background" artists.
        """
        self.fig.canvas.restore_region(self.background)
        self.axe.draw_artist(self.interpolated_line)
        self.axe.draw_artist(self.control_line)
        self.canvas.update()
        self.canvas.flush_events()

    def reset_lines(self):
        self.interpolated_line.set_data([[], []])
        self.control_line.set_data([[], []])
        self.update_lines()

        self.x_control = []
        self.y_control = []


class PathGenerate(ManualPath):

    def __init__(self, axe, autoencoder, pdb_path, save_path=None, n_points=200):
        super(PathGenerate, self).__init__(axe, n_points=n_points)

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
