# -*- coding: utf-8 -*-
# encodermap/plot/plotting.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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
    * Add interactive Plotting
    * Find a way to use interactive plotting with less points but still cluster everything.

"""

##############################################################################
# Imports
##############################################################################


from __future__ import annotations

import os
import shutil
import subprocess
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .._optional_imports import _optional_import
from ..encodermap_tf1.misc import periodic_distance_np, sigmoid
from ..misc.clustering import gen_dummy_traj, rmsd_centroid_of_cluster

################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")


##############################################################################
# Globals
##############################################################################

__all__ = ["distance_histogram"]

##############################################################################
# Utilities
##############################################################################


def distance_histogram(
    data, periodicity, sigmoid_parameters, axes=None, low_d_max=5, bins="auto"
):
    """
    Plots the histogram of all pairwise distances in the data.
    It also shows the sigmoid function and its normalized derivative.

    Args:
        data (np.ndarray): 2-dimensional numpy array. Columns should iterate over the dimensions of the datapoints,
            i.e. the dimensionality of the data. The rows should iterate over datapoints.
        periodicity (float): Periodicity of the data. Use float("inf") for non-periodic data.
        sigmoid_parameters (tuple): Tuple of sketchmap sigmoid parameters in shape (sigma, a, b).
        axes (Union[np.ndarray, None], optional): A numpy array of two matplotlib.axes objects or None. If None is provided,
            the axes will be created. Defaults to None.
        low_d_max (int, optional): Upper limit for plotting the low_d sigmoid. Defaults to 5.
        bins (Union[str, int]. optional): Number of bins for histogram. Use 'auto' to let matplotlib decide how
            many bins to use. Defaults to 'auto'.


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
                arrowprops=dict(facecolor="black", arrowstyle="-", clip_on=False),
            )
    axes[0].figure.tight_layout()
    return axes[0], axe2, axes[1]


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

    Examples:
        >>> pdb_file = '/path/to/pdbfile.pdb'
        >>> image = render_vmd(pdb_file, scale=2)
        >>> plt.imshow(image)

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
                    import warnings

                    warnings.warn(
                        "Rendering multiple frames with STL may lead to undesired results. Instead of yielding the union of all single-frame surfaces, you will get a mishmash of all surfaces with intersection faces etc."
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
