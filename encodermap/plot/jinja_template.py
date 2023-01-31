# -*- coding: utf-8 -*-
# encodermap/plot/jinja_template.py
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
"""This is a template for a README.md generated when a user writes a cluster to disk."""

template = """#  Cluster {{cluster_id}} generated at {{now}}

## What just happened?

You either selected a cluster with the `InteractivePlotting` class of `encodermap` our you called the `_unpack_cluster_info()` function from `encodermap.plot.utils`. Many files have been put into a directory at {{cluster_abspath}} which can be used to rebuild the cluster. The cluster you selected has been assigned the number {{cluster_id}}. If your cluster number is 0, your cluster is the first selected cluster of these MD trajectories (outliers are assigned -1). If your cluster has a number different than 0, you have selected another cluster and the cluster_membership is given by this unique identifier.

Here is a general rundown of the files created:

## {{parents_trajs}}

This plain text document contains the absolute paths to all trajectory files, their corresponding topology files and their corresponding `common_str`, that were considered during the clustering. Some of the trajectory files here might not take part in the actual cluster, but they are here in this file nonetheless. You can reload the trajectories with the `from_textfile()` alternative constructor of the `TrajEnsemble` class.

```python
import encodermap as em
trajs = em.TrajEnsemble.from_textfile('{{parents_trajs}}')
```

## {{pdb_name}}

This file contains ca. 10 frames. These 10 frames were selected from the original {{cluster_n_points}} points inside the cluster. By evenly slicing it (That's why it is only roughly 10 structures. Sometimes its more). You can load this pdb whichever way you like and render a nice image of the cluster.

## Other pdb and xtc files

The other pdb and xtc files contain data to rebuild not only the ca. 10 frames from the pdb, but the whole cluster. They are enumerated the same way they are enumerated in {{parents_trajs}}. The fille `cluster_id_{{cluster_id}}_traj_0.xtc` corresponds to `cluster_id_{{cluster_id}}_start_traj_0_from_{{basename}}.pdb`, `cluster_id_{{cluster_id}}_traj_1.xtc` corresponds to `cluster_id_{{cluster_id}}_start_traj_1_from_{{basename}}.pdb` and so on.

## {{lowd_npy_name}}

A 2D numpy array with the same number of points, as there are frames in the `cluster_id_{{cluster_id}}_traj_X.xtc` files combined. This is the low-dimensional representation of this whole cluster.

## {{indices_npy_name}}

This file can be used to rebuild the clustering of the trajectories like so:

```python
>>> import encodermap as em
>>> import numpy as np
>>> trajs = em.TrajEnsemble.from_textfile('{{parents_trajs}}')
>>> cluster_membership = np.full(trajs.n_frames, -1) 		# fill array with -1, meaning outliers
>>> indices = np.load('{{indices_npy_name}}')				    # load the indices
>>> cluster_membership[indices] = {{cluster_id}}			# set the cluster number of the indices
>>> trajs.load_CVs(cluster_membership, 'cluster_membership') # load the cluster membership as collectvie variables
>>> traj_indices = trajs.id[indices] 					   # more on this line in a separate section
>>> cluster_trajs = trajs[traj_indices]
```

## {{pdb_origin_names}}

This file is only created, when the structures inside the cluster have a different number of atoms and thus, can not be loaded with the same topology. This plain text file contains information from where the pdb files were copied. This might only be useful in very niche scenarios.

## {{csv_name}}

This .csv table contains Info about every point inside the cluster. Its columns give the following information:

| trajectory file                       | Contains the trajectory data (file formats such as .xtc, .dcd, .h5). |
| ------------------------------------- | ------------------------------------------------------------ |
| topology file                         | Contains the topology of the file (i.e. atom types, masses, residues) (file formats such as .pdb, .gro, .h5). Some trajectory files (.h5) might also contain the topology. In that case `trajectory file` and `topology` file are identical. |
| frame number                          | The number of the frame of the `trajectory file`. If you index your trajectories by frame number use this number to reload this specific trajectory frame. `import mdtraj as md; frame = md.load_frame(trajectory_file, index=frame, top=topology_file)`<br />or<br />`imprt MDAnalysis as mda; frame = mda.Universe(topology_file, trajectory_file).trajectory[frame]` |
| time                                  | The time of the frame. This can be used for time-based indexing of trajectories. `gmx trjconv -f $traj_file -s $top_file -dump $time` |
| cluster id                            | The id of the cluster. This column is identical in the whole csv file but can be used to merge multiple csv files to analyye multiple clusters at once. |
| trajectory number                     | The number of the trajectory in the full dataset. This corresponds to the line number in the file {{parents_trajs}}. If many trajectories have been loaded, the first trajectory is 0, and so on. If only one trajectory is loaded, its `trajectory number` might also be `None`. |
| unique id in set of {{n_trajs}} trajs | This is an integer number with a unique identifier of every frame of every trajectory given in {{parents_trajs}}. The frames of trajectory number 0 are enumerated starting from 0, 1, ... n. The frames of the next trajectory (trajectory number 1) are enumerated n + 1, n + 2, ... n + m. The frames of traj 3 are enumerated as n + m + 1, n + m + 2, and so on. This way every frame gets a unique integer identifier. |

## {{selector_npy_name}}

This is a 2D numpy array of the points of the Selector used. The selector is a matplotlib.widget that can interactively select points in a 2D scatter plot. In `encodermap` 4 selectors are available:

- Rectangle: For Rectangle the Selector will contain 4 points. The xy coordinates of the corners of the rectangle in data coordinates.
- Polygon: Similar to Rectangle a collection of points. The first and last point are identical.
- Ellipse: A collection of points describing the outline of the Ellipse.
- Lasso: A collection of points following a free-hand drawn shape.

## {{current_clustering}}

This is a numpy array containing the cluster numbers of all previously selected clusters. If this cluster has a cluster id of 0, this array will only contain 0s and -1s and will be the same lengths as there are frames in the analyzed trajectories.

```python
>>> import encodermap as em
>>> import numpy as np
>>> trajs = em.TrajEnsemble.from_textfile('{{parents_trajs}}')
>>> current_clustering = np.load('{{current_clustering}}')
>>> len(current_clustering) == trajs.n_frames
True
```

If this cluster has a higher cluster id all previously selected clusters can be accessed with this array:

```python
>>> import encodermap as em
>>> import numpy as np
>>> trajs = em.TrajEnsemble.from_textfile('{{parents_trajs}}')
>>> current_clustering = np.load('{{current_clustering}}')	# fill array with -1, meaning outliers
>>> trajs.load_CVs(current_clustering, 'cluster_membership') # load the cluster membership as collectvie variables
>>> indices_some_other_cluster = np.where(trajs.cluster_membership == 2)[0]
>>> traj_indices = trajs.id[indices_some_other_cluster] 	 # more on this line in a separate section
>>> cluster_trajs = trajs[traj_indices]
>>> len(traj_indices == cluster_trajs.n_frames)
True
```

## {{png_name}}

This is just an image. Here it is

![Cluster Image]({{png_name}})

## Why the `trajs.id[indices]` part?

This comes down to the question of: What should be returned if an `TrajEnsemble` object is indexed via a list or numpy array. For this we will first fall back and try to figure out, what should happen if the `TrajEnsemble` class is indexed via a single integer. The most sensical way would be that you get an `SingleTraj` class indexed by this integer. Consider this example:

```python
>>> import encodermap as em
>>> traj1 = em.SingleTraj('path/to/traj1.xtc', top='path/to/top1.pdb')
>>> print(traj1.basename)
traj1
>>> traj2 = em.SingleTraj('path/to/traj2.xtc', top='path/to/top2.pdb')
>>> traj3 = em.SingleTraj('path/to/traj3.xtc', top='path/to/top3.pdb')
>>> trajs = em.TrajEnsemble([traj1, traj2, traj3])
>>> print([t.basename for t in trajs])
['traj1', 'traj2', 'traj3']
>>> integer_indexing = trajs[2]
>>> print(integer_indexing == traj3)
True
>>> print(integer_indexing.basename)
traj3
```

Using a list of int or a numpy array of int thus returns a new `TrajEnsemble` class, but with the `SingleTraj` classes indexed by the ints. Consider this example:

```python
>>> import encodermap as em
>>> traj1 = em.SingleTraj('path/to/traj1.xtc', top='path/to/top1.pdb')
>>> traj2 = em.SingleTraj('path/to/traj2.xtc', top='path/to/top2.pdb')
>>> traj3 = em.SingleTraj('path/to/traj3.xtc', top='path/to/top3.pdb')
>>> trajs = em.TrajEnsemble([traj1, traj2, traj3])
>>> print([t.basename for t in trajs])
['traj1', 'traj2', 'traj3']
>>> indices = [1, 2]
>>> new_trajs = trajs[indices]
>>> print([t.basename for t in new_trajs])
['traj2', 'traj3']
```

And finally we arrived at the point of using the `traj_indices = trajs.id[indices]` syntax in section {{indices_npy_name}}. This will return a numpy array with ndim = 2 with which you can index single frames. Let's say we want to have a `TrajEnsemble` class, but only with frame 10 of traj 0, frame 20 of traj 2 and frame 30 of traj 3. Maybe we will also add the frames 2 to 5 from traj 2. The syntax will be as follows:

```python
>>> import encodermap as em
>>> traj1 = em.SingleTraj('path/to/traj1.xtc', top='path/to/top1.pdb')
>>> traj2 = em.SingleTraj('path/to/traj2.xtc', top='path/to/top2.pdb')
>>> traj3 = em.SingleTraj('path/to/traj3.xtc', top='path/to/top3.pdb')
>>> trajs = em.TrajEnsemble([traj1, traj2, traj3])
>>> print([t.basename for t in trajs])
['traj1', 'traj2', 'traj3']
>>> print([t.n_frames for t in trajs])
[100, 100, 100]
>>> indices = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [2, 2].
    [2, 3],
    [2, 4],
    [2, 5]
])
>>> new_trajs = trajs[indices]
>>> print([t.basename for t in new_trajs])
['traj1', 'traj2', 'traj3', 'traj2', 'traj2', 'traj2', 'traj2']
>>> print([t.n_frames for t in new_trajs])
[1, 1, 1, 1, 1, 1, 1]
>>> print(set([type(t) for t in new_trajs]))
[encodermap.SingleTraj]
```

So all in all a 1D array of ints indexes single trajectories a 2D array of ints indexes trajs and frames.

## What is a `common_str`?

Encodermap's `TrajEnsemble` and `SingleTraj` classes contain a class variable called `comon_str`. The common string is a way to order trajectory files from the same topology. This comes in handy, when you run many simulations with the same topology and want to compare them to simulations with a similar, but different topology. Let's consider this scenario. You run simulations of short peptides AFFA and FAAF. Both peptides have the same number of atoms but different topologies. Somehow they still share some joint phase space and can be considered similar to some regards. You set up some simulations from your AFFA.pdb and FAAF.pdb files and them Now you have these files to consider:

- AFFA.pdb: AFFA_traj1.xtc, AFFA_traj2.xtc, AFFA_traj3.xtc
- FAAF.pdb: FAAF_traj.xtc

And you want to compare them. For this you need to assign the pdb files to the corresponding xtc files. Luckily you chosen a naming scheme that lets you group them by the substrings AFFA and FAAF. You can load all trajectories with encodermap using the `TrajEnsemble` class.

```python
import encodermap as em
trajs = em.TrajEnsemble(
	[AFFA_traj1.xtc, AFFA_traj2.xtc, AFFA_traj3.xtc, FAAF_traj.xtc],
    [AFFA.pdb, FAAF.pdb],
    common_str=['FAAF', 'AFFA']
)
```

## Rendering this document

If you don't like to view plain markdown files with a text-viewer there are many viewers available, that are able to render markdown nicely. I am currently using typora:

https://typora.io/

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
