.. _getting_started:

Getting started
===============

Coming from EncoderMap 2.0?
---------------------------

.. button-ref:: tf1_tf2_changes
    :color: primary
    :expand:

    Make EncoderMap 2.0 code work with the new EncoderMap

Installation
------------

This version of EncoderMap now supports Tensorflow 2. The Tensorflow 1 code is still usable :ref:`(more on that in Changes between TF1 and TF2) <tf1_tf2_changes>`.

Encodermap can be installed via pip.

.. code-block:: bash

  $ pip3 install --user encodermap

If you are in a virtual environment run

.. code-block:: bash

  $ pip3 install encodermap

If you are building from source you can run

.. code-block:: bash

  $ git clone https://github.com/AG-Peter/encoder_map_private.git
  $ cd encoder_map_private
  $ pip3 install .

If you don't want pip to copy the encodermap files into your python site-packages directory you can install the package in editable mode which allows you to make changes in the same directory you cloned EncoderMap into and use them in your python environment.

.. code-block:: bash

  $ pip3 install -e .

Minimal Example
---------------

This example shows how to use EncoderMap to project points from a high dimensional data set to a low dimensional space using the default parameters. In the data set, each row should represent one data point and the number of columns should be equal to the number of dimensions.

.. code-block:: python

  import encodermap as em
  import numpy as np

  high_dimensional_data = np.loadtxt("my_high_d_data.csv", delimiter=",")
  parameters = em.Parameters()

  e_map = em.EncoderMap(parameters, high_dimensional_data)
  e_map.train()

  low_dimensional_projection = e_map.encode(high_dimensional_data)

The resulting `low_dimensional_projection` array has the same number of rows as the `high_dimensional_data` but the number of columns is two as high dimensional points are projected to a 2d space with default settings.

In contrast to many other dimensionality reduction algorithms EncoderMap does not only allow to efficiently project form a high dimensional to a low dimensional space. Also the generation of new high dimensional points for any given points in the low dimensional space is possible:

.. code-block:: python

  low_d_points = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]])
  newly_generated_high_d_points = e_map.generate(low_d_points)


.. toctree::
    :maxdepth: 2
    :glob:
