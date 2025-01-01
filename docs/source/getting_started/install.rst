.. _getting_started/install:

Installation
------------

Clone
~~~~~

While EncoderMap can be installed from the Python Package Index via pip, it is recommended to install it from source, so you get all tutorials.

.. tab-set::

    .. tab-item:: Bash

        .. code-block:: bash

            $ git clone --branch latest https://github.com/AG-Peter/encodermap

    .. tab-item:: Powershell

        .. code-block:: powershell

            PS C:\> git clone --branch latest https://github.com/AG-Peter/encodermap

Install
~~~~~~~

You can then change into the new directory and install the required packaged.

.. tab-set::

    .. tab-item:: Bash

        .. code-block:: bash

            $ cd encodermap
            $ pip install -r requirements.txt

    .. tab-item:: Powershell

        .. code-block:: powershell

            PS C:\> cd encodermap
            PS C:\> pip install -r requirements.txt

Installing optional MD dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan to use EncoderMap on MD data, we recommend installing the additional MD requirements.

.. tab-set::

    .. tab-item:: Bash

        .. code-block:: bash

            $ pip install -r md_requirements.txt

    .. tab-item:: Powershell

        .. code-block:: powershell

            PS C:\> pip install -r md_requirements.txt


Installing EncoderMap
~~~~~~~~~~~~~~~~~~~~~

Then you can install EncoderMap into your virtual Environments.

.. tab-set::

    .. tab-item:: Bash

        .. code-block:: bash

            $ pip install .

    .. tab-item:: Powershell

        .. code-block:: powershell

            PS C:\> pip install .

Coming from EncoderMap 2.0?
---------------------------

.. button-ref:: tf1_tf2_changes
    :color: primary
    :expand:

    Read about the EncoderMap 1.0/2.0 compatibility layer.

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


Original README
---------------

 .. card:: Review the original README.md
    :link: getting_started/link_to_readme
    :link-type: ref

    It contains information on using GradientChromatography to get your own project started.
