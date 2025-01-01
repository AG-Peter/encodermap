.. _user_guide/index:

User Guide
==========

The user guide contains the top-level classes and concepts of EncoderMap.

Working with MD data
--------------------

These classes offer new ways of working with large quantities of inhomogeneous molecular dynamics data and implement lazy loading of coordinates and other collective variables (see `Featurization`_). The API function `encodermap.load` returns either of the new trajectory classes.

.. grid:: 2

    .. grid-item::

        .. button-ref:: singletraj
           :color: primary
           :expand:
           :class: single-traj-btn

            SingleTraj

    .. grid-item::

        .. button-ref:: trajensemble
            :color: primary
            :expand:
            :class: traj-ensemble-btn

            TrajEnsemble


Neural network model building and training
------------------------------------------

These classes build `tf.keras.Model` models which will be trained and `tf.data.Dataset` datasets which will be used for training. The easiest pipelines provide the `__init__` method of these classes with an numpy.ndarray as data and a `encodermap.parameters.Parameters` class (more on that later) as parameters. That's all you need to use these classes.

.. code-block:: python

    import encodermap as em
    import numpy as np

    # create some random data 100 observations with 10 features
    data = np.random.random((100, 10))

    # create the parameters
    p = em.Parameters()
    emap = em.EndoerMap(p, data)

    # create lowd by calling the encoder
    lowd = emap.encode(data)

    # create highd by callind the decoder
    highd = em.decode(lowd)

To learn more about these classes, visit:

.. button-ref:: autoencoder_classes
    :color: primary
    :expand:
    :class: autoencoder-classes-btn

    Autoencoder classes

Parameters
----------

These classes are containers for parameters and allow an easy way to adjust the training of the neural network models. They can be loaded from and saved to .json files. They support indexing via keys and dot notation:

.. code-block:: python

    import encodermap as em

    # setting parameters at instantiation
    p = em.Parameters(center_cost_scale=2.5)

    # setting some other parameters
    p.auto_cost_variant = "mean_square"

    # print a pretty-formatted table of the parameters
    print(p.parameters)

To learn more about these classes and also what parameters affect what in EncoderMap, visit:

.. button-ref:: parameter_classes
    :color: primary
    :expand:
    :class: parameter-classes-btn

    Parameter classes

Featurization
-------------

Accompanying the new classes for MD trajectories, the featurization of EncoderMap.

.. button-ref:: featurization
    :color: primary
    :expand:
    :class: featurization-btn

    Featurization

Lower-level modules
-------------------

Models
~~~~~~

The models module is what handles the creation of the tensorflow models, using custom layers, implementing custom learning, etc.

.. button-ref:: models
    :color: primary
    :expand:
    :class: models-btn

    Models

Layers
~~~~~~

EncoderMap implements some custom tensorflow layers, that are documented here.

.. button-ref:: layers
    :color: primary
    :expand:
    :class: layers-btn

    Layers

Loss functions
~~~~~~~~~~~~~~

You can also write custom loss functions or experiment with EncoderMap's loss functions.

.. button-ref:: loss_functions
    :color: primary
    :expand:
    :class: loss-functions.btn

    Loss functions


.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Trajectory Classes

    singletraj
    trajensemble
    featurization


.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Neural Network Classes

    autoencoder_classes
    parameter_classes


.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Models, Layers, Losses

    models
    layers
    loss_functions
