.. _notebooks/index:

Notebook Gallery
================

Here, you can find static renders of EncoderMap's example notebooks. You can run them interactively on Google Colab or MyBinder. You can also run them on your local machine by cloning EncoderMap's repository.

.. code-block::
    :bash:

    $ git clone https://github.com/AG-Peter/encodermap && encodermap
    $ pip install -r requirements.txt && pip install md_requirements.txt
    $ cd tutorials
    $ jupyter-notebook

Starter Notebooks
-----------------

The starter notebooks help you with your first steps with EncoderMap.

.. grid:: 3

    .. grid-item-card:: Basic Cube Example
        :margin: 2 0 0 0
        :img-bottom: ../_static/starter_nb_01_basic_thumbnail.png
        :img-alt: A cube with colored vertices.
        :link: starter_nb/starter_nb_01_basic
        :link-type: doc

        Get started with EncoderMap

    .. grid-item-card:: Advanced Asp7
        :margin: 2 0 0 0
        :img-bottom: ../_static/starter_nb_02_advanced_thumbnail.png
        :img-alt: the Mybinder logo
        :link: starter_nb/starter_nb_02_advanced
        :link-type: doc

        Advanced EncoderMap usage with MD data

    .. grid-item-card:: Your Data
        :margin: 2 0 0 0
        :img-bottom: ../_static/starter_nb_03_your_data_thumbnail.png
        :img-alt: the Mybinder logo
        :link: starter_nb/starter_nb_03_your_data
        :link-type: doc

        Upload your own data and use this notebook


Notebooks MD
------------

The MD notebooks contain more detailed descriptions of how EncoderMap deals with MD data. It helps you in saving and loading large MD datasets and using them to train EncoderMap. It also helps you in understanding the terms `feature space` and `collective variable`.

.. grid:: 1

    .. grid-item-card:: Trajectory Ensembles
        :margin: 2 0 0 0
        :img-bottom: ../_static/md_nb_01_traj_ensemble_thumbnail.png
        :img-alt: Img description.
        :link: md_nb/md_nb_01_traj_ensemble
        :link-type: doc

        Trajectory Ensembles


Notebooks Customization
-----------------------

These notebooks help you in customizing EncoderMap. These tools can assist you in understanding how EncoderMap trains on your data. Furthermore, you will learn how to implement new cost functions and vary the training rate of the Neural Network.

.. grid:: 4

    .. grid-item-card:: Custom Scalars
        :margin: 2 0 0 0
        :img-bottom: ../_static/customization_nb_01_scalars_thumbnail.png
        :img-alt: Img description.
        :link: customization_nb/customization_nb_01_scalars
        :link-type: doc

        Monitor in TensorBoard

    .. grid-item-card:: Custom Loss
        :margin: 2 0 0 0
        :img-bottom: ../_static/customization_nb_02_loss_thumbnail.png
        :img-alt: Img description.
        :link: customization_nb/customization_nb_02_loss
        :link-type: doc

        Add new loss functions

    .. grid-item-card:: Custom Images
        :margin: 2 0 0 0
        :img-bottom: ../_static/customization_nb_03_images_thumbnail.png
        :img-alt: Img description.
        :link: customization_nb/customization_nb_03_images
        :link-type: doc

        Write images to TensorBoard

    .. grid-item-card:: Learning Rate Scheduler
        :margin: 2 0 0 0
        :img-bottom: ../_static/customization_nb_04_lr_thumbnail.png
        :img-alt: Img description.
        :link: customization_nb/customization_nb_04_lr
        :link-type: doc

        Adjust the learning rate


Notebooks Advanced
------------------

These notebooks contain more advanced training examples, like when you want to train with EncoderMap's sparse networks, that allow simultaneous training of various protein topologies, that could stem from mutations or protein families.

.. grid:: 1

    .. grid-item-card:: Different Topologies
        :margin: 2 0 0 0
        :img-bottom: ../_static/intermediate_nb_01_sparse_thumbnail.png
        :img-alt: Img description.
        :link: intermediate_nb/intermediate_nb_01_sparse
        :link-type: doc

        Training EncoderMap on inhomogeneous MD datasets


.. toctree::
    :maxdepth: 3
    :caption: Starter Notebooks

    starter_nb/starter_nb_01_basic
    starter_nb/starter_nb_02_advanced
    starter_nb/starter_nb_03_your_data

.. toctree::
    :maxdepth: 3
    :caption: MD Notebooks

    md_nb/md_nb_01_traj_ensemble

.. toctree::
    :maxdepth: 3
    :caption: Customization Notebooks

    customization_nb/customization_nb_01_scalars
    customization_nb/customization_nb_02_loss
    customization_nb/customization_nb_03_images
    customization_nb/customization_nb_04_lr


.. toctree::
    :maxdepth: 3
    :caption: Intermediate Notebooks

    intermediate_nb/intermediate_nb_01_sparse

.. toctree::
    :maxdepth: 3
    :caption: Static Code Examples

    static_code
