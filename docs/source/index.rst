.. encodermap documentation master file, created by
   sphinx-quickstart on Mon Sep  7 12:10:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Encodermap
==========

For a quick intro have a look at the following video:

..  youtube:: JV59OABhNTY

Encodermap is a neural-network autoencoder based approach to dimensionality reduction.


Start with the tutorials
------------------------

You can access all EncoderMap tutorials online on mybinder.org:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/AG-Peter/encodermap/HEAD


.. grid:: 2

    .. grid-item-card:: Getting started
        :link: getting_started
        :link-type: ref
        :img-top: ../../pic/rocket_pictogram.svg

        For installation instructions, and FAQs refer this section.

    .. grid-item-card:: User guide
        :link: user_guide
        :link-type: ref
        :img-top: ../../pic/book_pictogram.svg

        The user guide contains more in-depth information about concepts and use cases.


.. grid:: 2

    .. grid-item-card::  API reference
        :link: reference
        :link-type: ref
        :img-top: ../../pic/brackets_pictogram.svg

        Collection of all code documentation.

    .. grid-item-card::  Developer guide
        :link: contributing
        :link-type: ref
        :img-top: ../../pic/tools_pictogram.svg

        The developer guide helps you in contributing to encodermap.



Test and Coverage Reports
=========================

.. grid:: 2

    .. grid-item::

        .. button-link:: _static/html_test_runner_report.html
           :color: primary
           :expand:

            Reports of unittests

    .. grid-item::

        .. button-link:: _static/coverage
            :color: primary
            :expand:

            Code coverage

Citations |DOI1| |DOI2|
=======================

You can find more information in these two articles. Please cite one or both if you use EncoderMap in your project. :cite:`lemke2019encodermap` and :cite:`lemke2019encodermap2`

.. bibliography:: refs.bib


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:


.. |DOI1| image:: https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.8b00975-blue.svg
   :target: https://doi.org/10.1021/acs.jctc.8b00975

.. |DOI2| image:: https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.9b00675-blue.svg
   :target: https://doi.org/10.1021/acs.jcim.9b00675


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started/index
    user_guide_and_examples/index
    reference/index
    development/index
    whatsnew/index
