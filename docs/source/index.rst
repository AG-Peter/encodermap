.. encodermap documentation master file, created by
   sphinx-quickstart on Mon Sep  7 12:10:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EncoderMap
==========

.. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fencodermap.site%2F_static%2Ftest_badge.json
   :alt: Unittest Badge
   :target: _static/html_test_runner_report.html

.. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fencodermap.site%2F_static%2Fcoverage_badge.json
   :alt: Coverage Badge
   :target: _static/coverage/index.html

.. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fencodermap.site%2F_static%2Fdocbuild_badge.json
   :alt: Docbuild Badge
   :target: https://encodermap.site

.. image:: https://img.shields.io/badge/color-e77800-e77800
   :alt: EncoderMap's main color

.. image:: https://img.shields.io/badge/License-LGPL_v3-blue.svg
   :alt: EncoderMap's license
   :target: https://www.gnu.org/licenses/lgpl-3.0

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :alt: Black
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :alt: Iosrt
   :target: https://pycqa.github.io/isort/

.. image:: https://www.mypy-lang.org/static/mypy_badge.svg
   :alt: MyPy
   :target: https://mypy-lang.org/

.. image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/kevinsawade/bcd9d09bc682b4743b84fc6e967478ac/raw/endpoint.json
   :alt: Made with love
   :target: https://www.chemie.uni-konstanz.de/ag-peter/

Encodermap is a neural-network autoencoder based approach to dimensionality reduction.  For a quick intro have a look at the following video:

..  youtube:: JV59OABhNTY
   :align: center

Interactive Tutorials
---------------------

You can access interactive versions of EncoderMap tutorials online on BinderHub or Google Colab:

Start with the tutorials
------------------------

You can access all EncoderMap tutorials online on mybinder.org:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/AG-Peter/encodermap/HEAD


.. grid:: 2

   .. grid-item-card:: Run EncoderMap's notebooks on Binder
      :img-bottom: _static/Binder_Hub.png
      :img-alt: the Mybinder logo
      :link: https://mybinder.org/v2/gh/AG-Peter/encodermap/HEAD

   .. grid-item-card:: Run EncoderMap's notebooks on Colab
      :img-bottom: _static/Google_Colaboratory.png
      :img-alt: the Google Colab logo
      :link: https://colab.research.google.com/github/AG-Peter/encodermap

Documentation
-------------

This is the main page of EncoderMap's documentation. Click on one of the cards below to look at specific parts of EncoderMap's documentation

.. grid:: 3

   .. grid-item-card:: :fas:`rocket` Getting Started
      :margin: 2 0 0 0
      :link: getting_started/index
      :link-type: ref

      The getting started with EncoderMap guide contains the most crucial information to get you started with EncoderMap.

   .. grid-item-card:: :fas:`clipboard` Notebook Galleries
      :margin: 2 0 0 0
      :link: notebooks/index
      :link-type: ref

      Have a look at static copies of EncoderMap's introductory notebooks. Each notebook contains a link to Google Colab to run the notebook interactively.

   .. grid-item-card:: :fas:`book` User Guide
      :margin: 2 0 0 0
      :link: user_guide/index
      :link-type: ref

      The user guide provides an overview of the basic concepts and top-level implementations of EncoderMap.

   .. grid-item-card:: :fas:`list` API Reference
      :margin: 2 0 0 0
      :link: api/index
      :link-type: ref

      The API section contains the complete documentation generated by Sphinx' autodoc. The User Guide and the API reference each other.

   .. grid-item-card:: :fas:`vials` Test Results
      :margin: 2 0 0 0
      :link: _static/html_test_runner_report.html

      The results of the tests.

   .. grid-item-card:: :fas:`shield` Coverage
      :margin: 2 0 0 0
      :link: _static/coverage/index.html

      The current code coverage.

   .. grid-item-card:: :fas:`database` MyPy type checking results
      :margin: 2 0 0 0
      :link: _static/htmlmypy/index.html

      Results from the MyPy static type checker.

   .. grid-item-card:: :fas:`pen-nib` Changelog
      :margin: 2 0 0 0
      :link: changelog/index
      :link-type: ref

      A link to CHANGELOG.md.

   .. grid-item-card:: :fas:`code` Developer Guide
      :margin: 2 0 0 0
      :link: contributing/index
      :link-type: ref

      Guides to allow you to make contributions to EncoderMap.



Citations |DOI1| |DOI2|
=======================

You can find more information in these two articles. Please cite one or both if you use EncoderMap in your project. :cite:`lemke2019encodermap` and :cite:`lemke2019encodermap2`

.. bibliography:: refs.bib


.. |DOI1| image:: https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.8b00975-blue.svg
   :target: https://doi.org/10.1021/acs.jctc.8b00975

.. |DOI2| image:: https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.9b00675-blue.svg
   :target: https://doi.org/10.1021/acs.jcim.9b00675


.. toctree::
   :maxdepth: 5
   :hidden:

   getting_started/index
   notebooks/index
   user_guide/index
   api/index
   contributing/index
