<img src="pic/new_logo_m.png" width=45% align="right" />


[![Test](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FAG-Peter%2Fencodermap%2Frefs%2Fheads%2Fgh-pages%2Ftest_badge.json)](https://ag-peter.github.io/encodermap/_static/html_test_runner_report.html)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FAG-Peter%2Fencodermap%2Frefs%2Fheads%2Fgh-pages%2Fcoverage_badge.json)
](https://ag-peter.github.io/encodermap/_static/coverage/index.html)
[![Docbuild](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FAG-Peter%2Fencodermap%2Frefs%2Fheads%2Fgh-pages%2Fdocbuild_badge.json)](https://ag-peter.github.io/encodermap/)
[![color](https://img.shields.io/badge/color-e77800-e77800)]()
[![badge](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![badge](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![MadeWithLove](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/kevinsawade/bcd9d09bc682b4743b84fc6e967478ac/raw/endpoint.json)](https://www.chemie.uni-konstanz.de/ag-peter/)





## Introduction
For a quick intro have a look at the following video:

[<p align="left"> <img src="pic/thumbnail.png" width=50% /> </p>](https://youtu.be/JV59OABhNTY)

You can find more information in these two articles:

[Lemke, Tobias, and Christine Peter. "EncoderMap: Dimensionality Reduction and Generation of Molecule Conformations." Journal of chemical theory and computation 15.2 (2019): 1209-1215.](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00975)

[Lemke, T., Berg, A., Jain, A., & Peter, C. "EncoderMap (II): Visualizing important molecular motions with improved generation of protein conformations" Journal of chemical information and modeling (2019).](https://doi.org/10.1021/acs.jcim.9b00675)

## Documentation

Find the documentation at:

https://ag-peter.github.io/encodermap/


## Quick start

To get started right away, you can try out the tutorial notebooks at mybinder.org or Google Colab:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AG-Peter/encodermap/HEAD)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AG-Peter/encodermap)


## Installation

**Version 3.0.0 is currently not on PyPI. Install with:**

```bash
pip install git+https://github.com/AG-Peter/encodermap.git
```

**Install optional packages for using MD data:**

```bash
pip install -r https://raw.githubusercontent.com/AG-Peter/encodermap/main/md_requirements.txt
```

EncoderMap requires TensorFlow >=2.9 to be installed.
Follow the [instructions on the TensorFlow website](https://www.tensorflow.org/install/pip) to install it
either in the cpu or gpu version.
Then install EncoderMap with pip.
If you want to install it in your home directory use:
```bash
pip3 install --user encodermap
```
If you are in a virtual environment use:
```bash
pip3 install encodermap
```

## Minimal Example
This example shows how to use EncoderMap to project points from a high dimensional data set to
a low dimensional space using the default parameters.
In the data set, each row should represent one data point and the number of columns should be equal to the
number of dimensions.
```python
import encodermap as em
import numpy as np

high_dimensional_data = np.loadtxt("my_high_d_data.csv", delimiter=",")
parameters = em.Parameters()

e_map = em.EncoderMap(parameters, high_dimensional_data)
e_map.train()

low_dimensional_projection = e_map.encode(high_dimensional_data)
```
The resulting `low_dimensional_projection` array has the same number of rows as the `high_dimensional_data`
but the number of columns is two as high dimensional points are projected to a 2d space with default settings.

In contrast to many other dimensionality reduction algorithms EncoderMap does not only allow to efficiently project
form a high dimensional to a low dimensional space. Also the generation of new high dimensional points for any
given points in the low dimensional space is possible:
```python
low_d_points = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]])
newly_generated_high_d_points = e_map.generate(low_d_points)
```
## Tutorials
To get started please check out the [tutorials](tutorials).

Also, have a look at the [examples](encodermap/examples).

## Documentation
More information is available in the [documentations](https://ag-peter.github.io/encodermap/).

## Questions
If you have any questions you can have a look at the [FAQ](questions.md) (not very extensive yet), and you are most welcome to [open an issue here on GitHub](https://github.com/AG-Peter/encodermap/issues/new).

## Citations

```
@article{lemke2019encodermap,
  title={Encodermap: Dimensionality reduction and generation of molecule conformations},
  author={Lemke, Tobias and Peter, Christine},
  journal={Journal of chemical theory and computation},
  volume={15},
  number={2},
  pages={1209--1215},
  year={2019},
  publisher={ACS Publications}
}
```

```
@article{lemke2019encodermap,
  title={EncoderMap (II): Visualizing important molecular motions with improved generation of protein conformations},
  author={Lemke, Tobias and Berg, Andrej and Jain, Alok and Peter, Christine},
  journal={Journal of Chemical Information and Modeling},
  volume={59},
  number={11},
  pages={4550--4560},
  year={2019},
  publisher={ACS Publications}
}
```

Since PyEMMA (https://github.com/markovmodel/PyEMMA) has been archived, EncoderMap incorporated some ideas of PyEMMA's featurization code. Find the citation here:

```
@article{scherer2015pyemma,
  title={PyEMMA 2: A software package for estimation, validation, and analysis of Markov models},
  author={Scherer, Martin K and Trendelkamp-Schroer, Benjamin and Paul, Fabian and P{\'e}rez-Hern{\'a}ndez, Guillermo and Hoffmann, Moritz and Plattner, Nuria and Wehmeyer, Christoph and Prinz, Jan-Hendrik and No{\'e}, Frank},
  journal={Journal of chemical theory and computation},
  volume={11},
  number={11},
  pages={5525--5542},
  year={2015},
  publisher={ACS Publications}
}

```
