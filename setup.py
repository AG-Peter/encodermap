from setuptools import setup

with open("description.md", "r") as fh:
    long_description = fh.read()

setup(name='encodermap',
      version='0.0.2',
      description='python library for dimensionality reduction',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Tobias Lemke',
      url="https://github.com/AG-Peter/encodermap",
      packages=['encodermap'],
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'MDAnalysis',
          'tqdm'
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
      ])
