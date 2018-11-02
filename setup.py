from setuptools import setup

setup(name='encoder_map',
      version='0.1',
      description='python library for dimensionality reduction',
      author='Tobias Lemke',
      packages=['encoder_map'],
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'MDAnalysis',
          'tqdm'
      ],
      zip_safe=False)
