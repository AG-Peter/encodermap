from setuptools import setup

with open("description.md", "r") as fh:
    long_description = fh.read()

# read _version.py to have a single source file for version.
exec(open("encodermap/_version.py").read())

# setup
setup(
    name="encodermap",
    version=__version__,
    python_requires=">=3.9",
    description="python library for dimensionality reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tobias Lemke, Kevin Sawade",
    url="https://github.com/AG-Peter/encodermap",
    packages=[
        "encodermap",
        "encodermap.plot",
        "encodermap.callbacks",
        "encodermap.autoencoder",
        "encodermap.data",
        "encodermap.encodermap_tf1",
        "encodermap.loading",
        "encodermap.loss_functions",
        "encodermap.misc",
        "encodermap.models",
        "encodermap.moldata",
        "encodermap.parameters",
        "encodermap.trajinfo",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "MDAnalysis",
        "tqdm>=4.4.0",
        "tensorflow",
        "nglview>=3.0.1",
        "seaborn>=0.11.1",
        "pillow==9.0.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
