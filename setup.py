# Third Party Imports
import versioneer
from setuptools import setup


with open("description.md", "r") as fh:
    long_description = fh.read()

# read _version.py to have a single source file for version.
# deprecated on 2023-08-25 and replaced with versioneer
# exec(open("encodermap/_version.py").read())

# setup
setup(
    name="encodermap",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.9",
    include_package_data=True,
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
        "transformations",
        "tomli",
        "versioneer",
        "numpy",
        "matplotlib",
        "scipy",
        "MDAnalysis",
        "tqdm>=4.4.0",
        "tensorflow>=2.15.0",
        "nglview>=3.0.1",
        "seaborn>=0.11.1",
        "pillow>=10.0.1",
        "ipywidgets>=8.0",
        "optional_imports>=1.0.4",
        "tensorflow_probability",
        "rich",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
