# README for EncoderMap.InteractivePlotting cluster

You just used EncoderMap's `InteractivePlotting` and saved a cluster. Here's some information about this cluster. The cluster was selected from a `TrajectoryEnsemble` containing 126 trajectories, 55326 frames and 6 unique topologies. This cluster was assigned the number 0. The file /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/cluster_0.h5 contains only  frames, chosen as representatives for this cluster. This file can be loaded with EncoderMap's `TrajEnsemble.from_dataset('/home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/cluster_0.h5')` method. Look at EncoderMap's documentation at https://ag-peter.github.io/encodermap/ to learn more about Trajectory Ensembles.

### The complete Ensemble is also present

If you want to get more information about the clustering you carried out, you can refer to these files:

### cluster_0.csv

This `.csv` file contains info about the complete ensemble this cluster was selected from. The columns are as follows:

| traj_num   | The number of the trajectory in the full dataset. This number is 0-based. If only one trajectory is loaded, its `trajectory number` might also be `None`. |
| ---------- | ------------------------------------------------------------ |
| frame_num  | The frame number. The trajectory number and frame number can be used to unmistakably identify frames in a trajectory ensemble. Frame numbers are also 0-based. |
| traj_file  | Contains the trajectory data (file formats such as .xtc, .dcd, .h5). |
| top_file   | Contains the topology of the file (i.e. atom types, masses, residues) (file formats such as .pdb, .gro, .h5). Some trajectory files (.h5) might also contain the topology. In that case `trajectory file` and `topology` file are identical. |
| time       | The time of the frame. This can be used for time-based indexing of trajectories. EncoderMap offers the `SingleTraj.tsel[time]` accessor to distinguish it from frame-based indexing via `SingleTraj[frame]`. |
| x          | The x coordinate of the low-dimensional projection.          |
| y          | The y-coordinate of the low-dimensional projection.          |
| cluster_id | This column contains -1, which are points not included in a cluster (outliers). Cluster 1 is denoted by a 0 in this column. If multiple clusters have been selected this column can contain multiple integer values. For every subsequent cluster, the `cluster_id` is advanced by 1. |

### cluster_0_selector.npy

This numpy array contains the (x, y)-coordinates of the selector, that was used to highlight the cluster. Be careful, this shape might not be convex, so using convex algortihms to find points inside this Polygon might not work.

### cluster_0.png

A nice render of the selected cluster.

## Loading a HDF5 file (.h5) with EncoderMap

EncoderMap introduces a way of storing multiple trajectories (a `TrajectorEnsemble`) in a
single file. These files can be loaded via:

```python
import encodermap as em
trajs = em.TrajEnsemble.from_dataset('/home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/cluster_0.h5')
```

## Rendering this document

If you don't like to view plain markdown files with a text-viewer there are many viewers available, that are able to render markdown nicely. I am currently using ghostwriter:

https://ghostwriter.kde.org/

If you want to create a pdf from this document you can try a combination of pandoc, latex and groff.

### HTML

```bash
pandoc /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/README.md.md -o /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/README.md.html
```

### Latex

```bash
pandoc /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/README.md.md -o /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/README.md.pdf
```

### Groff

```bash
pandoc /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/README.md.md -t ms -o /home/kevin/encodermap_paper_images/submission/datasets/trained_nns/figure_3/mds/clusters/2025-05-04T21:39:43+02:00/README.md.pdf
```

## Debug Info

```
encodermap.__version__ = 3.0.1+6.g6db4b33.dirty
system_user = kevin
platform = Linux
platform_release = 6.8.0-57-generic
platform_version = #59~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Mar 19 17:07:41 UTC 2
architecture = x86_64
hostname = Marlin
ip_address = 127.0.1.1
mac_address = 73:a9:3b:f3:9c:fe
processor = x86_64
ram = 31 GB
pip freeze = absl-py==1.4.0
accessible-pygments==0.0.4
alabaster==0.7.16
anyio==4.2.0
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
array-record==0.5.0
arrow==1.3.0
asttokens==2.4.1
astunparse==1.6.3
async-lru==2.0.4
attrs==23.2.0
autoclasstoc==1.7.0
Babel==2.14.0
beartype==0.18.5
beautifulsoup4==4.12.3
binaryornot==0.4.4
biopython==1.83
black==24.2.0
bleach==6.1.0
blinker==1.7.0
blosc2==2.5.1
bokeh==3.4.1
brukeropusreader==1.3.4
cachetools==5.3.2
certifi==2024.2.2
cffi==1.16.0
cfgv==3.4.0
chardet==5.2.0
charset-normalizer==3.3.2
click==8.1.7
cloudpickle==3.0.0
colour==0.1.5
comm==0.2.1
compose-format==1.2.0
contourpy==1.2.0
cookiecutter==2.5.0
coverage==7.4.1
cycler==0.12.1
Cython==0.29.37
dash==2.15.0
dash-bio==1.0.2
dash-bootstrap-components==1.5.0
dash-bootstrap-templates==1.1.2
dash-core-components==2.0.0
dash-html-components==2.0.0
dash-table==5.0.0
dash_auth==2.2.0
dash_mantine_components==0.15.2
dask==2024.2.0
dask-jobqueue==0.8.5
data-science-types==0.2.23
debugpy==1.8.1
decorator==4.4.2
deeptime==0.4.4
defusedxml==0.7.1
dill==0.3.8
distlib==0.3.8
distributed==2024.2.0
dm-tree==0.1.8
docker==7.0.0
docstring-parser==0.15
docutils==0.20.1
einops==0.8.0
-e git+https://github.com/AG-Peter/encoder_map_private.git@6db4b332be2e2e4fc69fc0daf9e395c701541903#egg=encodermap
entrypoints==0.4
etils==1.7.0
exceptiongroup==1.2.0
executing==2.0.1
fasteners==0.19
fastjsonschema==2.19.1
filelock==3.13.1
flake8==5.0.4
Flask==3.0.2
flatbuffers==23.5.26
fonttools==4.49.0
fqdn==1.5.1
fsspec==2024.2.0
gast==0.5.4
GEOparse==2.0.3
gitdb==4.0.11
GitPython==3.1.42
google-auth==2.28.0
google-auth-oauthlib==1.2.0
google-pasta==0.2.0
googleapis-common-protos==1.62.0
graphviz==0.20.1
greenlet==3.1.1
GridDataFormats==1.0.2
grpcio==1.60.1
h11==0.14.0
h5netcdf==1.3.0
h5py==3.10.0
hdbscan==0.8.33
html-testRunner==1.2.1
httpcore==1.0.3
httpx==0.26.0
humanfriendly==10.0
identify==2.5.35
idna==3.6
igraph==0.11.4
imageio==2.34.0
imageio-ffmpeg==0.4.9
imagesize==1.4.1
imohash==1.1.0
importlib-metadata==7.0.1
importlib-resources==6.1.1
iniconfig==2.0.0
ipycanvas==0.13.1
ipykernel==6.29.5
ipympl==0.9.3
ipython==8.21.0
ipython-genutils==0.2.0
ipywidgets==8.1.5
isoduration==20.11.0
isort==5.13.2
itsdangerous==2.1.2
jedi==0.19.1
Jinja2==3.1.3
joblib==1.3.2
json-spec==0.12.0
json5==0.9.14
jsonpointer==2.4
jsonschema==4.21.1
jsonschema-specifications==2023.12.1
jupyter-cache==1.0.1
jupyter-events==0.9.0
jupyter-lsp==2.2.2
jupyter_client==8.6.0
jupyter_core==5.7.1
jupyter_server==2.12.5
jupyter_server_terminals==0.5.2
jupyterlab==4.1.1
jupyterlab_pygments==0.3.0
jupyterlab_server==2.25.3
jupyterlab_widgets==3.0.13
kaggle==1.6.6
kaleido==0.2.1
keras==2.15.0
kiwisolver==1.4.5
latexcodec==2.0.1
libclang==16.0.6
lipsum==0.1.2
livereload==2.7.1
llvmlite==0.42.0
locket==1.0.0
Markdown==3.5.2
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.8.3
matplotlib-inline==0.1.6
mccabe==0.7.0
mda-xdrlib==0.2.0
MDAnalysis==2.7.0
mdit-py-plugins==0.4.2
mdshare==0.4.2
mdtraj==1.9.9
mdurl==0.1.2
mistune==3.0.2
ml-dtypes==0.2.0
mmh3==4.1.0
mmtf-python==1.1.3
more-itertools==10.5.0
moviepy==1.0.3
mpmath==1.3.0
mrcfile==1.5.0
msgpack==1.0.7
multiprocess==0.70.16
mypy==1.10.0
mypy-extensions==1.0.0
myst-nb==1.1.2
myst-parser==4.0.0
nbclient==0.9.0
nbconvert==7.16.4
nbformat==5.9.2
nbsphinx==0.9.6
nbsphinx-link==1.3.1
nbval==0.11.0
ndindex==1.8
nest-asyncio==1.6.0
networkx==3.2.1
nglview==3.0.8
nodeenv==1.8.0
notebook==7.1.0
notebook_shim==0.2.4
numba==0.59.0
numexpr==2.9.0
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.5.40
nvidia-nvtx-cu12==12.1.105
oauthlib==3.2.2
OpenEXR==1.3.9
OpenMM==8.1.0
opt-einsum==3.3.0
optional-imports==1.0.4
orjson==3.9.15
outcome==1.3.0.post0
overrides==7.7.0
packaging==23.2
pandas==2.2.0
pandoc==2.3
pandoc-acro==0.10.1
pandocfilters==1.5.1
panflute==2.3.1
papermill==2.5.0
ParmEd==4.2.2
parso==0.8.3
partd==1.4.1
pathos==0.3.2
pathspec==0.12.1
pdbfixer @ file:///home/kevin/git/pdbfixer
peppercorn==0.6
periodictable==1.6.1
pexpect==4.9.0
pillow==11.0.0
pip==22.0.2
platformdirs==4.2.0
plotly==5.19.0
pluggy==1.5.0
plumbum==1.8.2
ply==3.11
pox==0.3.4
ppft==1.7.6.8
pre-commit==3.6.2
proglog==0.1.10
progress-reporter==2.0
prometheus_client==0.20.0
promise==2.3
prompt-toolkit==3.0.43
protobuf==3.20.3
psutil==5.9.8
ptyprocess==0.7.0
pure-eval==0.2.2
py-cpuinfo==9.0.0
pyarrow==17.0.0
pyasn1==0.5.1
pyasn1-modules==0.3.0
pybtex==0.24.0
pybtex-docutils==1.0.3
pycodestyle==2.9.1
pycparser==2.21
pydata-sphinx-theme==0.16.1
pydoctest==0.1.22
pydot==2.0.0
pydssp==0.9.0
pyEMMA==2.5.12
pyflakes==2.5.0
Pygments==2.17.2
pyparsing==3.1.1
pyPept @ git+https://github.com/Boehringer-Ingelheim/pyPept.git@2e9b8cce49bfeb26920053c1e1797d9017e52724
pyproject-flake8==5.0.4
PyQt5==5.15.10
PyQt5-Qt5==5.15.2
PyQt5-sip==12.13.0
PySocks==1.7.1
pytest==8.2.2
python-dateutil==2.8.2
python-json-logger==2.0.7
python-slugify==8.0.4
python-version==0.0.2
pytz==2024.1
PyYAML==6.0.1
pyzmq==25.1.2
rdkit==2023.9.5
referencing==0.33.0
requests==2.31.0
requests-oauthlib==1.3.1
retrying==1.3.4
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==13.7.0
rpds-py==0.18.0
rsa==4.9
ruamel.yaml==0.18.6
ruamel.yaml.clib==0.2.8
scikit-learn==1.4.1.post1
scikit-spatial==7.2.0
scipy==1.12.0
seaborn==0.13.2
selenium==4.21.0
Send2Trash==1.8.2
setuptools==59.6.0
six==1.16.0
smmap==5.0.1
sniffio==1.3.0
snowballstemmer==2.2.0
sortedcontainers==2.4.0
soupsieve==2.5
Sphinx==8.1.3
sphinx-click==6.0.0
sphinx-copybutton==0.5.2
sphinx-gallery==0.18.0
sphinx_design==0.6.1
sphinxcontrib-applehelp==1.0.8
sphinxcontrib-bibtex==2.6.3
sphinxcontrib-devhelp==1.0.6
sphinxcontrib-fulltoc==1.2.0
sphinxcontrib-htmlhelp==2.1.0
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==1.0.7
sphinxcontrib-serializinghtml==1.1.10
sphinxcontrib-youtube==1.4.1
SQLAlchemy==2.0.36
stack-data==0.6.3
style==1.1.0
sympy==1.12.1
tables==3.9.2
tabulate==0.9.0
tblib==3.0.0
tenacity==8.2.3
tensor-annotations==2.0.3
tensor-annotations-tensorflow-stubs==2.0.3
tensorboard==2.15.2
tensorboard-data-server==0.7.2
tensorflow==2.15.0.post1
tensorflow-addons==0.23.0
tensorflow-datasets==4.9.4
tensorflow-estimator==2.15.0
tensorflow-graphics==2021.12.3
tensorflow-io-gcs-filesystem==0.36.0
tensorflow-metadata==1.14.0
tensorflow-probability==0.23.0
termcolor==2.4.0
terminado==0.18.0
testbook==0.4.2
text-unidecode==1.3
texttable==1.7.0
threadpoolctl==3.3.0
tinycss2==1.2.1
tk==0.1.0
toml==0.10.2
tomli==2.0.1
toolz==0.12.1
torch==2.3.0
tornado==6.4
tqdm==4.66.2
traitlets==5.14.1
transformations==2024.6.1
trimesh==4.1.4
trio==0.25.1
trio-websocket==0.11.1
triton==2.3.0
typeguard==2.13.3
types-python-dateutil==2.8.19.20240106
typing_extensions==4.9.0
tzdata==2024.1
unrpa==2.3.0
uri-template==1.3.0
urllib3==2.2.0
varint==1.0.2
versioneer==0.29
virtualenv==20.25.0
vulture==2.11
wcwidth==0.2.13
webcolors==1.13
webencodings==0.5.1
websocket-client==1.7.0
Werkzeug==3.0.1
wheel==0.42.0
widgetsnbextension==4.0.13
-e git+https://github.com/kevinsawade/word_clock.git@a338bc1b71c5d5538198844a2a7a8841177d64cd#egg=word_clock
wrapt==1.14.1
wsproto==1.2.0
xarray==2024.1.1
xyzservices==2024.4.0
zict==3.0.0
zipp==3.17.0

```