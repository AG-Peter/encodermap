# Changelog

## Version 3.1.0 (Todo Date)

### Additions

- More Tests.
  - A lot more tests have been implemented. The software is pretty stable for general use.
- Reference training.
  - The `AngleDihedralCartesianEncoderMap` class now runs a short reference training before the actual neural network training. The idea behind this training is to get typical values for the cartesian cost contributions (dihedral_cost, angle_cost, cartesian_cost). And adjust them to roughly the same values.
- Plotting:
  - Added plotly interactive distance histogram, in which the sigmoid parameters can be adjusted interactively. This function is accessible as the `distance_histogram_interactive` function in `encodermap.plot`:

```python
from encodermap.plot import distance_histogram_interactive
distance_histogram_interactive(
    highd_data,
    periodicity=3.141,
)
```
- Added a plotly/dash-based interactive plotting, which supports clustering and path generation.

- The `TrajEnsemble` and `SingleTraj` classes:
  - Added a way for a `TrajEnsemble` to be saved to a single netCDF-formatted HDF5 file.
- Better Docs:
  - Added a `run_livereload.py` script to `docs` with which the docs can be built interactively.
  - Reworked the notebooks to allow better Google Colab integration and added the new interactive plotting tools.
  - Reworked the notebooks to allow better Google Colab integration and added the new interactive plotting tools.
- Tensorflow stuff:
  - Added a clashes metric, that calculates the clashes during training.
  - A RMSD metric is currently being developed.
- Changed to `pyproject.toml` and added better import sorting with isort. Can be run with `$ isort .`
- Added the possibility to choose different kernel and bias initializers. You can now supply the keywords:
  - "VarianceScaling": For the normal variance scaling initializer. This is the default.
  - "deterministic": This will use the old `tf.compat.v1.variance_scaling_initizlier` wiht a seed, so that the starting weights are deterministic.
  - "ones": For a weight matrix full of ones.
  - You can also provide a dict following TensorFlow's naming convention for unnamed dense layers: ["dense/kernel", "dense_1/kernel", "dense_2/kerne", etc.] as the keys and np.ndarrays as the values and these arrays will be used. This can also be used to rebuilt a model from bare numpy weight arrays.
- The BiasInitializer doesn't recognize the standard "VarianceScaling" keyword, but uses the "RandomNormal" keyword as a default. It also recognizes "deterministic", "ones" and can use a dict with these keys: ["dense/bias", "dense_1/bias", "dense_2/bias", etc.]

### Fixes

- Fixed error in `TrajEnsemble._pyemma_indexing`, which did not correctly concatenate the `SingleTraj`'s.
- Fixed error using the `TrajEnsemble.load_CVs` method with the `labels` argument, which did not recognize the passed in `data` as a `np.array`.
- Fixed the calculation of sidechains in the tf1 version according to #25.
- Fixed the activation functions for the ADCEMap model.
- Fixed the splits of the ADCEMap model, which led to bad decoding.
- Replaced the `tf.keras.backend.Variable` `_train_step` in all encodermap models that implement custom training with `_my_train_counter`, because the subclasses models advanced the `train_step` by themselves, leading to train_counters advancing in multiples of two for all logging purposes (tensorboard).
- Fixed the way atom indices are passed for `TrajEnsemble`s. Previously the `xr.Dataset`s of the `SingleTraj` classes had attributes with numpy arrays giving the indices of the atoms from which this attribute was calculated. E.g. when a dihedral feature was loaded, the `traj._CVs.attrs["indices"]` had a numpy array of shape (n_dihedrals, 4) to index the atoms from which the dihedrals were loaded. Adding a distance feature to this traj would have made the `traj._CVs.attrs["indices"]` into a list of numpy arrays with shapes [(n_dihedrals, 4), (n_distances, 2)]. This attribute could not be serialized correctly. Now the datasets contain their own `feature_indices` dataarrays, that omit the frame coordinate and keep track of the atoms from which this feature was calculated.
- Fixed the `dihedral_to_cartesian_tf_one_way_layers` function to be more compatible with the tf1 version of EncoderMap.
- Fixed that the argument `dataset` for `AngleCartesianEncoderMap` is not used, when the `trajs` have the property` _CVs_in_file` as True.
- Fixed the `np.mean` section in `dihedral_to_cartesian_diubi.py`:
```python
means = []
for values in costs:
    means.append(np.mean([i.value for i in values]))
```

## Version 3.0.1 (2023-07-27)

### Fixes

- Fixed error in `encodermap_tf1/moldata.py`, where sidechain dihedrals are not
correctly indexed. Now using MDTraj's chi1, ch2, ... indexers (#25)
- In plot/utils.py fixed `_unpack_cluster_info` to allow pandas versions < 2.0 and >= 2.0.
- Fixed github workflow to accomodate newer versions of html-testRunner (see: https://stackoverflow.com/questions/71858651/attributeerror-htmltestresult-object-has-no-attribute-count-relevant-tb-lev)
- Fixed the optional imports for dask.
- Fixed the loss function for the ADCEncoderMap cartesian dist sig cost.
- Fixed the unittest for optional imports in tests/test_optional_imports_before_installing_reqs.py.
- Fixed the unittests in tests/test_moldata.py (#25).
- Fixed the unittests in tests/test_losses.py
- Fixed the .gitignore and excluded test files.

### Additions

- Added `del_CVs` to `SingleTraj` and `TrajEnsemble` to easily delete attached CVs.

## 2021-02-01

- Reduced the number of training steps in the example notebooks.
- Added Google Colab links to the example notebooks.

## Version 3.0.0 (2021-01-25)

- Updated tests to TensorFlow2
- Added trajinfo classes.
- Autoencoder model and high-level training available.

- Started tracking changes via this changelog.
- Started semantic versioning.
- Features from Tensorflow 1 version are all ported to tf2.
