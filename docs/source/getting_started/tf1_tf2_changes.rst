.. _tf1_tf2_changes:

Changes between Tensorflow 1 and Tensorflow 2 and EncoderMap
============================================================

Version 2 of the Tensorflow package brought some changes with it.

Tensorflow 2
------------

.. code-block:: python

  >>> import tensorflow as tf
  >>> tf.__version__
  2.1
  >>> tf.executing_eagerly()
  True
  >>> import encodermap as em
  >>> encodermap.__version__
  3.0

Tensorflow 2 in Tensorflow 1 compatibility mode
-----------------------------------------------

.. code-block:: python

  >>> import tensorflow as tf2
  >>> import tensorflow.compat.v1 as tf
  >>> tf.disable_eager_execution()
  >>> tf2.executing_eagerly()
  False
  >>> import encodermap.encodermap_tf1 as em
  >>> encodermap.__version__
  2.0
