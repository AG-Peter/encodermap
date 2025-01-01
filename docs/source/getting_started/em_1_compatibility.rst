.. _getting_started/em_1_compatibility:

EncoderMap 1.0 and 2.0 Compatibility
------------------------------------

Version 2 of the TensorFlow package brought some changes with it. While TensorFlow 1 generally has higher performance, TensorFlow 2 allows for more customization. EncoderMap 3.0 leverages this new feature to allow you to tune the model and the training to your liking. With the new EncoderMap it is even easier to understand what your MD data contains and how it is trained in EncoderMap's Neural Network.

Here's how TensorFlow 2 code looks like with EncoderMap 3.0 looks like.

Tensorflow 2
~~~~~~~~~~~~

.. code-block:: python

  >>> import tensorflow as tf
  >>> tf.__version__
  2.16
  >>> tf.executing_eagerly()
  True
  >>> import encodermap as em
  >>> encodermap.__version__
  3.0

If you come from EncoderMap 1.0/2.0 and want to run your old EncoderMap code and gradually shift to the new TensorFlow version you can use EncoderMap's compatibility mode like so:

Tensorflow 2 in Tensorflow 1 compatibility mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  >>> import tensorflow as tf2
  >>> import tensorflow.compat.v1 as tf
  >>> tf.disable_eager_execution()
  >>> tf2.executing_eagerly()
  False
  >>> import encodermap.encodermap_tf1 as em
  >>> encodermap.__version__
  2.0
