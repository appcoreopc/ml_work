from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import tensorflow as tf

np.set_printoptions(precision=4)


dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

for elem in dataset:
  print(elem.numpy())

x = np.array([2,3,1,0])

dataset = tf.data.Dataset.from_tensor_slices(x)

for elem in dataset:
  print(elem.numpy())


 ###  tf.random.uniform

ru = tf.random.uniform([4, 10])
print(ru)


TFRecords 

# is a variety of file formats so that you can process large datasets that do not fit in memory. For example, the TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data. The tf.data.TFRecordDataset class enables you to stream over the contents of one or more TFRecord files as part of an input pipeline.



