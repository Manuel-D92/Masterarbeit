from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import pandas as pd
import os
#from tensorflow.keras.layers.experimental import preprocessing

#class WindowGenerator():
#  def __init__(self, input_width, label_width, shift,
#               train_df=train_df, val_df=val_df, test_df=test_df,
#               label_columns=None):
#    # Store the raw data.
#    self.train_df = train_df
#    self.val_df = val_df
#    self.test_df = test_df
#
#    # Work out the label column indices.
#    self.label_columns = label_columns
#    if label_columns is not None:
#      self.label_columns_indices = {name: i for i, name in
#                                    enumerate(label_columns)}
#    self.column_indices = {name: i for i, name in
#                           enumerate(train_df.columns)}
#
#    # Work out the window parameters.
#    self.input_width = input_width
#    self.label_width = label_width
#    self.shift = shift
#
#    self.total_window_size = input_width + shift
#
#    self.input_slice = slice(0, input_width)
#    self.input_indices = np.arange(self.total_window_size)[self.input_slice]
#
#    self.label_start = self.total_window_size - self.label_width
#    self.labels_slice = slice(self.label_start, None)
#    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
#
#  def __repr__(self):
#    return '\n'.join([
#        f'Total window size: {self.total_window_size}',
#        f'Input indices: {self.input_indices}',
#        f'Label indices: {self.label_indices}',
#        f'Label column name(s): {self.label_columns}'])
#
#  def split_window(self, features):
#      inputs = features[:, self.input_slice, :]
#      labels = features[:, self.labels_slice, :]
#      if self.label_columns is not None:
#          labels = tf.stack(
#              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#              axis=-1)
#
#      # Slicing doesn't preserve static shape information, so set the shapes
#      # manually. This way the `tf.data.Datasets` are easier to inspect.
#      inputs.set_shape([None, self.input_width, None])
#      labels.set_shape([None, self.label_width, None])
#
#      return inputs, labels

path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1"
path_test_sequenz = r"C:\Users\dauserml\Documents\2020_09_25\Testsequenz_1"
time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+"\\all_Files.csv",delimiter=';')
time = time_xyz_antennen_Signal_Komplex_all_Files[:,0]

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,4:]
Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,1:4]

data_gen = TimeseriesGenerator(X_1,Y_1, length=16, sampling_rate=4,batch_size=32)

assert len(data_gen) == 2236
batch_0 = data_gen[0]
batch_1 =data_gen[1]
x, y = batch_0

print('fsdf')