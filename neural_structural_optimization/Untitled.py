# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from IPython import display
from PIL import Image
import time
import seaborn
import matplotlib.pyplot as plt
import xarray
import pandas as pd
import tensorflow as tf
import autograd.numpy as np

from neural_structural_optimization import topo_api

def UpSampling3D(factor):
  return layers.UpSampling3D((factor, factor, factor),data_format='channels_last')

def Conv3D(filters, kernel_size, **kwargs):
  return layers.Conv3D(filters, kernel_size, padding='same', **kwargs)

def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
  return net

problem = problems.PROBLEMS_BY_NAME['mbb_beam_192x64x2_0.4']

resizes=(1, 2, 2, 2, 1)
conv_filters=(128, 64, 32, 16, 1)
latent_size=128
dense_channels=32
layers = tf.keras.layers
activation=tf.nn.tanh
args=topo_api.specified_task(problem)

activation = layers.Activation(activation)
normalization=global_normalization

total_resize = int(np.prod(resizes))
d = args['nelz'] // total_resize
h = args['nely'] // total_resize
w = args['nelx'] // total_resize

net = inputs = layers.Input((latent_size,), batch_size=1)
filters = d * h * w * dense_channels
dense_initializer = tf.initializers.orthogonal(dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
net = layers.Reshape([d, h, w, dense_channels])(net)

for resize, filters in zip(resizes, conv_filters):
    net = activation(net)
    #print(net)
    net = UpSampling3D(resize)(net)
    #print(net)
    net = normalization(net)
    #print(net)
    net = Conv3D(filters, kernel_size, kernel_initializer=conv_initializer)(net)
    #print(net)
    
    if offset_scale != 0:
        net = AddOffset(offset_scale)(net)
# -


