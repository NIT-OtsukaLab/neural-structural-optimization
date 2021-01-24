# -*- coding: utf-8 -*-
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

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train

def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
  return net

#UpSampling2D→UpSampling3D	20201214 K.Taniguchi
def UpSampling3D(factor):
  return layers.UpSampling3D((factor, factor, factor),data_format='channels_last')

#Conv2D→Conv3D	20201214 K.Taniguchi
def Conv3D(filters, kernel_size, **kwargs):
  return layers.Conv3D(filters, kernel_size, padding='same', **kwargs)


problem = problems.PROBLEMS_BY_NAME['mbb_beam_192x64x2_0.4']
layers = tf.keras.layers

#print(problem)

max_iterations = 100
seed=0
args=topo_api.specified_task(problem)
latent_size=128
dense_channels=32
resizes=(1, 2, 2, 2, 1)
conv_filters=(128, 64, 32, 16, 1)
offset_scale=10
kernel_size=(5, 5, 5)
latent_scale=1.0
dense_init_scale=1.0
activation=tf.nn.tanh
conv_initializer=tf.initializers.VarianceScaling
normalization=global_normalization

if len(resizes) != len(conv_filters):
    raise ValueError('resizes and filters must be same size')

activation = layers.Activation(activation)
    
#print("args:",args)

total_resize = int(np.prod(resizes))
d = args['nelz'] // total_resize
h = args['nely'] // total_resize
w = args['nelx'] // total_resize

#print("nelz:",args['nelz'],"nely:",args['nely'],"nelx:",args['nelx'])
#print("total_resize:",total_resize,", d:",d,", h:",h,", w:",w)

net = inputs = layers.Input((latent_size,), batch_size=1)
filters = d * h * w * dense_channels
dense_initializer = tf.initializers.orthogonal(dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
net = layers.Reshape([d, h, w, dense_channels])(net)

print(resizes)
print(conv_filters)

for resize, filters in zip(resizes, conv_filters):
    net = activation(net)
    net = UpSampling3D(resize)(net)
    net = normalization(net)
    net = Conv3D(filters, kernel_size, kernel_initializer=conv_initializer)(net)

    if offset_scale != 0:
        net = AddOffset(offset_scale)(net)

outputs = tf.squeeze(net, axis=[-1])

self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
self.z = self.add_weight(shape=inputs.shape, initializer=latent_initializer, name='z')

def call(self, inputs=None):
    return self.core_model(self.z)
    

ds_cnn = train.train_lbfgs(model, max_iterations)
dims = pd.Index(['cnn-lbfgs'], name='model')
return xarray.concat([ds_cnn], dim=dims)
# -


