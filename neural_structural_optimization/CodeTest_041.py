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
'''UpSampling2D抜き出し'''

import tensorflow as tf
import autograd.numpy as np

def UpSampling3D(factor):
  return layers.UpSampling3D((factor, factor, factor),data_format='channels_last')

def UpSampling2D(factor):
  return layers.UpSampling2D((factor, factor),data_format='channels_last')

resizes=(1, 2, 2, 2, 1)
conv_filters=(128, 64, 32, 16, 1)
nelx=192
nely=64
nelz=2
latent_size=128
dense_channels=32
dense_init_scale=1.0
layers = tf.keras.layers
activation = layers.Activation(tf.nn.tanh)

total_resize = int(np.prod(resizes))
#d = nelz // total_resize
h = nely // total_resize
w = nelx // total_resize

net = inputs = layers.Input((latent_size,), batch_size=1)
#filters = d * h * w * dense_channels
filters = h * w * dense_channels
print("filters=",filters)

dense_initializer = tf.initializers.orthogonal(dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
print("dense_init=",dense_initializer)

net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
print("net(Dense)=",net)

#net = layers.Reshape([d, h, w, dense_channels])(net)
net = layers.Reshape([h, w, dense_channels])(net)
print("net(Reshape)=",net)

for resize, filters in zip(resizes, conv_filters):
    print("resize=",resize," filters=",filters)
    net = activation(net)
    print("net(activation)=",net)
    #net = UpSampling3D(resize)(net)
    net = UpSampling2D(resize)(net)
    print("net(UpSamp2D)=",net)
    
# -


