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
import numpy as np

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train

def train_all(problem, max_iterations, cnn_kwargs=None):
    args = topo_api.specified_task(problem)
    if cnn_kwargs is None:
        cnn_kwargs = {}

    model = models.CNNModel(args=args, **cnn_kwargs)
    print("CNN Modeling is Done.")
    ds_cnn = train.train_lbfgs(model, max_iterations)
    print("Training Done.")
    dims = pd.Index(['cnn-lbfgs'], name='model')
    return xarray.concat([ds_cnn], dim=dims)

"""MBB beam with a larger grid"""
problem = problems.PROBLEMS_BY_NAME['mbb_beam_8x8x8_0.5']
max_iterations = 100

# #%time ds = train_all(problem, max_iterations) #%timeが機能しないため,以下の処理に変更
start = time.time()
ds = train_all(problem, max_iterations)
e_time = time.time() - start
print ("e_time:{0}".format(e_time) + "[s]")

#Steps - Compliance
ds.loss.transpose().to_pandas().cummin().loc[:200].plot(linewidth=2)
plt.ylim(0, 330)
plt.ylabel('Compliance (loss)')
plt.xlabel('Optimization step')
seaborn.despine()

#Three-sided view
ds.design.ffill('step').sel(step=max(ds.design.step)).plot.imshow(
    col='model', x='x', y='y', size=2, aspect=2.5, col_wrap=2,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.suptitle(problem.name, y=1.02)

model = ds.design['model']
step = ds.design['step']
x_axis = ds.design['x']
y_axis = ds.design['y']
z_axis = ds.design['z']

design_xy = xarray.DataArray(dims=['model','step','x','y'],
                            coords={'model':model,'step':step,'x':x_axis, 'y':y_axis},
                            name='xy-plane')
design_yz = xarray.DataArray(dims=['model','step','y','z'],
                            coords={'model':model,'step':step,'y':y_axis, 'z':z_axis},
                            name='yz-plane')
design_zx = xarray.DataArray(dims=['model','step','x','z'],
                            coords={'model':model,'step':step,'x':x_axis, 'z':z_axis},
                            name='zx-plane')

design_xy.ffill('step').sel(step=max(ds.design.step)).plot.imshow(
    col='model', x='x', y='y', size=2, aspect=2.5, col_wrap=2,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.suptitle(problem.name, y=1.02)

design_yz.ffill('step').sel(step=max(ds.design.step)).plot.imshow(
    col='model', x='y', y='z', size=2, aspect=2.5, col_wrap=2,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.suptitle(problem.name, y=1.02)

design_zx.ffill('step').sel(step=max(ds.design.step)).plot.imshow(
    col='model', x='x', y='z', size=2, aspect=2.5, col_wrap=2,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.suptitle(problem.name, y=1.02)


#Steps - Models
design_xy.sel(step=[0, 1, 2, 5, 10, 20, 50, 100]).plot.imshow(
    row='model', col='step', x='x', y='y', size=2, aspect=0.5,
    yincrease=False, add_colorbar=False, cmap='Greys')
design_yz.sel(step=[0, 1, 2, 5, 10, 20, 50, 100]).plot.imshow(
    row='model', col='step', x='y', y='z', size=2, aspect=0.5,
    yincrease=False, add_colorbar=False, cmap='Greys')
design_zx.sel(step=[0, 1, 2, 5, 10, 20, 50, 100]).plot.imshow(
    row='model', col='step', x='x', y='z', size=2, aspect=0.5,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.suptitle(problem.name, y=1.02)

print("All Processes are Over")
# -


