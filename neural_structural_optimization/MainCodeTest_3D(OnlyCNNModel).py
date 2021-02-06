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
from mpl_toolkits.mplot3d import Axes3D
import xarray
import pandas as pd

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
problem = problems.PROBLEMS_BY_NAME['mbb_beam_192x64x64_0.4']
max_iterations = 100

# # #%time ds = train_all(problem, max_iterations) %timeが機能しないため,以下の処理に変更
start = time.time()
ds = train_all(problem, max_iterations)
e_time = time.time() - start
print ("e_time:{0}".format(e_time) + "[s]")

ds.loss.transpose().to_pandas().cummin().loc[:200].plot(linewidth=2)
plt.ylim(230, 330)
plt.ylabel('Compliance (loss)')
plt.xlabel('Optimization step')
seaborn.despine()

# the pixel-lbfgs does not run for the full 100 steps (it terminates
# early due to reaching a local minima), so use fill() to forward fill
# to the last valid design.
ds.design.ffill('step').sel(step=100).plot.imshow(
    col='model', x='x', y='y', size=2, aspect=2.5, col_wrap=2,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.suptitle(problem.name, y=1.02)

ds.design.sel(step=[0, 1, 2, 5, 10, 20, 50, 100]).plot.imshow(
    row='model', col='step', x='x', y='y', size=2, aspect=0.5,
    yincrease=False, add_colorbar=False, cmap='Greys')
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.suptitle(problem.name, y=1.02)
# -

