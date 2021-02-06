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
# PYTHON_MATPLOTLIB_3D_PLOT_03

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Figureの大きさと3DAxeS
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection="3d")

# 軸ラベルの大きさを設定
ax.set_xlabel("x", size = 16)
ax.set_ylabel("y", size = 16)
ax.set_zlabel("z", size = 16)

# 円周率の定義
pi = np.pi

# (x,y)データを作成
x = np.linspace(-3*pi, 3*pi, 256)
y = np.linspace(-3*pi, 3*pi, 256)

# 格子点を作成
X, Y = np.meshgrid(x, y)

# 高度の計算式
Z = np.cos(X/pi) * np.sin(Y/pi)

# 曲面を描画
ax.plot_surface(X, Y, Z, cmap = "summer")

# 底面に等高線を描画
ax.contour(X, Y, Z, colors = "black", offset = -1)

plt.show()

# +
# PYTHON_MATPLOTLIB_3D_PLOT_02

# 3次元散布図

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("", size = 20)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

# 軸目盛を設定
ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])

# -5～5の乱数配列(100要素)
x = 10 * np.random.rand(100, 1) - 5
y = 10 * np.random.rand(100, 1) - 5
z = 10 * np.random.rand(100, 1) - 5

# 曲線を描画
ax.scatter(x, y, z, s = 40, c = "blue")

plt.show()
# +
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
# %matplotlib inline

ds = xr.tutorial.load_dataset('air_temperature')
air = ds.air.isel(time=0)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# The first plot (in kelvins) chooses "viridis" and uses the data's min/max
air.plot(ax=ax1, cbar_kwargs={'label': 'K'})
ax1.set_title('Kelvins: default')
ax2.set_xlabel('')

# The second plot (in celsius) now chooses "BuRd" and centers min/max around 0
airc = air - 273.15
airc.plot(ax=ax2, cbar_kwargs={'label': '°C'})
ax2.set_title('Celsius: default')
ax2.set_xlabel('')
ax2.set_ylabel('')

# The center doesn't have to be 0
air.plot(ax=ax3, center=273.15, cbar_kwargs={'label': 'K'})
ax3.set_title('Kelvins: center=273.15')

# Or it can be ignored
airc.plot(ax=ax4, center=False, cbar_kwargs={'label': '°C'})
ax4.set_title('Celsius: center=False')
ax4.set_ylabel('')

# Make it nice
plt.tight_layout()
# -



