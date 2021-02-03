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
"""3D配列→2D配列→Sparse行列処理(2D)→3D配列"""

import numpy as np

a_2d = [[0, 1, 2], [3, 4, 5]]
arr_2d = np.array(a_2d)
print("arr_2d=",arr_2d)
print("ndim, shape, size")
print(arr_2d.ndim, arr_2d.shape, arr_2d.size)

a_3d = [[[0,1],[2,3],[4,5]]]
arr_3d = np.array(a_3d)
print("arr_3d=",arr_3d)
print("ndim, shape, size")
print(arr_3d.ndim, arr_3d.shape, arr_3d.size)

arr_3d_2d=np.reshape(arr_3d,(2,-1))
print("arr_3d_2d=",arr_3d_2d)

#arr_2d_3d=np.reshape(arr_3d_2d,(3,-1))
#print("arr_2d_3d=",arr_2d_3d)

# +
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

coo_2d = scipy.sparse.coo_matrix(a_2d)
print("Sparse化")
print(coo_2d)

#coo_3d = scipy.sparse.coo_matrix(a_3d)
# → TypeError: expected dimension <= 2 array or matrix

print("data=",coo_2d.data)
print("row=",coo_2d.row)
print("col=",coo_2d.col)


# +
shape = coo_2d.shape
print(shape)

coo_3d = scipy.sparse.coo_matrix((coo_2d.data,(coo_2d.row,coo_2d.col)),shape).toarray()
print("Sparseから復元(toarray)")
print(coo_3d)
coo_3d = scipy.sparse.coo_matrix((coo_2d.data,(coo_2d.row,coo_2d.col)),shape).tocsr()
print("Sparseから復元(tocsr)")
print(coo_3d)
# -


