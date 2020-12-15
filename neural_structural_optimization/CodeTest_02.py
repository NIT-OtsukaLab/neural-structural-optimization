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


import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

nelx = 4
nely = 3
nelz = 2
radius = 2
mask = 1


# Cone filter
def _cone_filter_matrix_3D(nelx, nely, nelz, radius, mask):
  x, y, z = np.meshgrid(np.arange(nelx), np.arange(nely), np.arange(nelz), indexing='ij')

  rows = []
  cols = []
  values = []
  r_bound = int(np.ceil(radius))
  for dx in range(-r_bound, r_bound+1):
    for dy in range(-r_bound, r_bound+1):
      for dz in range(-r_bound, r_bound+1):
        weight = np.maximum(0, radius - np.sqrt(dx**2 + dy**2 + dz**2))
        #weight = np.maximum(0, radius - np.sqrt(dx**2 + dy**2))
        row = x + nelx * y
        column = x + dx + nelx * (y + dy)
        value = np.broadcast_to(weight, x.shape)

        # exclude cells beyond the boundary
        valid = (
            (mask > 0) &
            ((x+dx) >= 0) &
            ((x+dx) < nelx) &
            ((y+dy) >= 0) &
            ((y+dy) < nely) &
            ((z+dz) >= 0) &
            ((z+dz) < nelz)
        )
        rows.append(row[valid])
        cols.append(column[valid])
        values.append(value[valid])

  data = np.concatenate(values)
  i = np.concatenate(rows)
  j = np.concatenate(cols)
  return scipy.sparse.coo_matrix((data, (i, j)), (nelx * nely * nelz,) * 2)
  #return scipy.sparse.coo_matrix((data, (i, j)), (nelx * nely,) * 2)

def _cone_filter_matrix(nelx, nely, radius, mask):
  x, y = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')

  rows = []
  cols = []
  values = []
  r_bound = int(np.ceil(radius))
  for dx in range(-r_bound, r_bound+1):
    for dy in range(-r_bound, r_bound+1):
      weight = np.maximum(0, radius - np.sqrt(dx**2 + dy**2))
      row = x + nelx * y
      column = x + dx + nelx * (y + dy)
      value = np.broadcast_to(weight, x.shape)

      # exclude cells beyond the boundary
      valid = (
          (mask > 0) &
          ((x+dx) >= 0) &
          ((x+dx) < nelx) &
          ((y+dy) >= 0) &
          ((y+dy) < nely)
      )
      rows.append(row[valid])
      cols.append(column[valid])
      values.append(value[valid])

  data = np.concatenate(values)
  i = np.concatenate(rows)
  j = np.concatenate(cols)

  return scipy.sparse.coo_matrix((data, (i, j)), (nelx * nely,) * 2)

result2D = _cone_filter_matrix(nelx, nely, radius, mask)
result3D = _cone_filter_matrix(nelx, nely, nelz, radius, mask)

print(result2D)

print(result3D)
