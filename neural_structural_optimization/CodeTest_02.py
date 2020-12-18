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
        print(x,'+',nelx,'*',y,'=',row)

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

nelx = 1
nely = 2
nelz = 2
radius = 2
mask = 1

#result2D = _cone_filter_matrix(nelx, nely, radius, mask)
result3D = _cone_filter_matrix_3D(nelx, nely, nelz, radius, mask).row

# +
#print(result2D)
#print(result2D, '\n', 'ccccccccccccccccccccccccccccccccccc', '\n', result3D)
# +
arr = np.array([10,20,40,46,33,14,12,46,52,30,59,18,11,22,30,2,11,58,22,72,12])
n = int(np.sqrt(len(arr)*2))

idx = np.tril_indices(n, k=0, m=n)
matrix = np.zeros((n,n)).astype(int)
matrix[idx] = arr

idx_ = np.triu_indices(n, k=0, m=n)
matrix_ = np.zeros((n,n)).astype(int)
matrix_[idx_] = arr

matrix = matrix + matrix_
print(matrix)

diag = np.diag(matrix)
matrix = matrix - diag

print(matrix)
print(diag)

# +
"""Stiffness Matrix"""

young = 18
poisson = 1

e, nu = young, poisson
ke = np.multiply( e/(1+nu)/(2*nu-1)/144, ([-32,-6,-6,8,6,6,10,6,3,-4,-6,-3,-4,-3,-6,10,
    3,6,8,3,3,4,-3,-3, -32,-6,-6,-4,-3,6,10,3,6,8,6,-3,-4,-6,-3,4,-3,3,8,3,
    3,10,6,-32,-6,-3,-4,-3,-3,4,-3,-6,-4,6,6,8,6,3,10,3,3,8,3,6,10,-32,6,6,
    -4,6,3,10,-6,-3,10,-3,-6,-4,3,6,4,3,3,8,-3,-3,-32,-6,-6,8,6,-6,10,3,3,4,
    -3,3,-4,-6,-3,10,6,-3,8,3,-32,3,-6,-4,3,-3,4,-6,3,10,-6,6,8,-3,6,10,-3,
    3,8,-32,-6,6,8,6,-6,8,3,-3,4,-3,3,-4,-3,6,10,3,-6,-32,6,-6,-4,3,3,8,-3,
    3,10,-6,-3,-4,6,-3,4,3,-32,6,3,-4,-3,-3,8,-3,-6,10,-6,-6,8,-6,-3,10,-32,
    6,-6,4,3,-3,8,-3,3,10,-3,6,-4,3,-6,-32,6,-3,10,-6,-3,8,-3,3,4,3,3,-4,6,
    -32,3,-6,10,3,-3,8,6,-3,10,6,-6,8,-32,-6,6,8,6,-6,10,6,-3,-4,-6,3,-32,6,
    -6,-4,3,6,10,-3,6,8,-6,-32,6,3,-4,3,3,4,3,6,-4,-32,6,-6,-4,6,-3,10,-6,3,
    -32,6,-6,8,-6,-6,10,-3,-32,-3,6,-4,-3,3,4,-32,-6,-6,8,6,6,-32,-6,-6,-4,
    -3,-32,-6,-3,-4,-32,6,6,-32,-6,-32] + np.multiply(nu, [48,0,0,0,-24,-24,-12,0,-12,0,
    24,0,0,0,24,-12,-12,0,-12,0,0,-12,12,12,48,0,24,0,0,0,-12,-12,-24,0,-24,
    0,0,24,12,-12,12,0,-12,0,-12,-12,0,48,24,0,0,12,12,-12,0,24,0,-24,-24,0,
    0,-12,-12,0,0,-12,-12,0,-12,48,0,0,0,-24,0,-12,0,12,-12,12,0,0,0,-24,
    -12,-12,-12,-12,0,0,48,0,24,0,-24,0,-12,-12,-12,-12,12,0,0,24,12,-12,0,
    0,-12,0,48,0,24,0,-12,12,-12,0,-12,-12,24,-24,0,12,0,-12,0,0,-12,48,0,0,
    0,-24,24,-12,0,0,-12,12,-12,0,0,-24,-12,-12,0,48,0,24,0,0,0,-12,0,-12,
    -12,0,0,0,-24,12,-12,-12,48,-24,0,0,0,0,-12,12,0,-12,24,24,0,0,12,-12,
    48,0,0,-12,-12,12,-12,0,0,-12,12,0,0,0,24,48,0,12,-12,0,0,-12,0,-12,-12,
    -12,0,0,-24,48,-12,0,-12,0,0,-12,0,12,-12,-24,24,0,48,0,0,0,-24,24,-12,
    0,12,0,24,0,48,0,24,0,0,0,-12,12,-24,0,24,48,-24,0,0,-12,-12,-12,0,-24,
    0,48,0,0,0,-24,0,-12,0,-12,48,0,24,0,24,0,-12,12,48,0,-24,0,12,-12,-12,
    48,0,0,0,-24,-24,48,0,24,0,0,48,24,0,0,48,0,0,48,0,48])))

n = int(np.sqrt(len(ke)*2))
idxu = np.triu_indices(n, k=0, m=n)
idxl = np.tril_indices(n, k=0, m=n)
matrixu = np.zeros((n,n)).astype(float)
matrixl = np.zeros((n,n)).astype(float)

matrixu[idxu] = ke
matrixl[idxl] = ke

matrix = matrixu + matrixl

print(matrix)

matrix0 = matrix - np.diag(matrixu)

print(np.diag(matrix))
#print(matrix0)


# +
young = 1
poisson = 0.3

e, nu = young, poisson
k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])

ke = e/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
                              ])
print(ke.dtype)

# -


