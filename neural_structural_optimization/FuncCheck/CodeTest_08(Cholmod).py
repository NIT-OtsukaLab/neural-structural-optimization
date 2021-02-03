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
import numpy as np
import scipy.ndimage
import scipy.sparse as ssp
import sksparse.cholmod
import scipy.sparse.linalg

params = {
    # material properties
    'young': 70000,
    'young_min': 1e-9,
    'poisson': 0.33,
    'g': 0,
    # constraints
    'volfrac': 0.4,
    'xmin': 0.001,
    'xmax': 1.0,
    # input parameters
    'nelx': 192,
    'nely': 64,
    'nelz': 64,
    'mask': 1,
#    'freedofs': freedofs,
#    'fixdofs': fixdofs,
#    'forces': problem.forces.ravel(),
    'penal': 3.0,
    'filter_width': 2,
}

def get_stiffness_matrix(young, poisson):	#20201219 K.Taniguchi
    # Element stiffness matrix
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
    matrix = np.tri(n)
    cm = scipy.sparse.coo_matrix(matrix)
    ke0 = scipy.sparse.coo_matrix((ke,(cm.row,cm.col))).toarray()
    det = ke0.T
    ke0 = ke0 + det
    diag = np.diag(ke0)
    diag0 = np.diag(diag)/2

    print("diag=",diag/2)
    
    return ke0 - diag0

ke1 = get_stiffness_matrix(young=1.0, poisson=0.3)
print("ke1=",ke1)
print("ke1.shape=",ke1.shape)
print("ke1.T.shape=",ke1.T.shape)
print("ke1.size=",ke1.size)
print("ke1.ndim=",ke1.ndim)
ke = ssp.csc_matrix(ke1)
print("ke.typoe=",type(ke))
# -
#2D KE
e, nu = 1, 0.3
k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
ke2d = e/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
                              ])
w, v = np.linalg.eig(ke2d)
print("固有値配列 w=",w)
print("2d_KE=",ke2d)
print("diag_2d",np.diag(ke2d))

# +
"""
print(np.all(ke1 == ke1.T))
print(np.array_equal(ke1, ke1.T))
print(np.array_equiv(ke1, ke1.T))
"""

w, v = np.linalg.eig(ke1)
print("固有値配列 w=",w)
print("固有ベクトル v=",v)

# +
SolveA = sksparse.cholmod.cholesky(ke).solve_A

print("cholesky=",SolveA)


# +
def get_k(stiffness, ke):
  print("get_k")
  # Constructs a sparse stiffness matrix, k, for use in the displace function.
  nelz, nely, nelx = stiffness.shape
  print("nelx=",nelx)
  print("nely=",nely)
  print("nely=",nelz)
    
  # get position of the nodes of each element in the stiffness matrix
  elz, ely, elx = np.meshgrid(range(nelz), range(nely), range(nelx))  # x, y, z coords
  elz, ely, elx = elz.reshape(-1, 1), ely.reshape(-1, 1), elx.reshape(-1, 1)

  print("elx=",elx)
  print("ely=",ely)
  print("elz=",elz)

  n1 = (nely+1)*(elx+0) + (ely+0)
  n2 = (nely+1)*(elx+1) + (ely+0)
  n3 = (nely+1)*(elx+1) + (ely+1)
  n4 = (nely+1)*(elx+0) + (ely+1)

  n5 = (nelx+1)*(elz+0) + (elx+0)
  n6 = (nelx+1)*(elz+1) + (elx+0)
  n7 = (nelx+1)*(elz+1) + (elx+1)
  n8 = (nelx+1)*(elz+0) + (elx+1)

  n9 = (nelz+1)*(ely+0) + (elz+0)
  n10 = (nelz+1)*(ely+1) + (elz+0)
  n11 = (nelz+1)*(ely+1) + (elz+1)
  n12 = (nelz+1)*(ely+0) + (elz+1)

  edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1,
                   2*n5, 2*n5+1, 2*n6, 2*n6+1, 2*n7, 2*n7+1, 2*n8, 2*n8+1,
                   2*n9, 2*n9+1, 2*n10, 2*n10+1, 2*n11, 2*n11+1, 2*n12, 2*n12+1])
  edof = edof.T[0]

  print("EDOF size, shape, dim", edof.size, edof.shape, edof.ndim)

#Num. of row/col in Element stiffness matrix is 24.
  x_list = np.repeat(edof, 24)  # flat list pointer of each node in an element
  y_list = np.tile(edof, 24).flatten()  # flat list pointer of each node in elem

  print("x_list=",x_list, x_list.shape)
  print("y_list=",y_list, y_list.shape)

  # make the stiffness matrix
  print("stiffness.shape=",stiffness.shape)
  kd = stiffness.T.reshape(nelx*nely*nelz, 1, 1)
  print("stiffness.T.reshape=",stiffness.T.reshape)
  value_list = (kd * np.tile(ke, kd.shape)).flatten()
  print("value_list=",value_list, value_list.shape)

  return value_list, y_list, x_list
