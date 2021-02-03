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
"""剛性マトリクス処理"""
# def get_k

import numpy as np

nelz = 64
nely = 64
nelx = 192
    
elz, ely, elx = np.meshgrid(range(nelz), range(nely), range(nelx))  # x, y, z coords
print("1")
print("1-1",elz)
print("1-2:",ely)
print("1-3:",elx)
    
elz, ely, elx = elz.reshape(-1, 1), ely.reshape(-1, 1), elx.reshape(-1, 1)
print("2")
print("2-1",elz)
print("2-2:",ely)
print("2-3:",elx)

n1 = (nely+1)*(elx+0) + (ely+0)
print("n1=",n1)
n2 = (nely+1)*(elx+1) + (ely+0)
print("n2=",n2)
n3 = (nely+1)*(elx+1) + (ely+1)
print("n3=",n3)
n4 = (nely+1)*(elx+0) + (ely+1)
print("n4=",n4)

n5 = (nelx+1)*(elz+0) + (elx+0) 
print("n5=",n5)
n6 = (nelx+1)*(elz+1) + (elx+0)
print("n6=",n6)
n7 = (nelx+1)*(elz+1) + (elx+1)
print("n7=",n7)
n8 = (nelx+1)*(elz+0) + (elx+1)
print("n8=",n8)

n9 = (nelz+1)*(ely+0) + (elz+0)
print("n9=",n9)
n10 = (nelz+1)*(ely+1) + (elz+0)
print("n10=",n10)
n11 = (nelz+1)*(ely+1) + (elz+1)
print("n11=",n11)
n12 = (nelz+1)*(ely+0) + (elz+1)
print("n12=",n12)

edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1,
                   2*n5, 2*n5+1, 2*n6, 2*n6+1, 2*n7, 2*n7+1, 2*n8, 2*n8+1,
                   2*n9, 2*n9+1, 2*n10, 2*n10+1, 2*n11, 2*n11+1, 2*n12, 2*n12+1])
print(edof)

edof = edof.T[0]
print(edof)
# +
"""剛性マトリクス定義"""
#get_stiffness_matrix

import numpy as np
import scipy.ndimage
import scipy.sparse

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
    print("n=",n)
    matrix = np.tri(n)
    print("matrix",matrix)
    coo_mat = scipy.sparse.coo_matrix(matrix)
    print("coo_mat.data=",coo_mat.data)
    print("coo_mat.row=",coo_mat.row)
    print("coo_mat.col=",coo_mat.col)
    ke0 = scipy.sparse.coo_matrix((ke,(coo_mat.row,coo_mat.col))).toarray()
    print(ke0)
    
    det = ke0.T
    print("ke0.T",det)
    
    ke0 = ke0 + det
    print("ke0+det=",ke0)
    
    diag = np.diag(ke0)
    print("diag=",diag)
    
    diag0 = np.diag(diag)/2
    print("diag0=",diag0)

    return ke0 - diag0

ke = get_stiffness_matrix(young=1.0, poisson=0.3)
print(ke)
# -


