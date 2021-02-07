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
model = 1
step = 58
nelx = 8
nely = 8
nelz =8

a = np.arange(model*step*nelx*nely*nelz).reshape((model,step,nelx,nely,nelz))
print(a.size)
print("a=",a)
a = np.delete(a, 0, axis=1)
print("a'=",a)

# +
import numpy as np
model = 1
step = 2
nelx = 3
nely = 3
nelz =3

a = np.zeros(model*step*nelx*nely*nelz).reshape((model,step,nelx,nely,nelz))
print(a.size)
print("a=",a)
print("a_x=",a[:,:,0,:,:])
a = np.delete(a, [np.delete(a, [np.delete(a, [0])])])
print(a)
# +
import numpy as np

model = [1]
step = [2]
nelx = [3]
nely = [3]
nelz = [3]

b = np.zeros(model.size*step.size*nelx.size*nely.size*nelz.size)
print(b)
# -



