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
# index map
import numpy as np

nelz, nely, nelx = 1, 1, 1
elz, ely, elx = np.meshgrid(range(nelz), range(nely), range(nelx))  # x, y coords

#print("nelz=",nelz,"nely=",nely,"nelx=",nelx)
#print("elz=",elz,"ely=",ely,"elx=",elx)

n1 = (nely+1)*(elx+0) + (ely+0) + (nelx+1)*(nely+1)*elz
n2 = (nely+1)*(elx+1) + (ely+0) + (nelx+1)*(nely+1)*elz
n3 = (nely+1)*(elx+1) + (ely+1) + (nelx+1)*(nely+1)*elz
n4 = (nely+1)*(elx+0) + (ely+1) + (nelx+1)*(nely+1)*elz

n5 = n1 + (nelx+1)*(nely+1)
n6 = n2 + (nelx+1)*(nely+1)
n7 = n3 + (nelx+1)*(nely+1)
n8 = n4 + (nelx+1)*(nely+1)

print("n1=",n1)
print("n2=",n2)
print("n3=",n3)
print("n4=",n4)
print("n5=",n5)
print("n6=",n6)
print("n7=",n7)
print("n8=",n8)
"""
#Old Edit.
edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1,
                 2*n5, 2*n5+1, 2*n6, 2*n6+1, 2*n7, 2*n7+1, 2*n8, 2*n8+1,])
"""
edof = np.array([3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2, 3*n3, 3*n3+1, 3*n3+2, 3*n4, 3*n4+1, 3*n4+2,
                 3*n5, 3*n5+1, 3*n5+2, 3*n6, 3*n6+1, 3*n6+2, 3*n7, 3*n7+1, 3*n7+2, 3*n8, 3*n8+1, 3*n8+2])

edof = edof.T[0]
print("EDOF",edof)
print("EDOF size, shape, dim", edof.size, edof.shape, edof.ndim)

#Num. of row/col in Element stiffness matrix is 24.
x_list = np.repeat(edof, 24)  # flat list pointer of each node in an element
y_list = np.tile(edof, 24).flatten()  # flat list pointer of each node in elem


print("x_list=",x_list, x_list.size, x_list.shape)
print("y_list=",y_list, y_list.size, y_list.shape)


kd = stiffness.T.reshape(nelx*nely*nelz, 1, 1)
value_list = (kd * np.tile(ke, kd.shape)).flatten()

print("kd=",kd)
print("value_list=",value_list)
# +
import numpy as np

nelz, nely, nelx = 64, 64, 192

nEl = nelx*nely*nelz
nDof = (nelx+1)*(nely+1)*(nelz+1)*3 
print("nEl=",nEl)
print("nDof=",nDof)
ANodes = np.meshgrid(range((1+nelx)*(1+nely)*(1+nelz)))
print("ANodes(list)=",ANodes)
ANodes = np.array(ANodes)
print("ANodes(array)=",ANodes)
nodeNrs = ANodes.reshape(1+nelz, 1+nely, 1+nelx)
print("nodeNrs",nodeNrs)
cVec = 3*nodeNrs[:nelz,:nely,:nelx]+4
cVec = cVec.reshape(nEl,-1)
print("cVec=",cVec,type(cVec))

L1 = [3*(nely+1)*(nelz+1)]*6
L2 = [3*(nely+1)]*3
L3 = [3*(nely+1)*(nelz+2)]*6
L4 = [3*(nely+1)]*3

print(L1,type(L1))
print(L2,type(L2))
print(L3,type(L3))
print(L4,type(L4))

cMat = cVec + np.concatenate([[0, 1, 2], L1+np.array([0, 1, 2, -3, -2, -1]), 
                [-3, -2, -1], L2+np.array([0, 1, 2]), 
                L3+np.array([0, 1, 2, -3, -2, -1]), 
                L4+np.array([-3, -2, -1])])

print("cMat=",cMat,cMat.shape)



# +
sI, sII, sII_ = list(range(24)), [], []
#np.meshgrid(range(24)), np.meshgrid(range(24))
#sI = np.array(sI).tolist()
#print(sI,sII)

#sI = np.repeat(sI, 24)
#sII = np.tile(sII, 24).flatten()

for i in range(23,0,-1):
    sI = np.concatenate([sI,sI[0:i]],0)

for j in range(0,24,1):
    sII_ = [j]*(24-j)
    sII.extend(sII_)
sII = np.array(sII)


print(type(cMat),cMat.shape)
ik, jk = [], []

for var in sI:
    ik_ = cMat[:, var]
    ik.extend(ik_)
ik = np.array(ik).T    

for var in sII:
    jk_ = cMat[:,var]
    jk.extend(jk_)
jk = np.array(jk).T

print("ik",ik.size)
print("jk",jk.size)
print("sI=",sI,sI.size,sI.dtype,type(sI))
print("sII=",sII,sII.size,sII.dtype,type(sII))
# +
# index map
import numpy as np

nelz, nely, nelx = 1, 1, 1

nEl = nelx*nely*nelz
nDof = (nelx+1)*(nely+1)*(nelz+1)*3 
ANodes = np.array(np.meshgrid(range((1+nelx)*(1+nely)*(1+nelz))))
nodeNrs = ANodes.reshape(1+nelz, 1+nely, 1+nelx)
cVec = (3*nodeNrs[:nelz,:nely,:nelx]+4).reshape(nEl,-1)

L1 = [3*(nely+1)*(nelz+1)]*6
L2 = [3*(nely+1)]*3
L3 = [3*(nely+1)*(nelz+2)]*6
L4 = [3*(nely+1)]*3

cMat = cVec + np.concatenate([[0, 1, 2], L1+np.array([0, 1, 2, -3, -2, -1]), 
                [-3, -2, -1], L2+np.array([0, 1, 2]), 
                L3+np.array([0, 1, 2, -3, -2, -1]), 
                L4+np.array([-3, -2, -1])])

sI, sII, sII_ = list(range(24)), [], []

for i in range(23,0,-1):
    sI = np.concatenate([sI,sI[0:i]],0)
    
for j in range(0,24,1):
    sII_ = [j]*(24-j)
    sII.extend(sII_)
sII = np.array(sII)
    
x_list = sII
y_list = sI

print(sI)
print(sII)

print("sI:",sI.size, sI.shape)
print("sII:",sII.size, sII.shape)

    
#kd = stiffness.T.reshape(nelx*nely*nelz, 1, 1)
#value_list = (kd * np.tile(ke, kd.shape)).flatten()

#print("kd=",kd)
#print("value_list=",value_list)

