"""
args
    # material properties
      'young': 1,
      'young_min': 1e-9,
      'poisson': 0.3,
      'g': 0,
    # constraints
      'volfrac': problem.density,
      'xmin': 0.001,
      'xmax': 1.0,
    # input parameters
      'nelx': problem.width,
      'nely': problem.height,
      'nelz': problem.depth,
      'mask': problem.mask,
      'freedofs': freedofs,
      'fixdofs': fixdofs,
      'forces': problem.forces.ravel(),
      'penal': 3.0,
      'filter_width': 2,
"""

import numpy as np

def train_lbfgs(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
    **kwargs
):
    model(None)  # build model, if not built

#Material and Continuation Parameters
    E0 = args['young']    #Young Modulus of Solid
    Emin = args['young_min']    #Young Modulus of "void"
    nu = args['poisson']    #Poisson Ratio
    penalCnt = {1, 1, 25, 0.25}    #Continuation Scheme on Penal
    betaCnt = {1, 1, 25, 2}    #Continuation Scheme on Beta
    if ftBC == 'N':
        bcF = 'symmetric'
    else:
        bcF = 0

#Discretization Features
    nEl = nelx * nely * nelz    #Number of Elements
    nodeNrs = reshape(0 : (nelx + 1) * (nely + 1) * (nelz + 1), 1 + nely, 1 + nelz, 1 + nelx)    #Nodes Numbering
    cVec = reshape(3 * nodeNrs(1 : nely, 1 : nelz, 1 : nelx) + 1, nEl, 1)    #3D
    cMat = cVec + ([0, 1, 2, 3 * (nely + 1) * (nelz + 1) + [0, 1, 2, -3, -2, -1], -3,-2, -1,
                    3 * (nely + 1) + [0, 1, 2], 3 * (nely + 1) * (nelz + 2) + [0, 1, 2, -3, -2, -1],
                    3 * (nely + 1) + [-3, -2, -1]])    #Connectivity Matrix
    nDof = (1 + nely) * (1 + nelz) * (1 + nelx)    #Total Number of DOFs
    sI = []
    sII = []

    for j in range(1, 24):
        sI = np.concatenate([sI, j:24], 1)
        sII = np.concatenate([sII, np.tile(j, (1, 24 - j + 1))], 1)

    [ik, jk] = (cMat(:, sI).T, cMAt(:,sII).T)    #??
    Iar = sort([ik(:), jk(:)], 2, 'descend')

    ke = np.multiply( E0/(1+nu)/(2*nu-1)/144, ([-32,-6,-6,8,6,6,10,6,3,-4,-6,-3,-4,-3,-6,10,
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
    idx = np.tril_indices(n, k=0, m=n)
    matrix = np.zeros((n,n)).astype(int)
    matrix[idx] = ke
    det = matrix.T
    matrix = matrix + det
    diag = np.diag(matrix)
    diag0 = np.diag(diag)/2

    Ke0 = matrix - diag0    #Recover full Matrix

    lcDof = 3 * nodeNrs(1 : nely + 1, 1, nelx + 1)
    fixed = 1 : 3 * (nely + 1) * (nelz + 1)

    [pasS, pasV] = ([],[])    #Passive Solid and Void Elements
    F =
