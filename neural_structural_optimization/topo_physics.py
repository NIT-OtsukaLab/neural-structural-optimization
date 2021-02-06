# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Autograd implementation of topology optimization for compliance minimization.

Exactly reproduces the result of "Efficient topology optimization in MATLAB
using 88 lines of code":
http://www.topopt.mek.dtu.dk/Apps-and-software/Efficient-topology-optimization-in-MATLAB
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=superfluous-parens

import autograd
import autograd.numpy as np
import scipy.sparse
import math
from neural_structural_optimization import autograd_lib
from neural_structural_optimization import caching

# A note on conventions:
# - forces and freedofs are stored flattened, but logically represent arrays of
#   shape (Y+1, X+1, 2)
# - mask is either a scalar (1) or an array of shape (X, Y).
# Yes, this is confusing. Sorry!


def default_args():
  # select the degrees of freedom
  nelz = 25
  nely = 25
  nelx = 80

  left_wall = list(range(0, 2*(nely+1), 2))
  right_corner = [2*(nelx+1)*(nely+1)-1]
  fixdofs = np.asarray(left_wall + right_corner)
  alldofs = np.arange(3*(nelz+1)*(nely+1)*(nelx+1))
  freedofs = np.asarray(list(set(alldofs) - set(fixdofs)))

  forces = np.zeros(2*(nelz+1)*(nely+1)*(nelx+1))
  forces[1] = -1.0

  return {'young': 1,     # material properties
          'young_min': 1e-9,
          'poisson': 0.3,
          'g': 0,  # force of gravity
          'volfrac': 0.4,  # constraints
          'nelx': nelx,     # input parameters
          'nely': nely,
          'nelz': nelz,
          'freedofs': freedofs,
          'fixdofs': fixdofs,
          'forces': forces,
          'mask': 1,
          'penal': 3.0,
          'rmin': 1.5,
          'opt_steps': 50,
          'filter_width': 3,
          'step_size': 0.5,
          'name': 'truss'}


def physical_density(x, args, volume_contraint=False, cone_filter=False):   #cone_filter=Truee
  #print("...physical_density")
  shape = (args['nelz'], args['nely'], args['nelx'])
  assert x.shape == shape or x.ndim == 1
  x = x.reshape(shape)
  if volume_contraint:
    mask = np.broadcast_to(args['mask'], x.shape) > 0
    x_designed = sigmoid_with_constrained_mean(x[mask], args['volfrac'])
    x_flat = autograd_lib.scatter1d(
        x_designed, np.flatnonzero(mask), x.size)
    x = x_flat.reshape(x.shape)
  else:
    x = x * args['mask']
  if cone_filter:
    x = autograd_lib.cone_filter(x, args['filter_width'], args['mask'])
  return x


def mean_density(x, args, volume_contraint=False, cone_filter=True):
  return (np.mean(physical_density(x, args, volume_contraint, cone_filter))
          / np.mean(args['mask']))


def get_stiffness_matrix(young, poisson):	#20201219 K.Taniguchi
  # Element stiffness matrix
  #print("...get_stiffness_matrix")
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
  ke = ke0 - diag0

  return ke

  """
  k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return e/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
                              ])
  """


@caching.ndarray_safe_lru_cache(1)
def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
  #print("..._get_dof_indices")
  index_map = autograd_lib.inverse_permutation(
      np.concatenate([freedofs, fixdofs]))
  keep = np.isin(k_xlist, freedofs) & np.isin(k_ylist, freedofs)
  i = index_map[k_ylist][keep]
  j = index_map[k_xlist][keep]

  return index_map, keep, np.stack([i, j])


def displace(x_phys, ke, forces, freedofs, fixdofs, *,
             penal=3, e_min=1e-9, e_0=1):
  #print("...displace")
  # Displaces the load x using finite element techniques. The spsolve here
  # occupies the majority of this entire simulation's runtime.
  stiffness = young_modulus(x_phys, e_0, e_min, p=penal)
  k_entries, k_ylist, k_xlist = get_k(stiffness, ke)
  index_map, keep, indices = _get_dof_indices(
      freedofs, fixdofs, k_ylist, k_xlist
  )
  u_nonzero = autograd_lib.solve_coo(k_entries[keep], indices, forces[freedofs],
                                     sym_pos=True)
  u_values = np.concatenate([u_nonzero, np.zeros(len(fixdofs))])

  return u_values[index_map]


def get_k(stiffness, ke):
### Revised Program Start ###
  #print("...get_k")
  # Constructs a sparse stiffness matrix, k, for use in the displace function.
  """
  nelz, nely, nelx = stiffness.shape
  nEl = nelx*nely*nelz
  nDof = (nelx+1)*(nely+1)*(nelz+1)*3
  ANodes = np.meshgrid(range((1+nelx)*(1+nely)*(1+nelz)))
  ANodes = np.array(ANodes)
  nodeNrs = ANodes.reshape(1+nelz, 1+nely, 1+nelx)
  cVec = 3*nodeNrs[:nelz,:nely,:nelx]+4
  cVec = cVec.reshape(nEl,-1)

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

  ik, jk = [], []
  for var in sI:
    ik_ = cMat[:, var]
    ik.extend(ik_)
  for var in sII:
    jk_ = cMat[:, var]
    jk.extend(jk_)

  Iar = sorted([ik[:],jk[:]], reverse=True)   #list形式
  Iar = np.array(Iar)

  ik = Iar[0,:]
  jk = Iar[1,:]
  #ik = np.array(ik).T
  #jk = np.array(jk).T

  x_list = jk
  y_list = ik

  ke = ke.reshape([-1,1])
  stiffness = stiffness.reshape([1,-1])

  # make the stiffness matrix
  #print("stiffness",stiffness.size)
  #kd = stiffness.T.reshape(nEl, 1, 1)
  kd = np.reshape(ke*stiffness,[ke.size*nEl,1]).squeeze(-1)

  #size compensation
  if max(max(ik),max(jk))>nDof:
      a = max(max(ik),max(jk))
  else:
      a = nDof
  print("size compensation a=",a)
  value_list = scipy.sparse.coo_matrix((kd,(ik,jk)),shape=([a,a])).toarray().flatten()
  #value_list = scipy.sparse.coo_matrix((kd,(ik,jk)),shape=(nDof,nDof)).toarray().flatten()
  #value_list = (kd * np.tile(ke, kd.shape)).flatten()
  print(value_list)
  print("'get_k' is done")
  return value_list, y_list, x_list

  """
  # Constructs a sparse stiffness matrix, k, for use in the displace function.
  nelz, nely, nelx = stiffness.shape

  # get position of the nodes of each element in the stiffness matrix
  elz, ely, elx = np.meshgrid(range(nelz), range(nely), range(nelx))  # x, y, z coords
  elz, ely, elx = elz.reshape(-1, 1), ely.reshape(-1, 1), elx.reshape(-1, 1)

  n1 = (nely+1)*(elx+0) + (ely+0) + (nelx+1)*(nely+1)*elz
  n2 = (nely+1)*(elx+1) + (ely+0) + (nelx+1)*(nely+1)*elz
  n3 = (nely+1)*(elx+1) + (ely+1) + (nelx+1)*(nely+1)*elz
  n4 = (nely+1)*(elx+0) + (ely+1) + (nelx+1)*(nely+1)*elz

  n5 = n1 + (nelx+1)*(nely+1)
  n6 = n2 + (nelx+1)*(nely+1)
  n7 = n3 + (nelx+1)*(nely+1)
  n8 = n4 + (nelx+1)*(nely+1)

  #edof = np.array([3*n1, 3*n1+1, 3*n2, 3*n2+1, 3*n3, 3*n3+1, 3*n4, 3*n4+1,
  #                 3*n5, 3*n5+1, 3*n6, 3*n6+1, 3*n7, 3*n7+1, 3*n8, 3*n8+1,])

  edof = np.array([3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2, 3*n3, 3*n3+1, 3*n3+2,
                   3*n4, 3*n4+1, 3*n4+2, 3*n5, 3*n5+1, 3*n5+2, 3*n6, 3*n6+1, 3*n6+2,
                   3*n7, 3*n7+1, 3*n7+2, 3*n8, 3*n8+1, 3*n8+2])

  edof = edof.T[0]

#Num. of row/col in Element stiffness matrix is 24.
  x_list = np.repeat(edof, 24)  # flat list pointer of each node in an element
  y_list = np.tile(edof, 24).flatten()  # flat list pointer of each node in elem

  kd = stiffness.T.reshape(nelx*nely*nelz, 1, 1)
  value_list = (kd*np.tile(ke, kd.shape)).flatten()

  return value_list, y_list, x_list


def young_modulus(x, e_0, e_min, p=3):
  #print("...young_modulus")
  return e_min + x ** p * (e_0 - e_min)


def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1):
  # Calculates the compliance
  # Read about how this was vectorized here:
  # https://colab.research.google.com/drive/1PE-otq5hAMMi_q9dC6DkRvf2xzVhWVQ4

  # index map
  #print("...compliance")
  nelz, nely, nelx = x_phys.shape
  elz, ely, elx = np.meshgrid(range(nelz), range(nely), range(nelx))  # x, y, z coords
  elz, ely, elx = elz.reshape(-1, 1), ely.reshape(-1, 1), elx.reshape(-1, 1)

  n1 = (nely+1)*(elx+0) + (ely+0) + (nelx+1)*(nely+1)*elz
  n2 = (nely+1)*(elx+1) + (ely+0) + (nelx+1)*(nely+1)*elz
  n3 = (nely+1)*(elx+1) + (ely+1) + (nelx+1)*(nely+1)*elz
  n4 = (nely+1)*(elx+0) + (ely+1) + (nelx+1)*(nely+1)*elz

  n5 = n1 + (nelx+1)*(nely+1)
  n6 = n2 + (nelx+1)*(nely+1)
  n7 = n3 + (nelx+1)*(nely+1)
  n8 = n4 + (nelx+1)*(nely+1)

  #edof = np.array([3*n1, 3*n1+1, 3*n2, 3*n2+1, 3*n3, 3*n3+1, 3*n4, 3*n4+1,
  #                 3*n5, 3*n5+1, 3*n6, 3*n6+1, 3*n7, 3*n7+1, 3*n8, 3*n8+1,])

  all_ixs = np.array([3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2, 3*n3, 3*n3+1, 3*n3+2,
                   3*n4, 3*n4+1, 3*n4+2, 3*n5, 3*n5+1, 3*n5+2, 3*n6, 3*n6+1, 3*n6+2,
                   3*n7, 3*n7+1, 3*n7+2, 3*n8, 3*n8+1, 3*n8+2])
  """
  nEl = nelx*nely*nelz
  ANodes = np.array(np.meshgrid(range((1+nelx)*(1+nely)*(1+nelz))))
  nodeNrs = ANodes.reshape(1+nelz, 1+nely, 1+nelx)
  cVec = (3*nodeNrs[:nelz,:nely,:nelx]+4).reshape(nEl,-1)

  all_ixs = cVec + np.concatenate([[0, 1, 2], L1+np.array([0, 1, 2, -3, -2, -1]),
                  [-3, -2, -1], L2+np.array([0, 1, 2]),
                  L3+np.array([0, 1, 2, -3, -2, -1]),
                  L4+np.array([-3, -2, -1])])
  """

  # select from u matrix
  u_selected = u[all_ixs]

  # compute x^penal * U.T @ ke @ U in a vectorized way
  ke_u = np.einsum('ij,jkl->ikl', ke, u_selected)
  ce = np.einsum('ijk,ijk->jk', u_selected, ke_u)
  x_phys = x_phys.reshape(-1,1)
  C = young_modulus(x_phys, e_0, e_min, p=penal) * ce.T
  return np.sum(C)


def optimality_criteria_combine(x, dc, dv, args, max_move=0.2, eta=0.5):
  """Fully differentiable version of the optimality criteria."""

  volfrac = args['volfrac']

  def pack(x, dc, dv):
    return np.concatenate([x.ravel(), dc.ravel(), dv.ravel()])

  def unpack(inputs):
    x_flat, dc_flat, dv_flat = np.split(inputs, [x.size, x.size + dc.size])
    return (x_flat.reshape(x.shape),
            dc_flat.reshape(dc.shape),
            dv_flat.reshape(dv.shape))

  def compute_xnew(inputs, lambda_):
    x, dc, dv = unpack(inputs)
    # avoid dividing by zero outside the design region
    dv = np.where(np.ravel(args['mask']) > 0, dv, 1)
    # square root is not defined for negative numbers, which can happen due to
    # small numerical errors in the computed gradients.
    xnew = x * np.maximum(-dc / (lambda_ * dv), 0) ** eta
    lower = np.maximum(0.0, x - max_move)
    upper = np.minimum(1.0, x + max_move)
    # note: autograd does not define gradients for np.clip
    return np.minimum(np.maximum(xnew, lower), upper)

  def f(inputs, lambda_):
    xnew = compute_xnew(inputs, lambda_)
    return volfrac - mean_density(xnew, args)

  # find_root allows us to differentiate through the while loop.
  inputs = pack(x, dc, dv)
  lambda_ = autograd_lib.find_root(f, inputs, lower_bound=1e-9, upper_bound=1e9)
  return compute_xnew(inputs, lambda_)


def sigmoid(x):
  return np.tanh(0.5*x)*.5 + 0.5


def logit(p):
  p = np.clip(p, 0, 1)
  return np.log(p) - np.log1p(-p)


# an alternative to the optimality criteria
def sigmoid_with_constrained_mean(x, average):
  f = lambda x, y: sigmoid(x + y).mean() - average
  lower_bound = logit(average) - np.max(x)
  upper_bound = logit(average) - np.min(x)
  b = autograd_lib.find_root(f, x, lower_bound, upper_bound)
  return sigmoid(x + b)


def calculate_forces(x_phys, args):
  applied_force = args['forces']

  if not args.get('g'):
    return applied_force

  density = 0
  for pad_left in [0, 1]:
    for pad_up in [0, 1]:
      padding = [(pad_left, 1 - pad_left), (pad_up, 1 - pad_up)]
      density += (1/4) * np.pad(
          x_phys.T, padding, mode='constant', constant_values=0
      )
  gravitional_force = -args['g'] * density[..., np.newaxis] * np.array([0, 1])
  return applied_force + gravitional_force.ravel()


def objective(x, ke, args, volume_contraint=False, cone_filter=True):
  #print("...objective")
  """Objective function (compliance) for topology optimization."""
  kwargs = dict(penal=args['penal'], e_min=args['young_min'], e_0=args['young'])
#  x_phys = physical_density(x, args, volume_contraint=volume_contraint, cone_filter=cone_filter)
  x_phys = physical_density(x, args, volume_contraint=volume_contraint, cone_filter=False)
  forces = calculate_forces(x_phys, args)
  u = displace(x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs)
  c = compliance(x_phys, u, ke, **kwargs)
  return c


def optimality_criteria_step(x, ke, args):
  """Heuristic topology optimization, as described in the 88 lines paper."""
  c, dc = autograd.value_and_grad(objective)(x, ke, args)
  dv = autograd.grad(mean_density)(x, args)
  x = optimality_criteria_combine(x, dc, dv, args)
  return c, x


def run_toposim(x=None, args=None, loss_only=True, verbose=True):
  # Root function that runs the full optimization routine
  if args is None:
    args = default_args()
  if x is None:
    x = np.ones((args['nely'], args['nelx'])) * args['volfrac']  # init mass

  if not loss_only:
    frames = [x.copy()]
  ke = get_stiffness_matrix(args['young'], args['poisson'])  # stiffness matrix

  losses = []
  for step in range(args['opt_steps']):
    c, x = optimality_criteria_step(x, ke, args)
    losses.append(c)

    if not loss_only and verbose and step % 5 == 0:
      print('step {}, loss {:.2e}'.format(step, c))

    if not loss_only:
      frames.append(x.copy())

  return losses[-1] if loss_only else (losses, x, frames)
