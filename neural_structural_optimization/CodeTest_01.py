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
"""PixelModel.__init_"""
import numpy as np
import tensorflow as tf
nelz = 2
nely = 2
nelx = 5
volfrac = 0.4
mask = 4

shape = (nelz, nely, nelx)
z_init = np.broadcast_to(volfrac*mask, shape)
z = tf.Variable(z_init, trainable=True)

print(shape)
print(z_init)
print(z)

# +
"""problems.py PROBLEMS_BY_CATEGORY/NAME"""
from typing import Optional, Union
import dataclasses
import numpy as np
import skimage.draw

X, Y, Z = 0, 1, 2

@dataclasses.dataclass
class Problem:
    normals: np.ndarray
    forces: np.ndarray
    density: float
    mask: Union[np.ndarray, float] = 1
    name: Optional[str] = None
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    depth: int = dataclasses.field(init=False)	#2020-12-07 K.Taniguchi
    mirror_left: bool = dataclasses.field(init=False)
    mirror_right: bool = dataclasses.field(init=False)

    def __post_init__(self):
        self.width = self.normals.shape[0] - 1
        self.height = self.normals.shape[1] - 1
        self.depth = self.normals.shape[2] - 1	#2020-12-07 K.Taniguchi

        if self.normals.shape != (self.width + 1, self.height + 1, self.depth + 1, 3):	#2020-12-07 K.Taniguchi
            raise ValueError(f'normals has wrong shape: {self.normals.shape}')
        if self.forces.shape != (self.width + 1, self.height + 1, self.depth + 1, 3):	#2020-12-07 K.Taniguchi
            raise ValueError(f'forces has wrong shape: {self.forces.shape}')
        if (isinstance(self.mask, np.ndarray) and self.mask.shape != (self.height, self.width, self.depth)):	#2020-12-07 K.Taniguchi
            raise ValueError(f'mask has wrong shape: {self.mask.shape}')

#2020-12-07 K.Taniguchi
        self.mirror_left = (
            self.normals[0, :, :, X].all() and not self.normals[0, :, :, Y].all() and not self.normals[0, :, :, Z].all()
        )
        self.mirror_right = (
            self.normals[-1, :, :, X].all() and not self.normals[-1, :, :, Y].all() and not self.normals[-1, :, :, Y].all()
        )

def mbb_beam(width=60, height=20, depth=2, density=0.5):
    """Textbook beam example."""
    normals = np.zeros((width + 1, height + 1, depth + 1, 3))
    normals[-1, -1, :, Y] = 1
    normals[-1, -1, :, Z] = 1
    normals[0, :, :, X] = 1
    normals[0, :, :, Z] = 1

    forces = np.zeros((width + 1, height + 1, depth +1, 3))
    forces[0, 0, :, Y] = -1
    return Problem(normals, forces, density)

def cantilever_beam_full(width=60, height=60, depth=2, density=0.5, force_position=0):
    """Cantilever supported everywhere on the left."""
    # https://link.springer.com/content/pdf/10.1007%2Fs00158-010-0557-z.pdf
    normals = np.zeros((width + 1, height + 1, depth +1, 3))
    normals[0, :, :, :] = 1

    forces = np.zeros((width + 1, height + 1, depth + 1, 3))
    forces[-1, round((1 - force_position)*height), round(depth/2), Y] = -1
    return Problem(normals, forces, density)

PROBLEMS_BY_CATEGORY = {
    "mbb_beam":[
        mbb_beam(96, 32, 1, density=0.5),
        mbb_beam(192, 64, 2, density=0.4),
        mbb_beam(384, 128, 1, density=0.3),
        mbb_beam(192, 32, 1, density=0.5),
        mbb_beam(384, 64, 1, density=0.4),
        mbb_beam(2, 2, 2, density=0.4)
    ],
    "cantilever_beam_full":[
        cantilever_beam_full(96, 32, 1, density=0.4),
        cantilever_beam_full(192, 64, 1, density=0.3),
        cantilever_beam_full(384, 128, 1, density=0.2),
        cantilever_beam_full(384, 128, 1, density=0.15),
    ],
}

PROBLEMS_BY_NAME = {}
for problem_class, problem_list in PROBLEMS_BY_CATEGORY.items():
    for problem in problem_list:
        name = f'{problem_class}_{problem.width}x{problem.height}x{problem.depth}_{problem.density}'
        problem.name = name
        assert name not in PROBLEMS_BY_NAME, f'redundant name {name}'
        PROBLEMS_BY_NAME[name] = problem

problem = PROBLEMS_BY_NAME['mbb_beam_2x2x2_0.4']
max_iterations = 100
#problem = PROBLEMS_BY_NAME['mbb_beam_3x3x1_0.4']

print(problem)

# +
"""topo_api.py specified_task"""
problem = PROBLEMS_BY_NAME['mbb_beam_2x2x2_0.4']
#problem = PROBLEMS_BY_NAME['mbb_beam_3x3x1_0.4']

def specified_task(problem):
    """Given a problem, return parameters for running a topology optimization."""
    fixdofs = np.flatnonzero(problem.normals.ravel())
    alldofs = np.arange(3 * (problem.width + 1) * (problem.height + 1) * (problem.depth + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

    #print(problem.normals)
    #print(problem.normals.ravel())
    #print("problem name is ", problem.name)
    print("alldofs = ", alldofs)
    print("fixdofs = ", fixdofs)
    print("freedofs = ", freedofs)

    params = {
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
    }
    return params

args = specified_task(problem)

#print("fixdofs =", args['fixdofs'])
#print("freedofs =", args['freedofs'])
#print(args)
# +
"""models.py"""
import autograd
import autograd.core
import autograd.numpy as np
from neural_structural_optimization import topo_api
import tensorflow as tf

# requires tensorflow 2.0

layers = tf.keras.layers


def batched_topo_loss(params, envs):
  losses = [env.objective(params[i], volume_contraint=True)
            for i, env in enumerate(envs)]
  return np.stack(losses)

def convert_autograd_to_tensorflow(func):
  @tf.custom_gradient
  def wrapper(x):
    vjp, ans = autograd.core.make_vjp(func, x.numpy())
    return ans, vjp
  return wrapper

def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


class Model(tf.keras.Model):

  def __init__(self, seed=None, args=None):
    super().__init__()
    set_random_seed(seed)
    self.seed = seed
    self.env = topo_api.Environment(args)

  def loss(self, logits):
    # for our neural network, we use float32, but we use float64 for the physics
    # to avoid any chance of overflow.
    # add 0.0 to work-around bug in grad of tf.cast on NumPy arrays
    logits = 0.0 + tf.cast(logits, tf.float64)
    f = lambda x: batched_topo_loss(x, [self.env])
    losses = convert_autograd_to_tensorflow(f)(logits)
    return tf.reduce_mean(losses)


class PixelModel(Model):

  def __init__(self, seed=None, args=None):
    super().__init__(seed, args)
    shape = (self.env.args['nelz'], self.env.args['nely'], self.env.args['nelx'])
    z_init = np.broadcast_to(args['volfrac'] * args['mask'], shape)
    self.z = tf.Variable(z_init, trainable=True)

    print(z_init.size)

  def call(self, inputs=None):
    return self.z


def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
  return net


def UpSampling2D(factor):
  return layers.UpSampling2D((factor, factor), interpolation='bilinear')


def Conv2D(filters, kernel_size, **kwargs):
  return layers.Conv2D(filters, kernel_size, padding='same', **kwargs)


class AddOffset(layers.Layer):

  def __init__(self, scale=1):
    super().__init__()
    self.scale = scale

  def build(self, input_shape):
    self.bias = self.add_weight(
        shape=input_shape, initializer='zeros', trainable=True, name='bias')

  def call(self, inputs):
    return inputs + self.scale * self.bias


class CNNModel(Model):

  def __init__(
      self,
      seed=0,
      args=None,
      latent_size=128,
      dense_channels=32,
      resizes=(1, 2, 2, 2, 1),
      conv_filters=(128, 64, 32, 16, 1),
      offset_scale=10,
      kernel_size=(5, 5),
      latent_scale=1.0,
      dense_init_scale=1.0,
      activation=tf.nn.tanh,
      conv_initializer=tf.initializers.VarianceScaling,
      normalization=global_normalization,
  ):
    super().__init__(seed, args)

    if len(resizes) != len(conv_filters):
      raise ValueError('resizes and filters must be same size')

    activation = layers.Activation(activation)

    total_resize = int(np.prod(resizes))
    h = self.env.args['nely'] // total_resize
    w = self.env.args['nelx'] // total_resize

    net = inputs = layers.Input((latent_size,), batch_size=1)
    filters = h * w * dense_channels
    dense_initializer = tf.initializers.orthogonal(
        dense_init_scale * np.sqrt(max(filters / latent_size, 1)))
    net = layers.Dense(filters, kernel_initializer=dense_initializer)(net)
    net = layers.Reshape([h, w, dense_channels])(net)

    for resize, filters in zip(resizes, conv_filters):
      net = activation(net)
      net = UpSampling2D(resize)(net)
      net = normalization(net)
      net = Conv2D(
          filters, kernel_size, kernel_initializer=conv_initializer)(net)
      if offset_scale != 0:
        net = AddOffset(offset_scale)(net)

    outputs = tf.squeeze(net, axis=[-1])

    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z')

  def call(self, inputs=None):
    return self.core_model(self.z)


model = PixelModel(args=args)
print(model.env)
print(model.trainable_variables)



# +
"""train.py def method_of_asymptotes"""
import functools

from absl import logging
import autograd
import autograd.numpy as np
from neural_structural_optimization import models
from neural_structural_optimization import topo_physics
import scipy.optimize
import tensorflow as tf
import xarray
import numpy

#print('args =', args)
model = PixelModel(args=args)
max_iterations = 100

print('model :', model.trainable_variables)

def _get_variables(variables):
    return np.concatenate([
      v.numpy().ravel() if not isinstance(v, np.ndarray) else v.ravel()
      for v in variables])

def constrained_logits(init_model):
    """Produce matching initial conditions with volume constraints applied."""
    logits = init_model(None).numpy().astype(np.float64)
    print('logits.shape-before :', logits.shape)
    logits = init_model(None).numpy().astype(np.float64).squeeze(axis=0)
    print('logits.shape-after :', logits.shape)
    return topo_physics.physical_density(logits, init_model.env.args, volume_contraint=True, cone_filter=False)


def method_of_moving_asymptotes(model, max_iterations, save_intermediate_designs=True, init_model=None,):
    import nlopt  # pylint: disable=g-import-not-at-top

    #if not isinstance(model, models.PixelModel):
    #    raise ValueError('MMA only defined for pixel models')
        
    env = model.env
    if init_model is None:
        x0 = _get_variables(model.trainable_variables).astype(np.float64)
    else:
        x0 = constrained_logits(init_model).ravel()

    def objective(x):
        return env.objective(x, volume_contraint=False)

    def constraint(x):
        return env.constraint(x)

    def wrap_autograd_func(func, losses=None, frames=None):
        def wrapper(x, grad):
            if grad.size > 0:
                value, grad[:] = autograd.value_and_grad(func)(x)
            else:
                value = func(x)
            if losses is not None:
                losses.append(value)
            if frames is not None:
                frames.append(env.reshape(x).copy())
            return value
        return wrapper

    losses = []
    frames = []

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.set_lower_bounds(0.0)
    opt.set_upper_bounds(1.0)
    opt.set_min_objective(wrap_autograd_func(objective, losses, frames))
    opt.add_inequality_constraint(wrap_autograd_func(constraint), 1e-8)
    opt.set_maxeval(max_iterations + 1)
    opt.optimize(x0)

    designs = [env.render(x, volume_contraint=False) for x in frames]
    return optimizer_result_dataset(np.array(losses), np.array(designs), save_intermediate_designs)

#print('x0 =', x0)
ds_mma = method_of_moving_asymptotes(model, max_iterations)
print(ds_mma)
# -

print(x0)
print(x0.size)
print(x0.shape)
print('reshape:\n', x0.reshape(args['nelz'], args['nely'], args['nelx']))
print('size aft reshape:', x0.reshape(args['nelz'], args['nely'], args['nelx']).size)
print('shape aft reshape:', x0.reshape(args['nelz'], args['nely'], args['nelx']).shape)

# +
"""train.py def method_of_asymptotes(CONT'D)"""

max_iterations = 100

losses = []
frames = []

opt = nlopt.opt(nlopt.LD_MMA, x0.size)
opt.set_lower_bounds(0.0)
opt.set_upper_bounds(1.0)
opt.set_min_objective(wrap_autograd_func(objective, losses, frames))
opt.add_inequality_constraint(wrap_autograd_func(constraint), 1e-8)
opt.set_maxeval(max_iterations + 1)
opt.optimize(x0)

designs = [env.render(x, volume_contraint=False) for x in frames]
optimizer_result_dataset(np.array(losses), np.array(designs), save_intermediate_designs)

print(designs)
print(optimizer_result_dataset)


# +
"""train.py optimality_criteria"""
import functools

from absl import logging
import autograd
import autograd.numpy as np
from neural_structural_optimization import models
from neural_structural_optimization import topo_physics
import scipy.optimize
import tensorflow as tf
import xarray
import numpy

model = PixelModel(args=args)
max_iterations = 100
init_model = None

def _get_variables(variables):
    return np.concatenate([
        v.numpy().ravel() if not isinstance(v, np.ndarray) else v.ravel()
        for v in variables])

def optimality_criteria(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
):
#  if not isinstance(model, models.PixelModel):
#    raise ValueError('optimality criteria only defined for pixel models')

  env = model.env
  if init_model is None:
    x = _get_variables(model.trainable_variables).astype(np.float64)
  else:
    x = constrained_logits(init_model).ravel()

  # start with the first frame but not its loss, since optimality_criteria_step
  # returns the current loss and the *next* design.
  losses = []
  frames = [x]
  for _ in range(max_iterations):
    c, x = topo_physics.optimality_criteria_step(x, env.ke, env.args)
    losses.append(c)
    if np.isnan(c):
      # no point in continuing to optimize
      break
    frames.append(x)
  losses.append(env.objective(x, volume_contraint=False))

  designs = [env.render(x, volume_contraint=False) for x in frames]
  return optimizer_result_dataset(np.array(losses), np.array(designs),
                                  save_intermediate_designs)

print(model)
ds_oc = optimality_criteria(PixelModel(args=args), max_iterations)
print(ds_oc)
# -

print('z=', model.z)
print('z_dtype=', model.z.dtype)
print('assign=', model.z.assign)

# +
"""train.py train_lbfgs"""

print('x =', x)
print('x0 =', x0)
print('shape of x:', x.shape)
print(model)

def _set_variables(variables, x):
  shapes = [v.shape.as_list() for v in variables]
  values = tf.split(x, [np.prod(s) for s in shapes])
  for var, value in zip(variables, values):
    var.assign(tf.reshape(tf.cast(value, var.dtype), var.shape))

def train_lbfgs(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
    **kwargs
):
  model(None)  # build model, if not built

  losses = []
  frames = []

  if init_model is not None:
    if not isinstance(model, models.PixelModel):
      raise TypeError('can only use init_model for initializing a PixelModel')
    model.z.assign(tf.cast(init_model(None), model.z.dtype))

  tvars = model.trainable_variables

  def value_and_grad(x):
    _set_variables(tvars, x)
    with tf.GradientTape() as t:
      t.watch(tvars)
      logits = model(None)
      loss = model.loss(logits)
    grads = t.gradient(loss, tvars)
    frames.append(logits.numpy().copy())
    losses.append(loss.numpy().copy())
    return float(loss.numpy()), _get_variables(grads).astype(np.float64)

  x0 = _get_variables(tvars).astype(np.float64)
  # rely upon the step limit instead of error tolerance for finishing.
  _, _, info = scipy.optimize.fmin_l_bfgs_b(
      value_and_grad, x0, maxfun=max_iterations, factr=1, pgtol=1e-14, **kwargs
  )
  logging.info(info)

  designs = [model.env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(
      np.array(losses), np.array(designs), save_intermediate_designs)

ds_pix = train_lbfgs(model, max_iterations)
print(ds_pix)
