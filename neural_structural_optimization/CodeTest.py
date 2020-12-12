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
nelz = 1
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
    
def mbb_beam(width=60, height=20, depth=1, density=0.5):
    """Textbook beam example."""
    normals = np.zeros((width + 1, height + 1, depth + 1, 3))
    normals[-1, -1, :, Y] = 1
    normals[-1, -1, :, Z] = 1
    normals[0, :, :, X] = 1
    normals[0, :, :, Z] = 1
        
    forces = np.zeros((width + 1, height + 1, depth +1, 3))
    forces[0, 0, :, Y] = -1
    return Problem(normals, forces, density)

def cantilever_beam_full(width=60, height=60, depth=1, density=0.5, force_position=0):
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
        mbb_beam(192, 64, 1, density=0.4),
        mbb_beam(384, 128, 1, density=0.3),
        mbb_beam(192, 32, 1, density=0.5),
        mbb_beam(384, 64, 1, density=0.4),
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

problem = PROBLEMS_BY_NAME['mbb_beam_192x64x1_0.4']

print(problem)

# +
"""topo_api.py specified_task"""
problem = PROBLEMS_BY_NAME['mbb_beam_192x64x1_0.4']

def specified_task(problem):
    """Given a problem, return parameters for running a topology optimization."""
    fixdofs = np.flatnonzero(problem.normals.ravel())
    alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
    freedofs = np.sort(list(set(alldofs) - set(fixdofs)))
    
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

# -


