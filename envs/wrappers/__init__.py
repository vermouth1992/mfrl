from .base import ModelBasedFn

model_based_wrapper_dict = {}

from .classic import model_based_wrapper_dict as classic_dict
from .pybullets import model_based_wrapper_dict as pybullet_dict
from .mujoco import model_based_wrapper_dict as mujoco_dict

model_based_wrapper_dict.update(classic_dict)
model_based_wrapper_dict.update(pybullet_dict)
model_based_wrapper_dict.update(mujoco_dict)
