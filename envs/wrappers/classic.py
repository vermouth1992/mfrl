import numpy as np
import tensorflow as tf
import torch

from .base import ModelBasedFn


class PendulumFn(ModelBasedFn):
    env_name = 'Pendulum-v0'
    terminate = True
    reward = True

    def cost_fn_tf_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = tf.atan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs

    def cost_fn_numpy_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = np.arctan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs

    def cost_fn_torch_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = torch.atan2(sin_th, cos_th)
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs


import sys
import inspect

model_based_wrapper_dict = {}


def register():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            model_based_wrapper_dict[obj().env_name] = obj


if len(model_based_wrapper_dict) == 0:
    register()
