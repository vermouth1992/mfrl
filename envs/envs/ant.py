import gym.envs.mujoco.ant as ant
import numpy as np


class AntEnv(ant.AntEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])
