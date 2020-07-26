import gym.envs.mujoco.humanoid as humanoid
import numpy as np


class HumanoidEnv(humanoid.HumanoidEnv):
    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               ])
