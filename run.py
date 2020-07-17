import unittest

from sac import sac
from td3 import td3
from utils.run_utils import ExperimentGrid

__all__ = ['sac', 'td3']


def run_exp(thunk, env_name, algo, params=dict()):
    grid = ExperimentGrid(f'{env_name}_{algo}')
    grid.add('seed', [1, 11, 21, 31, 41])
    for key, item in params.items():
        grid.add(key, item)
    grid.add('env_name', env_name)
    grid.run(thunk=thunk, data_dir='data', )


class SAC(unittest.TestCase):
    algo = 'sac'

    def test_hopper(self):
        run_exp(eval(self.algo), 'Hopper-v2', self.algo)

    def test_walker(self):
        run_exp(eval(self.algo), 'Walker2d-v2', self.algo)

    def test_halfcheetah(self):
        run_exp(eval(self.algo), 'HalfCheetah-v2', self.algo, params={
            'epochs': 600
        })

    def test_ant(self):
        run_exp(eval(self.algo), 'Ant-v2', self.algo, params={
            'epochs': 600
        })

    def test_humanoid(self):
        run_exp(eval(self.algo), 'Humanoid-v2', self.algo, params={
            'epochs': 2000
        })

    def test_pybullet_hopper(self):
        run_exp(eval(self.algo), 'pybullet_envs:HopperBulletEnv-v0', self.algo)

    def test_pybullet_walker(self):
        run_exp(eval(self.algo), 'pybullet_envs:Walker2DBulletEnv-v0', self.algo)

    def test_pybullet_halfcheetah(self):
        run_exp(eval(self.algo), 'pybullet_envs:HalfCheetahBulletEnv-v0', self.algo, params={
            'epochs': 600
        })

    def test_pybullet_ant(self):
        run_exp(eval(self.algo), 'pybullet_envs:AntBulletEnv-v0', self.algo, params={
            'epochs': 600
        })

    def test_pybullet_humanoid(self):
        run_exp(eval(self.algo), 'pybullet_envs:HumanoidBulletEnv-v0', self.algo, params={
            'epochs': 2000
        })


class TD3(SAC):
    algo = 'td3'


if __name__ == '__main__':
    unittest.main()
