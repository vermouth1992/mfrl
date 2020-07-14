import unittest

from utils.run_utils import ExperimentGrid
from sac import sac

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
        run_exp(sac, 'Hopper-v2', self.algo)

    def test_walker(self):
        run_exp(sac, 'Walker2d-v2', self.algo)

    def test_halfcheetah(self):
        run_exp(sac, 'HalfCheetah-v2', self.algo, params={
            'epochs': 600
        })

    def test_ant(self):
        run_exp(sac, 'Ant-v2', self.algo, params={
            'epochs': 600
        })

    def test_humanoid(self):
        run_exp(sac, 'Humanoid-v2', self.algo, params={
            'epochs': 2000
        })


class TD3(unittest.TestCase):
    def test_hopper(self):
        grid = ExperimentGrid('Hopper_sac')

    def test_walker(self):
        pass

    def test_halfcheetah(self):
        pass

    def test_ant(self):
        pass

    def test_humanoid(self):
        pass


if __name__ == '__main__':
    unittest.main()
