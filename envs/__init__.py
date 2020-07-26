import gym
from gym import register

from .wrappers import model_based_wrapper_dict, ModelBasedFn
from .sync_vector_env import SyncVectorEnv

print('Warning! Using truncated obs Ant-v2')
del gym.envs.registry.env_specs['Ant-v2']
print('Warning! Using truncated obs Humanoid-v2')
del gym.envs.registry.env_specs['Humanoid-v2']

register(
    id='Ant-v2',
    entry_point='envs.envs.ant:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='envs.envs.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)


def get_wrapper(env_name):
    return model_based_wrapper_dict[env_name]
