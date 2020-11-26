"""
Implement soft actor critic agent here
"""

import copy
import math
import time

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as td
from torch import nn
from tqdm.auto import tqdm

from utils.logx import EpochLogger

eps = 1e-6

soft_log_std_range = (-10., 5.)


def soft_update(target: nn.Module, source: nn.Module, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def inverse_softplus(x, beta=1.):
    return np.log(np.exp(x * beta) - 1.) / beta


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch


class EnsembleLinear(nn.Module):
    __constants__ = ['num_ensembles', 'in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, num_ensembles: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.num_ensembles = num_ensembles
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_ensembles, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_ensembles, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'num_ensembles={}, in_features={}, out_features={}, bias={}'.format(
            self.num_ensembles, self.in_features, self.out_features, self.bias is not None
        )


class SqueezeLayer(nn.Module):
    def __init__(self, dim=-1):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.squeeze(inputs, dim=self.dim)


def build_mlp_ensemble(input_dim, output_dim, mlp_hidden, num_ensembles, num_layers=3,
                       activation=nn.ReLU, out_activation=None, squeeze=False):
    layers = []
    if num_layers == 1:
        layers.append(EnsembleLinear(num_ensembles, input_dim, output_dim))
    else:
        layers.append(EnsembleLinear(num_ensembles, input_dim, mlp_hidden))
        layers.append(activation())
        for _ in range(num_layers - 2):
            layers.append(EnsembleLinear(num_ensembles, mlp_hidden, mlp_hidden))
            layers.append(activation())
        layers.append(EnsembleLinear(num_ensembles, mlp_hidden, output_dim))

    if out_activation is not None:
        layers.append(out_activation())
    if output_dim == 1 and squeeze is True:
        layers.append(SqueezeLayer(dim=-1))
    model = nn.Sequential(*layers)
    return model


def build_mlp(input_dim, output_dim, mlp_hidden, num_layers=3,
              activation=nn.ReLU, out_activation=None, squeeze=False):
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, mlp_hidden))
        layers.append(activation())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            layers.append(activation())
        layers.append(nn.Linear(mlp_hidden, output_dim))

    if out_activation is not None:
        layers.append(out_activation())
    if output_dim == 1 and squeeze is True:
        layers.append(SqueezeLayer(dim=-1))
    model = nn.Sequential(*layers)
    return model


def make_normal_distribution(params):
    loc_params, scale_params = torch.split(params, params.shape[-1] // 2, dim=-1)
    scale_params = torch.clip(scale_params, min=soft_log_std_range[0],
                              max=soft_log_std_range[1])
    scale_params = F.softplus(scale_params)
    pi_distribution = td.Independent(base_distribution=td.Normal(loc=loc_params, scale=scale_params),
                                     reinterpreted_batch_ndims=1)
    return pi_distribution


def apply_squash(log_pi, pi_action):
    log_pi -= torch.sum(2. * (math.log(2.) - pi_action - F.softplus(-2. * pi_action)), dim=-1)
    return log_pi


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = lambda param: make_normal_distribution(param)

    def select_action(self, inputs):
        inputs, deterministic = inputs
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        if deterministic:
            pi_action = pi_distribution.mean
        else:
            pi_action = pi_distribution.rsample()
        pi_action_final = torch.tanh(pi_action)
        return pi_action_final

    def forward(self, inputs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        if deterministic:
            pi_action = pi_distribution.mean
        else:
            pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi = apply_squash(logp_pi, pi_action)
        pi_action_final = torch.tanh(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution


class EnsembleQNet(nn.Module):
    def __init__(self, ob_dim, ac_dim, mlp_hidden, num_ensembles=2):
        super(EnsembleQNet, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.q_net = build_mlp_ensemble(input_dim=self.ob_dim + self.ac_dim,
                                        output_dim=1,
                                        mlp_hidden=self.mlp_hidden,
                                        num_ensembles=self.num_ensembles,
                                        num_layers=3,
                                        squeeze=True)

    def forward(self, inputs, training=None):
        obs, act = inputs
        inputs = torch.cat((obs, act), dim=-1)
        inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
        inputs = inputs.repeat(self.num_ensembles, 1, 1)
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return torch.min(q, dim=0)[0]


class LagrangeLayer(nn.Module):
    def __init__(self, initial_value=0.):
        super(LagrangeLayer, self).__init__()
        self.log_alpha = nn.Parameter(data=torch.as_tensor(inverse_softplus(initial_value), dtype=torch.float32))

    def forward(self):
        return F.softplus(self.log_alpha)


class SACAgent(nn.Module):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 mlp_hidden=256,
                 learning_rate=3e-4,
                 alpha=1.0,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 ):
        super(SACAgent, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.mlp_hidden = mlp_hidden
        self.policy_net = SquashedGaussianMLPActor(ob_dim, ac_dim, mlp_hidden)
        self.q_network = EnsembleQNet(ob_dim, ac_dim, mlp_hidden)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.alpha_net = LagrangeLayer(initial_value=alpha)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=learning_rate)
        self.alpha_optimizer = torch.optim.Adam(params=self.alpha_net.parameters(), lr=learning_rate)
        self.target_entropy = -ac_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=False)
        self.logger.log_tabular('Q2Vals', with_min_and_max=False)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def _update_nets(self, obs, actions, next_obs, done, reward):
        """ Sample a mini-batch from replay buffer and update the network

        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)

        Returns: None

        """
        with torch.no_grad():
            alpha = self.alpha_net()
            next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
            target_q_values = self.target_q_network((next_obs, next_action),
                                                    training=False) - alpha * next_action_log_prob
            q_target = reward + self.gamma * (1.0 - done) * target_q_values

        # q loss
        q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
        q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)
        # (num_ensembles, None)
        q_values_loss = torch.sum(q_values_loss, dim=0)  # (None,)
        # apply importance weights
        q_values_loss = torch.mean(q_values_loss)
        self.q_optimizer.zero_grad()
        q_values_loss.backward()
        self.q_optimizer.step()

        # policy loss
        action, log_prob, _, _ = self.policy_net((obs, False))
        q_values_pi_min = self.q_network((obs, action), training=False)
        policy_loss = torch.mean(log_prob * alpha - q_values_pi_min)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha = self.alpha_net()
        alpha_loss = -torch.mean(alpha * (log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
        )
        return info

    def update(self, obs, act, obs2, done, rew, update_target=True):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        obs2 = torch.as_tensor(obs2, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)

        info = self._update_nets(obs, act, obs2, done, rew)
        for key, item in info.items():
            info[key] = item.detach().cpu().numpy()
        self.logger.store(**info)

        if update_target:
            self.update_target()

    def act(self, obs, deterministic):
        with torch.no_grad():
            obs = torch.unsqueeze(obs, dim=0)
            pi_final = self.act_batch(obs, deterministic)[0]
            return pi_final

    def act_batch(self, obs, deterministic):
        with torch.no_grad():
            pi_final = self.policy_net.select_action((obs, deterministic))
            return pi_final


def sac(env_name,
        env_fn=None,
        max_ep_len=1000,
        steps_per_epoch=5000,
        epochs=200,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        update_per_step=1,
        batch_size=256,
        num_test_episodes=20,
        logger_kwargs=dict(),
        seed=1,
        # sac args
        nn_size=256,
        learning_rate=3e-4,
        alpha=0.2,
        tau=5e-3,
        gamma=0.99,
        # replay
        replay_size=int(1e6),
        ):
    if env_fn is None:
        env_fn = lambda: gym.make(env_name)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.set_deterministic(True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name) if env_fn is None else env_fn()
    env.seed(seed)
    test_env = gym.vector.make(env_name, num_envs=num_test_episodes, asynchronous=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]
    assert act_limit == 1.0

    agent = SACAgent(obs_dim, ac_dim=act_dim, mlp_hidden=nn_size,
                     learning_rate=learning_rate, alpha=alpha, tau=tau,
                     gamma=gamma)
    agent.device = torch.device("cuda")
    agent.to(agent.device)
    agent.set_logger(logger)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def get_action(o, deterministic=False):
        return agent.act(torch.as_tensor(o, dtype=torch.float32, device=agent.device), deterministic).cpu().numpy()

    def get_action_batch(o, deterministic=False):
        return agent.act_batch(torch.as_tensor(o, dtype=torch.float32, device=agent.device),
                               deterministic).cpu().numpy()

    def test_agent():
        o, d, ep_ret, ep_len = test_env.reset(), np.zeros(shape=num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=num_test_episodes), np.zeros(shape=num_test_episodes, dtype=np.int64)
        t = tqdm(total=1, desc='Testing')
        while not np.all(d):
            a = get_action_batch(o, deterministic=True)
            o, r, d_, _ = test_env.step(a)
            ep_ret = r * (1 - d) + ep_ret
            ep_len = np.ones(shape=num_test_episodes, dtype=np.int64) * (1 - d) + ep_len
            d = np.logical_or(d, d_)
        t.update(1)
        t.close()
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    bar = tqdm(total=steps_per_epoch)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every * update_per_step):
                batch = replay_buffer.sample_batch(batch_size)
                agent.update(**batch, update_target=True)

        bar.update(1)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            bar.close()

            epoch = (t + 1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            agent.log_tabular()
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            if t < total_steps:
                bar = tqdm(total=steps_per_epoch)


if __name__ == '__main__':
    import argparse
    from utils.run_utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1)
    # agent arguments
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nn_size', '-s', type=int, default=256)
    # training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_test_episodes', type=int, default=20)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--update_per_step', type=int, default=1)

    args = vars(parser.parse_args())

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_sac_pytorch_test',
                                        data_dir='data', seed=args['seed'])

    sac(**args, logger_kwargs=logger_kwargs)
