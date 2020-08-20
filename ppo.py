"""
Proximal Policy Optimization
"""

import time

import gym.wrappers.time_limit
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.auto import trange

from envs.vector import SyncVectorEnv
from utils.logx import EpochLogger

tfl = tfp.layers
tfd = tfp.distributions


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, num_envs, length, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(shape=(num_envs, length, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(shape=(num_envs, length, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.rew_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.ret_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.val_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.logp_buf = np.zeros(shape=(num_envs, length), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.num_envs = num_envs
        self.max_size = length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.ptr, self.path_start_idx = 0, np.zeros(shape=(self.num_envs), dtype=np.int32)

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.val_buf[:, self.ptr] = val
        self.logp_buf[:, self.ptr] = logp
        self.ptr += 1

    def finish_path(self, dones, last_vals):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        for i in range(self.num_envs):
            if dones[i]:
                path_slice = slice(self.path_start_idx[i], self.ptr)
                rews = np.append(self.rew_buf[i, path_slice], last_vals[i])
                vals = np.append(self.val_buf[i, path_slice], last_vals[i])

                # the next two lines implement GAE-Lambda advantage calculation
                deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
                self.adv_buf[i, path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

                # the next line computes rewards-to-go, to be targets for the value function
                self.ret_buf[i, path_slice] = discount_cumsum(rews, self.gamma)[:-1]

                self.path_start_idx[i] = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        assert np.all(self.path_start_idx == self.ptr)
        self.reset()
        # ravel the data
        obs_buf = np.reshape(self.obs_buf, newshape=(-1, self.obs_dim))
        act_buf = np.reshape(self.act_buf, newshape=(-1, self.act_dim))
        ret_buf = np.reshape(self.ret_buf, newshape=(-1,))
        adv_buf = np.reshape(self.adv_buf, newshape=(-1,))
        logp_buf = np.reshape(self.logp_buf, newshape=(-1,))
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        data = dict(obs=obs_buf, act=act_buf, ret=ret_buf,
                    adv=adv_buf, logp=logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in data.items()}


class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(SqueezeLayer, self).__init__()
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, axis=self.axis)


def build_mlp(input_dim, output_dim, mlp_hidden, num_layers=3,
              activation='relu', out_activation=None, squeeze=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for _ in range(num_layers - 1):
        model.add(tf.keras.layers.Dense(mlp_hidden, activation=activation))
    model.add(tf.keras.layers.Dense(output_dim, activation=out_activation))
    if output_dim == 1 and squeeze is True:
        model.add(SqueezeLayer(axis=-1))
    return model


def make_truncated_normal_distribution(loc_params, scale_params):
    scale_params = tf.math.softplus(scale_params)
    loc_params = tf.tanh(loc_params)
    pi_distribution = tfd.Independent(distribution=tfd.TruncatedNormal(loc=loc_params, scale=scale_params,
                                                                       low=-1., high=1.),
                                      reinterpreted_batch_ndims=1)
    return pi_distribution


class TruncatedNormalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_lim, mlp_hidden):
        super(TruncatedNormalActor, self).__init__()
        self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim, mlp_hidden=mlp_hidden)
        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(act_dim))
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_truncated_normal_distribution(t, self.log_std))
        self.act_lim = act_lim

    def call(self, inputs):
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        return pi_distribution


class PPOAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_lim, mlp_hidden=64,
                 pi_lr=1e-3, vf_lr=1e-3, lam=1., clip_ratio=0.2,
                 entropy_coef=0.001, target_kl=0.05
                 ):
        """
        Args:
            policy_net: The policy net must implement following methods:
                - forward: takes obs and return action_distribution and value
                - forward_action: takes obs and return action_distribution
                - forward_value: takes obs and return value.
            The advantage is that we can save computation if we only need to fetch parts of the graph. Also, we can
            implement policy and value in both shared and non-shared way.
            learning_rate:
            lam:
            clip_param:
            entropy_coef:
            target_kl:
            max_grad_norm:
        """
        super(PPOAgent, self).__init__()
        self.policy_net = TruncatedNormalActor(obs_dim=obs_dim, act_dim=act_dim,
                                               act_lim=act_lim, mlp_hidden=mlp_hidden)
        self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)
        self.value_net = build_mlp(input_dim=obs_dim, output_dim=1, squeeze=True, mlp_hidden=mlp_hidden)
        self.value_net.compile(optimizer=self.v_optimizer, loss='mse')
        self.lam = lam

        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_lim = act_lim

        self.logger = None

    def _build_tf_function(self):
        self.act_batch = tf.function(func=self.act_batch, input_signature=[
            tf.TensorSpec(shape=[None, self.obs_dim], dtype=tf.float32)
        ])

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('PolicyLoss', average_only=True)
        self.logger.log_tabular('ValueLoss', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('AvgKL', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)

    def call(self, inputs, training=None, mask=None):
        pi_distribution = self.policy_net(inputs)
        pi_action = pi_distribution.sample()
        return pi_action

    def act_batch(self, obs):
        pi_distribution = self.policy_net(obs)
        pi_action = pi_distribution.sample()
        log_prob = pi_distribution.log_prob(pi_action)
        v = self.value_net(obs)
        return pi_action, log_prob, v

    @tf.function
    def _update_policy_step(self, obs, act, adv, old_log_prob):
        print(f'Tracing _update_policy_step with obs={obs}')
        with tf.GradientTape() as tape:
            distribution = self.policy_net(obs)
            entropy = tf.reduce_mean(distribution.entropy())
            log_prob = distribution.log_prob(act)
            negative_approx_kl = log_prob - old_log_prob
            approx_kl_mean = tf.reduce_mean(-negative_approx_kl)

            ratio = tf.exp(negative_approx_kl)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            loss = policy_loss - entropy * self.entropy_coef

        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

        info = dict(
            PolicyLoss=policy_loss,
            Entropy=entropy,
            AvgKL=approx_kl_mean,
        )
        return info

    def update_policy(self, obs, act, ret, adv, logp,
                      train_pi_iters=80, train_vf_iters=80):
        assert tf.is_tensor(obs), f'obs must be a tf tensor. Got {obs}'
        for i in range(train_pi_iters):
            info = self._update_policy_step(obs, act, adv, logp)
            if info['AvgKL'] > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

        self.logger.store(StopIter=i)

        for i in range(train_vf_iters):
            loss = self.value_net.train_on_batch(x=obs, y=ret)

        for key, item in info.items():
            info[key] = item.numpy()
        info['ValueLoss'] = loss

        self.logger.store(**info)


def ppo(env_name, env_fn=None, mlp_hidden=256, seed=0,
        steps_per_epoch=5000, epochs=200, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_vf_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.05, entropy_coef=1e-3, logger_kwargs=dict(), save_freq=10):
    if env_fn is None:
        env_fn = lambda: gym.make(env_name)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    assert steps_per_epoch % max_ep_len == 0
    num_envs = steps_per_epoch // max_ep_len
    env = SyncVectorEnv(env_fns=[env_fn for _ in range(num_envs)])
    env.seed(seed)

    dummy_env = env_fn()
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    act_lim = dummy_env.action_space.high[0]
    del dummy_env

    assert act_lim == 1., f'act_lim must be 1. Got {act_lim}'

    # Instantiate policy
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, act_lim=act_lim, mlp_hidden=mlp_hidden,
                     pi_lr=pi_lr, vf_lr=vf_lr, lam=lam, clip_ratio=clip_ratio,
                     entropy_coef=entropy_coef, target_kl=target_kl)
    agent.set_logger(logger)

    buffer = PPOBuffer(obs_dim=obs_dim, act_dim=act_dim, num_envs=num_envs, length=max_ep_len,
                       gamma=gamma, lam=lam)

    def collect_trajectories():
        obs = env.reset()
        ep_ret = np.zeros(shape=num_envs, dtype=np.float32)
        ep_len = np.zeros(shape=num_envs, dtype=np.int32)
        for t in trange(max_ep_len, desc='Collecting'):
            act, logp, val = agent.act_batch(tf.convert_to_tensor(obs, dtype=tf.float32))
            act = act.numpy()
            logp = logp.numpy()
            val = val.numpy()
            obs2, rew, dones, infos = env.step(act)
            buffer.store(obs, act, rew, val, logp)
            logger.store(VVals=val)
            ep_ret += rew
            ep_len += 1

            # There are four cases there:
            # 1. if done is False. Bootstrap (truncated due to trajectory length)
            # 2. if done is True, if TimeLimit.truncated not in info. Don't bootstrap (didn't truncate)
            # 3. if done is True, if TimeLimit.truncated in info, if it is True, Bootstrap (true truncated)
            # 4. if done is True, if TimeLimit.truncated in info, if it is False. Don't bootstrap (same time)

            if t == max_ep_len - 1:
                time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                                dtype=np.bool_)
                # need to finish path for all the environments
                last_vals = agent.value_net.predict(obs2)
                last_vals = last_vals * np.logical_or(np.logical_not(dones), time_truncated_dones)
                buffer.finish_path(dones=np.ones(shape=num_envs, dtype=np.bool_),
                                   last_vals=last_vals)
                logger.store(EpRet=ep_ret[dones], EpLen=ep_len[dones])
                obs = None
            elif np.any(dones):
                time_truncated_dones = np.array([info.get('TimeLimit.truncated', False) for info in infos],
                                                dtype=np.bool_)
                last_vals = agent.value_net.predict(obs2) * time_truncated_dones
                buffer.finish_path(dones=dones,
                                   last_vals=last_vals)
                logger.store(EpRet=ep_ret[dones], EpLen=ep_len[dones])
                ep_ret[dones] = 0.
                ep_len[dones] = 0
                obs = env.reset_done()

            else:
                obs = obs2

    start_time = time.time()

    for epoch in range(epochs):
        collect_trajectories()
        agent.update_policy(**buffer.get(),
                            train_pi_iters=train_pi_iters,
                            train_vf_iters=train_vf_iters)
        logger.log_tabular('Epoch', epoch + 1)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        agent.log_tabular()
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    from utils.run_utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1)

    args = vars(parser.parse_args())

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_ppo', data_dir='data', seed=args['seed'])
    ppo(**args, logger_kwargs=logger_kwargs)
