"""
Implement soft actor critic agent here
"""

import time

import gym
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from utils.logx import EpochLogger
from utils.tf_utils import set_tf_allow_growth

set_tf_allow_growth()

import tensorflow_probability as tfp

eps = 1e-6
tfd = tfp.distributions

soft_log_std_range = (-10., 5.)


def hard_update(target: tf.keras.Model, source: tf.keras.Model):
    target.set_weights(source.get_weights())


def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau):
    new_weights = []
    for target_weights, source_weights in zip(target.get_weights(), source.get_weights()):
        new_weights.append(target_weights * (1. - tau) + source_weights * tau)
    target.set_weights(new_weights)


def inverse_softplus(x, beta=1.):
    return np.log(np.exp(x * beta) - 1.) / beta


def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss.
    https://en.wikipedia.org/wiki/Huber_loss
    """
    error = y_true - y_pred
    cond = tf.abs(error) < delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)

    return tf.where(cond, squared_loss, linear_loss)


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


class EnsembleDense(tf.keras.layers.Dense):
    def __init__(self, num_ensembles, units, **kwargs):
        super(EnsembleDense, self).__init__(units=units, **kwargs)
        self.num_ensembles = num_ensembles

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.num_ensembles, last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.num_ensembles, 1, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        outputs = tf.linalg.matmul(inputs, self.kernel)  # (num_ensembles, None, units)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(SqueezeLayer, self).__init__()
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, axis=self.axis)


def build_mlp_ensemble(input_dim, output_dim, mlp_hidden, num_ensembles, num_layers=3,
                       activation='relu', out_activation=None, squeeze=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(batch_input_shape=(num_ensembles, None, input_dim)))
    for _ in range(num_layers - 1):
        model.add(EnsembleDense(num_ensembles, mlp_hidden, activation=activation))
    model.add(EnsembleDense(num_ensembles, output_dim, activation=out_activation))
    if output_dim == 1 and squeeze is True:
        model.add(SqueezeLayer(axis=-1))
    return model


def build_mlp(input_dim, output_dim, mlp_hidden, activation='relu', out_activation=None):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(mlp_hidden, activation=activation),
        tf.keras.layers.Dense(mlp_hidden, activation=activation),
        tf.keras.layers.Dense(output_dim, activation=out_activation),
    ])


def make_distribution(params):
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    scale_params = tf.clip_by_value(scale_params, clip_value_min=soft_log_std_range[0],
                                    clip_value_max=soft_log_std_range[1])
    scale_params = tf.math.softplus(scale_params)
    pi_distribution = tfd.Independent(distribution=tfd.Normal(loc=loc_params, scale=scale_params),
                                      reinterpreted_batch_ndims=1)
    return pi_distribution


class SquashedGaussianMLPActor(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, act_lim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim * 2, mlp_hidden)
        self.ac_dim = ac_dim
        self.act_lim = act_lim
        self.call = tf.function(func=self.call, input_signature=[
            (tf.TensorSpec(shape=[None, ob_dim], dtype=tf.float32),
             tf.TensorSpec(shape=(), dtype=tf.bool))
        ])

    def call(self, inputs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_distribution(t))(params)
        pi_action = tf.cond(pred=deterministic, true_fn=lambda: pi_distribution.mean(),
                            false_fn=lambda: pi_distribution.sample())
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi -= tf.reduce_sum(2. * (tf.math.log(2.) - pi_action - tf.math.softplus(-2. * pi_action)), axis=-1)
        pi_action_final = tf.tanh(pi_action) * self.act_lim
        return pi_action_final, logp_pi, pi_action, pi_distribution


class EnsembleQNet(tf.keras.Model):
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
        self.build(input_shape=[(None, ob_dim), (None, ac_dim)])

    def get_config(self):
        config = super(EnsembleQNet, self).get_config()
        config.update({
            'ob_dim': self.ob_dim,
            'ac_dim': self.ac_dim,
            'mlp_hidden': self.mlp_hidden,
            'num_ensembles': self.num_ensembles
        })
        return config

    def call(self, inputs, training=None, mask=None):
        obs, act = inputs
        inputs = tf.concat((obs, act), axis=-1)
        inputs = tf.tile(tf.expand_dims(inputs, axis=0), (self.num_ensembles, 1, 1))
        q = self.q_net(inputs)  # (num_ensembles, None)
        if training:
            return q
        else:
            return tf.reduce_min(q, axis=0)


class SACAgent(object):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 act_lim,
                 mlp_hidden=256,
                 learning_rate=3e-4,
                 alpha=1.0,
                 tau=5e-3,
                 gamma=0.99,
                 target_entropy=None,
                 huber_delta=None,
                 ):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.act_lim = act_lim
        self.mlp_hidden = mlp_hidden
        self.huber_delta = huber_delta
        self.policy_net = SquashedGaussianMLPActor(ob_dim, ac_dim, act_lim, mlp_hidden)
        self.q_network = EnsembleQNet(ob_dim, ac_dim, mlp_hidden)
        self.target_q_network = EnsembleQNet(ob_dim, ac_dim, mlp_hidden)
        hard_update(self.target_q_network, self.q_network)

        self.policy_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        self.log_alpha = tf.Variable(initial_value=inverse_softplus(alpha), dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.target_entropy = -ac_dim if target_entropy is None else target_entropy

        self.tau = tau
        self.gamma = gamma

        self.build_tf_function()

    def build_tf_function(self):
        self.act_batch = tf.function(func=self.act_batch, input_signature=[
            tf.TensorSpec(shape=[None, self.ob_dim], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ])

        self._update_nets = tf.function(func=self._update_nets, input_signature=[
            tf.TensorSpec(shape=[None, self.ob_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[None, self.ac_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[None, self.ob_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
        ])

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
        alpha = tf.math.softplus(self.log_alpha)

        next_action, next_action_log_prob, _, _ = self.policy_net((next_obs, False))
        target_q_values = self.target_q_network((next_obs, next_action), training=False) - alpha * next_action_log_prob
        q_target = reward + self.gamma * (1.0 - done) * target_q_values

        # q loss
        with tf.GradientTape() as q_tape:
            q_values = self.q_network((obs, actions), training=True)  # (num_ensembles, None)
            if self.huber_delta is not None:
                q_values_loss = huber_loss(tf.expand_dims(q_target, axis=0), q_values, delta=self.huber_delta)
            else:
                q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))

        # policy loss
        with tf.GradientTape() as policy_tape:
            action, log_prob, _, _ = self.policy_net((obs, False))
            q_values_pi_min = self.q_network((obs, action), training=False)
            policy_loss = tf.reduce_mean(log_prob * alpha - q_values_pi_min)
        policy_gradients = policy_tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy_net.trainable_variables))

        with tf.GradientTape() as alpha_tape:
            alpha = tf.math.softplus(self.log_alpha)
            alpha_loss = -tf.reduce_mean(alpha * (log_prob + self.target_entropy))
        alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients(zip([alpha_gradient], [self.log_alpha]))

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
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.float32)
        obs2 = tf.convert_to_tensor(obs2, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        rew = tf.convert_to_tensor(rew, dtype=tf.float32)

        info = self._update_nets(obs, act, obs2, done, rew)
        for key, item in info.items():
            info[key] = item.numpy()
        self.logger.store(**info)

        if update_target:
            self.update_target()

    def act(self, obs, deterministic):
        obs = tf.expand_dims(obs, axis=0)
        pi_final = self.act_batch(obs, deterministic)[0]
        return pi_final

    def act_batch(self, obs, deterministic):
        print(f'Tracing sac act_batch with obs {obs}')
        pi_final = self.policy_net((obs, deterministic))[0]
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

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = gym.vector.make(env_name, num_envs=num_test_episodes, asynchronous=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    agent = SACAgent(obs_dim, ac_dim=act_dim, act_lim=act_limit, mlp_hidden=nn_size,
                     learning_rate=learning_rate, alpha=alpha, tau=tau,
                     gamma=gamma)
    agent.set_logger(logger)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def get_action(o, deterministic=False):
        return agent.act(tf.convert_to_tensor(o, dtype=tf.float32), tf.convert_to_tensor(deterministic)).numpy()

    def get_action_batch(o, deterministic=False):
        return agent.act_batch(tf.convert_to_tensor(o, dtype=tf.float32), tf.convert_to_tensor(deterministic)).numpy()

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
    parser.add_argument('--nn_size', '-s', type=int, default=128)
    # training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_test_episodes', type=int, default=20)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--update_per_step', type=int, default=1)

    args = vars(parser.parse_args())

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_sac', data_dir='data', seed=args['seed'])

    sac(**args, logger_kwargs=logger_kwargs)
