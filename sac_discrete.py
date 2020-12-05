import os
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.auto import tqdm

from utils.logx import EpochLogger
from utils.tf_utils import set_tf_allow_growth

set_tf_allow_growth()

tfd = tfp.distributions
tfl = tfp.layers


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(shape=[size], dtype=np.int32)
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


class QNetwork(tf.keras.layers.Layer):
    """
    Both Q value and policy with shared features
    """

    def __init__(self, obs_dim, act_dim, mlp_hidden=256):
        super(QNetwork, self).__init__()
        self.num_ensembles = 2
        self.model = build_mlp_ensemble(input_dim=obs_dim, output_dim=act_dim, activation='relu',
                                        mlp_hidden=mlp_hidden, squeeze=False, num_ensembles=self.num_ensembles)
        self.build(input_shape=[(None, obs_dim)])

    def call(self, inputs, training=None):
        inputs = tf.tile(tf.expand_dims(inputs, axis=0), (self.num_ensembles, 1, 1))
        q = self.model(inputs)
        if training:
            return q
        else:
            return tf.reduce_min(q, axis=0)


class CatagoricalMLPActor(tf.keras.Model):
    def __init__(self, ob_dim, ac_dim, mlp_hidden):
        super(CatagoricalMLPActor, self).__init__()
        self.net = build_mlp(ob_dim, ac_dim, mlp_hidden)
        self.ac_dim = ac_dim
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Categorical(logits=t))
        self.build(input_shape=[(None, ob_dim), True])

    def call(self, inputs):
        inputs, deterministic = inputs
        # print(f'Tracing call with inputs={inputs}, deterministic={deterministic}')
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        pi_action = tf.cond(pred=deterministic, true_fn=lambda: tf.argmax(pi_distribution.probs_parameter(),
                                                                          axis=-1, output_type=tf.int32),
                            false_fn=lambda: pi_distribution.sample())
        return pi_action, pi_distribution


@tf.function
def hard_update(target: tf.keras.layers.Layer, source: tf.keras.layers.Layer):
    print('Tracing hard_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(source_param)


@tf.function
def soft_update(target: tf.keras.layers.Layer, source: tf.keras.layers.Layer, tau):
    print('Tracing soft_update_tf')
    for target_param, source_param in zip(target.variables, source.variables):
        target_param.assign(target_param * (1. - tau) + source_param * tau)


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


def gather_q_values(q_values, actions):
    batch_size = tf.shape(actions)[0]
    idx = tf.stack([tf.range(batch_size), actions], axis=-1)  # (None, 2)
    q1 = tf.gather_nd(q_values[0], indices=idx)
    q2 = tf.gather_nd(q_values[1], indices=idx)
    q_values = tf.stack([q1, q2], axis=0)
    return q_values


class LagrangeLayer(tf.keras.layers.Layer):
    def __init__(self, initial_value):
        super(LagrangeLayer, self).__init__()
        self.log_alpha = tf.Variable(initial_value=initial_value, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.nn.softplus(self.log_alpha)


class SACAgent(tf.keras.Model):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 mlp_hidden=256,
                 learning_rate=3e-4,
                 alpha=1.0,
                 gamma=0.99,
                 target_entropy=None,
                 huber_delta=None,
                 tau=5e-3
                 ):
        super(SACAgent, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.huber_delta = huber_delta
        self.tau = tau
        self.q_network = QNetwork(ob_dim, ac_dim, mlp_hidden=mlp_hidden)
        self.target_q_network = QNetwork(ob_dim, ac_dim, mlp_hidden=mlp_hidden)
        hard_update(self.target_q_network, self.q_network)
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.q_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        self.log_alpha = LagrangeLayer(initial_value=inverse_softplus(alpha))
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        target_entropy = np.log(ac_dim) if target_entropy is None else target_entropy
        self.target_entropy = tf.Variable(initial_value=target_entropy, dtype=tf.float32, trainable=False)

        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def set_target_entropy(self, target_entropy):
        print(f'Setting target entropy to {target_entropy:.4f}')
        target_entropy = tf.cast(target_entropy, dtype=tf.float32)
        self.target_entropy.assign(target_entropy)

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def _get_pi_distribution(self, obs):
        q_values = self.q_network(obs, training=False)
        q_values = q_values / self.log_alpha(obs)
        q_values = tf.stop_gradient(q_values)
        return tfd.Categorical(logits=q_values)

    @tf.function
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
        print('Tracing _update_nets')

        with tf.GradientTape() as q_tape, tf.GradientTape() as alpha_tape:
            q_tape.watch(self.q_network.trainable_variables)

            alpha = self.log_alpha(obs)
            # compute target Q value with double Q learning
            target_q_values = self.target_q_network(next_obs, training=False)  # (None, act_dim)
            next_policy = self._get_pi_distribution(obs)
            v = tf.reduce_sum(target_q_values * next_policy.probs_parameter(), axis=-1)
            policy_entropy = next_policy.entropy()
            target_q_values = v + alpha * policy_entropy
            q_target = reward + self.gamma * (1.0 - done) * target_q_values
            q_target = tf.stop_gradient(q_target)
            # compute Q and actor loss
            q_values = self.q_network(obs, training=True)  # (2, None, act_dim)
            # selection using actions
            q_values = gather_q_values(q_values, actions)  # (2, None)

            # q loss
            if self.huber_delta is not None:
                q_values_loss = huber_loss(tf.expand_dims(q_target, axis=0), q_values, delta=self.huber_delta)
            else:
                q_values_loss = 0.5 * tf.square(tf.expand_dims(q_target, axis=0) - q_values)

            q_values_loss = tf.reduce_sum(q_values_loss, axis=0)  # (None,)
            # apply importance weights
            q_values_loss = tf.reduce_mean(q_values_loss)

            alpha_loss = -tf.reduce_mean(alpha * (-policy_entropy + self.target_entropy))

        # update Q network
        q_gradients = q_tape.gradient(q_values_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network.trainable_variables))
        # update alpha network
        alpha_gradient = alpha_tape.gradient(alpha_loss, self.log_alpha.trainable_variables)
        self.alpha_optimizer.apply_gradients(zip(alpha_gradient, self.log_alpha.trainable_variables))

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=-policy_entropy,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
        )
        return info

    def update(self, obs, act, obs2, done, rew, update_target=True):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.int32)
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

    @tf.function
    def act_batch(self, obs, deterministic):
        print(f'Tracing sac act_batch with obs {obs}')
        pi_distribution = self._get_pi_distribution(obs)
        pi_final = tf.cond(pred=deterministic,
                           true_fn=lambda: tf.argmax(pi_distribution.probs_parameter(), axis=-1, output_type=tf.int32),
                           false_fn=lambda: pi_distribution.sample())
        return pi_final


def sac(env_name,
        env_fn=None,
        steps_per_epoch=5000,
        epochs=200,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        update_per_step=1,
        batch_size=256,
        num_test_episodes=10,
        logger_kwargs=dict(),
        seed=1,
        # sac args
        learning_rate=3e-4,
        alpha=0.2,
        tau=5e-3,
        gamma=0.99,
        # replay
        replay_size=int(1e6),
        save_freq=10,
        ):
    if env_fn is None:
        env_fn = lambda: gym.make(env_name)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name) if env_fn is None else env_fn()
    max_episode_steps = env._max_episode_steps
    print(f'max_episode_steps={max_episode_steps}')
    env.seed(seed)
    test_env = gym.vector.make(env_name, num_envs=num_test_episodes, asynchronous=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f'Observation dim: {obs_dim}. Action dim: {act_dim}')

    agent = SACAgent(ob_dim=obs_dim, ac_dim=act_dim,
                     learning_rate=learning_rate, alpha=alpha, gamma=gamma)
    agent.set_logger(logger)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    def get_action(o, deterministic=False):
        o = tf.convert_to_tensor(o, dtype=tf.float32)
        return agent.act(o, tf.convert_to_tensor(deterministic)).numpy()

    def get_action_batch(o, deterministic=False):
        o = tf.convert_to_tensor(o, dtype=tf.float32)
        return agent.act_batch(o, tf.convert_to_tensor(deterministic)).numpy()

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

    base_entropy = np.log(act_dim)
    # schedules for target_entropy
    target_entropy_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=base_entropy,
        decay_steps=1e5,
        end_learning_rate=base_entropy * 0.1
    )

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
        d = False if ep_len == max_episode_steps else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_episode_steps):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and (t + 1) % update_every == 0:
            for j in range(update_every * update_per_step):
                batch = replay_buffer.sample_batch(batch_size)
                agent.update(**batch, update_target=True)

        bar.update(1)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            bar.close()

            epoch = (t + 1) // steps_per_epoch

            if save_freq is not None:
                if epoch % save_freq == 0:
                    agent.save_weights(filepath=os.path.join(logger_kwargs['output_dir'], f'agent_final_{epoch}.ckpt'))

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # set target entropy
            agent.set_target_entropy(target_entropy_scheduler(t + 1))

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            agent.log_tabular()
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            if (t + 1) < total_steps:
                bar = tqdm(total=steps_per_epoch)

    agent.save_weights(filepath=os.path.join(logger_kwargs['output_dir'], f'agent_final.ckpt'))


if __name__ == '__main__':
    import argparse
    from utils.run_utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=1)
    # agent arguments
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    # training arguments
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--update_per_step', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=None)

    args = vars(parser.parse_args())

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_sac_test', data_dir='data', seed=args['seed'])

    sac(**args, logger_kwargs=logger_kwargs)
