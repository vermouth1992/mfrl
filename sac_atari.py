"""
The code for the scheduler and the replay buffer is adapted from CS294-112 Spring 2017 HW3
"""

import os
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.wrappers import AtariPreprocessing, FrameStack
from tqdm.auto import tqdm

from utils.logx import EpochLogger
from utils.tf_utils import set_tf_allow_growth

set_tf_allow_growth()

tfd = tfp.distributions
tfl = tfp.layers


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        if outside_value is None:
            self._outside_value = self._endpoints[-1][-1]
        else:
            self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class ReplayBufferFrame(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        all_obs_batch = np.stack([self._encode_observation(idx + 1, self.frame_history_len + 1) for idx in idxes], 0)
        obs_batch = all_obs_batch[:, :, :, 0:self.frame_history_len]
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = all_obs_batch[:, :, :, 1:self.frame_history_len + 1]
        done_mask = self.done[idxes].astype(np.float32)

        return dict(
            obs=obs_batch,
            act=act_batch,
            obs2=next_obs_batch,
            done=done_mask,
            rew=rew_batch,
        )

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = np.random.choice(self.num_in_buffer - 1, size=batch_size, replace=True)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size, num_frames=self.frame_history_len)

    def _encode_observation(self, idx, num_frames):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - num_frames
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = num_frames - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Parameters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done


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


class QNetwork(tf.keras.layers.Layer):
    """
    Both Q value and policy with shared features
    """

    def __init__(self, obs_dim, act_dim, num_ensembles=2):
        super(QNetwork, self).__init__()
        self.features = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=obs_dim),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'),
            tf.keras.layers.Flatten()
        ])
        self.num_ensembles = num_ensembles
        self.q_feature = EnsembleDense(num_ensembles=num_ensembles, units=256, activation='relu')
        self.adv_fc = EnsembleDense(num_ensembles=num_ensembles, units=act_dim)
        self.value_fc = EnsembleDense(num_ensembles=num_ensembles, units=1)
        self.build(input_shape=[(None,) + obs_dim, True])

    def call(self, inputs, training=None):
        features = self.features(inputs, training=training)
        features = tf.tile(tf.expand_dims(features, axis=0), multiples=(self.num_ensembles, 1, 1))
        q_value = self.q_feature(features)
        adv = self.adv_fc(q_value)
        adv = adv - tf.reduce_mean(adv, axis=-1, keepdims=True)
        value = self.value_fc(q_value)
        q_value = value + adv
        if training:
            return q_value
        else:
            return tf.reduce_min(q_value, axis=0)


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


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


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
        self.log_alpha = tf.Variable(initial_value=inverse_softplus(initial_value), dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.nn.softplus(self.log_alpha)


class SACAgent(tf.keras.Model):
    def __init__(self,
                 ob_dim,
                 ac_dim,
                 learning_rate=3e-4,
                 alpha=1.0,
                 gamma=0.99,
                 target_entropy=None,
                 huber_delta=1.0,
                 tau=5e-3,
                 ):
        super(SACAgent, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.huber_delta = huber_delta
        self.tau = tau

        self.q_network = QNetwork(ob_dim, ac_dim)
        self.target_q_network = QNetwork(ob_dim, ac_dim)
        hard_update(self.target_q_network, self.q_network)

        self.q_optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        self.log_alpha = LagrangeLayer(initial_value=alpha)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        target_entropy = np.log(ac_dim) if target_entropy is None else target_entropy
        self.target_entropy = tf.Variable(initial_value=target_entropy, dtype=tf.float32, trainable=False)

        self.gamma = gamma

    def set_logger(self, logger):
        self.logger = logger

    def set_target_entropy(self, target_entropy):
        EpochLogger.log(f'Setting target entropy to {target_entropy:.4f}')
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

            alpha = self.log_alpha(next_obs)
            # compute target Q value with double Q learning
            target_q_values = self.target_q_network(next_obs, training=False)  # (None, act_dim)
            next_policy = self._get_pi_distribution(next_obs)
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

        self.update_target()

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=-policy_entropy,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
        )
        return info

    def update(self, obs, act, obs2, done, rew):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32) / 255.
        act = tf.convert_to_tensor(act, dtype=tf.int32)
        obs2 = tf.convert_to_tensor(obs2, dtype=tf.float32) / 255.
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        rew = tf.convert_to_tensor(rew, dtype=tf.float32)

        info = self._update_nets(obs, act, obs2, done, rew)
        for key, item in info.items():
            info[key] = item.numpy()
        self.logger.store(**info)

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

    frame_history_len = 4

    atari_preprocess_wrapper = lambda env: AtariPreprocessing(env=env, frame_skip=4 if 'NoFrameskip' in env_name else 1)
    frame_stack_wrapper = lambda env: FrameStack(env=env, num_stack=frame_history_len)

    env = gym.make(env_name) if env_fn is None else env_fn()
    max_episode_steps = env._max_episode_steps
    print(f'max_episode_steps={max_episode_steps}')
    env = atari_preprocess_wrapper(env)
    env.seed(seed)
    # test_env = gym.vector.make(env_name, num_envs=num_test_episodes, asynchronous=True,
    #                            wrappers=[atari_preprocess_wrapper, frame_stack_wrapper])
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    print(f'Observation dim: {obs_dim}. Action dim: {act_dim}')

    agent = SACAgent(ob_dim=obs_dim + (frame_history_len,), ac_dim=act_dim,
                     learning_rate=learning_rate, alpha=alpha, gamma=gamma, tau=tau)
    agent.set_logger(logger)
    replay_buffer = ReplayBufferFrame(size=replay_size, frame_history_len=frame_history_len)

    def get_action(o, deterministic=False):
        o = tf.convert_to_tensor(o, dtype=tf.float32) / 255.
        return agent.act(o, tf.convert_to_tensor(deterministic)).numpy()

    def get_action_batch(o, deterministic=False):
        o = tf.convert_to_tensor(o, dtype=tf.float32) / 255.
        return agent.act_batch(o, tf.convert_to_tensor(deterministic)).numpy()

    def test_agent():
        o, d, ep_ret, ep_len = test_env.reset(), np.zeros(shape=num_test_episodes, dtype=np.bool), \
                               np.zeros(shape=num_test_episodes), np.zeros(shape=num_test_episodes, dtype=np.int64)
        t = tqdm(total=1, desc='Testing')
        while not np.all(d):
            o = np.transpose(o, axes=(0, 2, 3, 1))  # (None, 84, 84, 4)
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
    bar = tqdm(total=steps_per_epoch, desc=f'Epoch {1}/{epochs}')

    base_entropy = np.log(act_dim)
    # schedules for target_entropy

    probs = [0.9] + [0.1 / (act_dim - 1)] * (act_dim - 1)
    epsilon_0_1 = tfd.Categorical(probs=probs).entropy().numpy().item()
    epsilon_0_0_1 = tfd.Categorical(probs=[0.99] + [0.01 / (act_dim - 1)] * (act_dim - 1)).entropy().numpy().item()

    print(f'Setting entropy: {base_entropy:.2f}, {epsilon_0_1:.2f}, {epsilon_0_0_1:.2f}')

    target_entropy_scheduler = PiecewiseSchedule(
        [
            (0, base_entropy),
            (1e6, epsilon_0_1),
            (total_steps / 2, epsilon_0_0_1),
        ], outside_value=epsilon_0_0_1
    )

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        idx = replay_buffer.store_frame(np.expand_dims(o, axis=-1))

        if t > start_steps:
            o = replay_buffer.encode_recent_observation()  # return the current o plus previous 3 frames
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
        replay_buffer.store_effect(idx, a, r, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_episode_steps):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and (t + 1) % update_every == 0:
            for j in range(int(update_every * update_per_step)):
                batch = replay_buffer.sample(batch_size)
                agent.update(**batch)

        bar.update(1)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            bar.close()

            epoch = (t + 1) // steps_per_epoch

            if save_freq is not None and epoch % save_freq == 0:
                agent.save_weights(filepath=os.path.join(logger_kwargs['output_dir'], f'agent_final_{epoch}.ckpt'))

            # Test the performance of the deterministic version of the agent.
            # test_agent()

            agent.set_target_entropy(target_entropy_scheduler.value(t + 1))

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            agent.log_tabular()
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            if t < total_steps:
                bar = tqdm(total=steps_per_epoch, desc=f'Epoch {epoch + 1}/{epochs}')

    agent.save_weights(filepath=os.path.join(logger_kwargs['output_dir'], f'agent_final.ckpt'))


if __name__ == '__main__':
    import argparse
    from utils.run_utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pong-v4')
    parser.add_argument('--seed', type=int, default=1)
    # agent arguments
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    # training arguments
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=100)
    parser.add_argument('--update_per_step', type=float, default=0.25)
    parser.add_argument('--save_freq', type=int, default=None)

    args = vars(parser.parse_args())

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_sac_test', data_dir='data', seed=args['seed'])

    sac(**args, logger_kwargs=logger_kwargs)
