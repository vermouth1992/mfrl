"""
Trust Region Policy Optimization (Schulman et al, 2017)
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


@tf.function
def flat_grads(grads):
    print(f'Tracing flat_grads grads={len(grads)}')
    grads = [tf.reshape(grad, shape=(-1,)) for grad in grads]
    return tf.concat(grads, axis=0)


@tf.function
def get_flat_params_from(model: tf.keras.Model):
    print(f'Tracing get_flat_params_from model={model.name}')
    params = [tf.reshape(p, shape=(-1,)) for p in model.trainable_variables]
    flat_params = tf.concat(params, axis=0)
    return flat_params


@tf.function
def set_flat_params_to(model: tf.keras.Model, flat_params):
    print(f'Tracing set_flat_params_to model={model.name}, flat_params={len(flat_params)}')
    prev_ind = 0
    for param in model.trainable_variables:
        flat_size = tf.reduce_prod(param.shape)
        param.assign(tf.reshape(flat_params[prev_ind:prev_ind + flat_size], shape=param.shape))
        prev_ind += flat_size


class GAEBuffer:
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
        self.gamma, self.lam = gamma, lam
        self.num_envs = num_envs
        self.max_size = length
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.ptr, self.path_start_idx = 0, np.zeros(shape=(self.num_envs), dtype=np.int32)

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.val_buf[:, self.ptr] = val
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
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        data = dict(obs=obs_buf, act=act_buf, ret=ret_buf, adv=adv_buf)
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


def make_normal_distribution(loc_params, scale_params):
    scale_params = tf.math.softplus(scale_params)
    loc_params = tf.tanh(loc_params)
    pi_distribution = tfd.Independent(distribution=tfd.Normal(loc=loc_params, scale=scale_params),
                                      reinterpreted_batch_ndims=1)
    return pi_distribution


class NormalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_lim, mlp_hidden):
        super(NormalActor, self).__init__()
        self.net = build_mlp(input_dim=obs_dim, output_dim=act_dim, mlp_hidden=mlp_hidden)
        self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(act_dim))
        self.pi_dist_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: make_normal_distribution(t, self.log_std))
        self.act_lim = act_lim

    def call(self, inputs):
        params = self.net(inputs)
        pi_distribution = self.pi_dist_layer(params)
        return pi_distribution


class TRPOAgent(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_lim, mlp_hidden=64,
                 delta=0.01, vf_lr=1e-3, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
                 backtrack_coeff=0.8, train_v_iters=80, algo='npg'
                 ):

        super(TRPOAgent, self).__init__()
        self.policy_net = NormalActor(obs_dim=obs_dim, act_dim=act_dim,
                                      act_lim=act_lim, mlp_hidden=mlp_hidden)
        self.v_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)
        self.value_net = build_mlp(input_dim=obs_dim, output_dim=1, squeeze=True, mlp_hidden=mlp_hidden)
        self.value_net.compile(optimizer=self.v_optimizer, loss='mse')

        self.delta = delta
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_v_iters = train_v_iters
        self.algo = algo

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
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('BacktrackIters', average_only=True)

    def call(self, inputs, training=None, mask=None):
        pi_distribution = self.policy_net(inputs)
        pi_action = pi_distribution.sample()
        return pi_action

    def act_batch(self, obs):
        pi_distribution = self.policy_net(obs)
        pi_action = pi_distribution.sample()
        v = self.value_net(obs)
        return pi_action, v

    def _compute_kl(self, obs, old_pi):
        pi = self.policy_net(obs)
        kl_loss = tfp.distributions.kl_divergence(pi, old_pi)
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss

    def _compute_loss_pi(self, obs, act, logp, adv):
        distribution = self.policy_net(obs)
        log_prob = distribution.log_prob(act)
        negative_approx_kl = log_prob - logp
        ratio = tf.exp(negative_approx_kl)
        surr1 = ratio * adv
        policy_loss = -tf.reduce_mean(surr1, axis=0)
        return policy_loss

    def _compute_gradient(self, obs, act, logp, adv):
        # compute pi gradients
        with tf.GradientTape() as tape:
            policy_loss = self._compute_loss_pi(obs, act, logp, adv)
        grads = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        grads = flat_grads(grads)
        # flat grads
        return grads, policy_loss

    def _hessian_vector_product(self, obs, p):
        # compute Hx
        old_pi = self.policy_net(obs)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                kl = self._compute_kl(obs, old_pi)
            inner_grads = t1.gradient(kl, self.policy_net.trainable_variables)
            # flat gradients
            inner_grads = flat_grads(inner_grads)
            kl_v = tf.reduce_sum(inner_grads * p)
        grads = t2.gradient(kl_v, self.policy_net.trainable_variables)
        grads = flat_grads(grads)
        _Avp = grads + p * self.damping_coeff
        return _Avp

    @tf.function
    def _conjugate_gradients(self, obs, b, nsteps, residual_tol=1e-10):
        """

        Args:
            Avp: a callable computes matrix vector produce. Note that vector here has NO dummy dimension
            b: A^{-1}b
            nsteps: max number of steps
            residual_tol:

        Returns:

        """
        print(f'Tracing _conjugate_gradients b={b}, nsteps={nsteps}')
        x = tf.zeros_like(b)
        r = tf.identity(b)
        p = tf.identity(b)
        rdotr = tf.tensordot(r, r, axes=1)
        for _ in tf.range(nsteps):
            _Avp = self._hessian_vector_product(obs, p)
            # compute conjugate gradient
            alpha = rdotr / tf.tensordot(p, _Avp, axes=1)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = tf.tensordot(r, r, axes=1)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def _compute_natural_gradient(self, obs, act, logp, adv):
        print(f'Tracing _compute_natural_gradient with obs={obs}, act={act}, logp={logp}, adv={adv}')
        grads, policy_loss = self._compute_gradient(obs, act, logp, adv)
        x = self._conjugate_gradients(obs, grads, self.cg_iters)
        alpha = tf.sqrt(2. * self.delta / (tf.tensordot(x, self._hessian_vector_product(obs, x),
                                                        axes=1) + 1e-8))
        return alpha * x, policy_loss

    def _set_and_eval(self, obs, act, logp, adv, old_params, old_pi, natural_gradient, step):
        new_params = old_params - natural_gradient * step
        set_flat_params_to(self.policy_net, new_params)
        loss_pi = self._compute_loss_pi(obs, act, logp, adv)
        kl_loss = self._compute_kl(obs, old_pi)
        return kl_loss, loss_pi

    @tf.function
    def _update_actor(self, obs, act, adv):
        print(f'Tracing _update_actor with obs={obs}, act={act}, adv={adv}')
        old_params = get_flat_params_from(self.policy_net)
        old_pi = self.policy_net(obs)
        logp = old_pi.log_prob(act)
        natural_gradient, pi_l_old = self._compute_natural_gradient(obs, act, logp, adv)

        if self.algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = self._set_and_eval(obs, act, logp, adv, old_params, old_pi,
                                              natural_gradient, step=1.)
            j = tf.constant(value=0, dtype=tf.int32)
        elif self.algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            pi_l_new = tf.zeros(shape=(), dtype=tf.float32)
            kl = tf.zeros(shape=(), dtype=tf.float32)
            for j in tf.range(self.backtrack_iters):
                steps = tf.pow(self.backtrack_coeff, tf.cast(j, dtype=tf.float32))
                kl, pi_l_new = self._set_and_eval(obs, act, logp, adv, old_params, old_pi,
                                                  natural_gradient, step=steps)
                if kl <= self.delta and pi_l_new <= pi_l_old:
                    tf.print('Accepting new params at step', j, 'of line search.')
                    break

                if j == self.backtrack_iters - 1:
                    tf.print('Line search failed! Keeping old params.')
                    kl, pi_l_new = self._set_and_eval(obs, act, logp, adv, old_params, old_pi,
                                                      natural_gradient, step=0.)
        info = dict(
            LossPi=pi_l_old, KL=kl,
            DeltaLossPi=(pi_l_new - pi_l_old),
            BacktrackIters=j
        )
        return info

    def update(self, obs, act, ret, adv):
        assert tf.is_tensor(obs), f'obs must be a tf tensor. Got {obs}'
        info = self._update_actor(obs, act, adv)
        for key, item in info.items():
            info[key] = item.numpy()

        # train the value network
        v_l_old = self.value_net.evaluate(x=obs, y=ret, verbose=False)
        for i in range(self.train_v_iters):
            loss_v = self.value_net.train_on_batch(x=obs, y=ret)

        info['LossV'] = v_l_old
        info['DeltaLossV'] = loss_v - v_l_old

        # Log changes from update
        self.logger.store(**info)


def trpo(env_name, env_fn=None, mlp_hidden=128, seed=0,
         steps_per_epoch=5000, epochs=200, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo'):
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
    agent = TRPOAgent(obs_dim=obs_dim, act_dim=act_dim, act_lim=act_lim, mlp_hidden=mlp_hidden,
                      delta=delta, vf_lr=vf_lr, damping_coeff=damping_coeff, cg_iters=cg_iters,
                      backtrack_iters=backtrack_iters, backtrack_coeff=backtrack_coeff,
                      train_v_iters=train_v_iters, algo=algo)
    agent.set_logger(logger)

    buffer = GAEBuffer(obs_dim=obs_dim, act_dim=act_dim, num_envs=num_envs, length=max_ep_len,
                       gamma=gamma, lam=lam)

    def collect_trajectories():
        obs = env.reset()
        ep_ret = np.zeros(shape=num_envs, dtype=np.float32)
        ep_len = np.zeros(shape=num_envs, dtype=np.int32)
        for t in trange(max_ep_len, desc='Collecting'):
            act, val = agent.act_batch(tf.convert_to_tensor(obs, dtype=tf.float32))
            act = act.numpy()
            val = val.numpy()
            obs2, rew, dones, infos = env.step(act)
            buffer.store(obs, act, rew, val)
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
        agent.update(**buffer.get())
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

    logger_kwargs = setup_logger_kwargs(exp_name=args['env_name'] + '_trpo_test', data_dir='data', seed=args['seed'])
    trpo(**args, logger_kwargs=logger_kwargs)
