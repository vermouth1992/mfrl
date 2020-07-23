import numpy as np
import tensorflow as tf
import torch


class ModelBasedFn(object):
    env_name = 'Base'
    terminate = False
    reward = False

    def terminate_fn_numpy_batch(self, states, actions, next_states):
        return np.zeros(shape=(states.shape[0]), dtype=np.bool)

    def terminate_fn_torch_batch(self, states, actions, next_states):
        return torch.as_tensor(np.zeros(shape=(states.shape[0]), dtype=np.bool))

    def terminate_fn_tf_batch(self, states, actions, next_states):
        return tf.zeros(shape=(states.shape[0]), dtype=tf.bool)

    def terminate_fn_batch(self, states, actions, next_states):
        if isinstance(states, torch.Tensor):
            return self.terminate_fn_torch_batch(states, actions, next_states)
        elif isinstance(states, np.ndarray):
            return self.terminate_fn_numpy_batch(states, actions, next_states)
        elif isinstance(states, tf.Tensor):
            return self.terminate_fn_tf_batch(states, actions, next_states)
        else:
            raise ValueError('Unknown data type {}'.format(type(states)))

    def terminate_fn(self, state, action, next_state):
        if isinstance(state, torch.Tensor):
            states = torch.unsqueeze(state, dim=0)
            actions = torch.unsqueeze(action, dim=0)
            next_states = torch.unsqueeze(next_state, dim=0)
            return self.terminate_fn_torch_batch(states, actions, next_states)[0]
        elif isinstance(state, np.ndarray):
            states = np.expand_dims(state, axis=0)
            actions = np.expand_dims(action, axis=0)
            next_states = np.expand_dims(next_state, axis=0)
            return self.terminate_fn_numpy_batch(states, actions, next_states)[0]
        elif isinstance(state, tf.Tensor):
            states = tf.expand_dims(state, axis=0)
            actions = tf.expand_dims(action, axis=0)
            next_states = tf.expand_dims(next_state, axis=0)
            return self.terminate_fn_tf_batch(states, actions, next_states)[0]
        else:
            raise ValueError('Unknown data type {}'.format(type(state)))

    def cost_fn_numpy_batch(self, states, actions, next_states):
        raise NotImplementedError

    def cost_fn_torch_batch(self, states, actions, next_states):
        raise NotImplementedError

    def cost_fn_tf_batch(self, states, actions, next_states):
        raise NotImplementedError

    def compute_reward(self, state, action, next_state):
        if isinstance(state, torch.Tensor):
            states = torch.unsqueeze(state, dim=0)
            actions = torch.unsqueeze(action, dim=0)
            next_states = torch.unsqueeze(next_state, dim=0)
            return -self.cost_fn_torch_batch(states, actions, next_states)[0]
        elif isinstance(state, np.ndarray):
            states = np.expand_dims(state, axis=0)
            actions = np.expand_dims(action, axis=0)
            next_states = np.expand_dims(next_state, axis=0)
            return -self.cost_fn_numpy_batch(states, actions, next_states)[0]
        elif isinstance(state, tf.Tensor):
            states = tf.expand_dims(state, axis=0)
            actions = tf.expand_dims(action, axis=0)
            next_states = tf.expand_dims(next_state, axis=0)
            return -self.cost_fn_tf_batch(states, actions, next_states)[0]
        else:
            raise ValueError('Unknown data type {}'.format(type(state)))

    def compute_reward_batch(self, states, actions, next_states):
        if isinstance(states, torch.Tensor):
            return -self.cost_fn_torch_batch(states, actions, next_states)
        elif isinstance(states, np.ndarray):
            return -self.cost_fn_numpy_batch(states, actions, next_states)
        elif isinstance(states, tf.Tensor):
            return -self.cost_fn_tf_batch(states, actions, next_states)
        else:
            raise ValueError('Unknown data type {}'.format(type(states)))
