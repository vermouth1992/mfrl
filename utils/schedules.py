""" This file is used for specifying various schedules that evolve over
time throughout the execution of the algorithm, such as:
 - learning rate for the optimizer
 - exploration epsilon for the epsilon greedy exploration strategy
 - beta parameter for beta parameter in prioritized replay
Each schedule has a function `value(t)` which returns the current value
of the parameter given the timestep t of the optimization procedure.

Some codes are directly copied from https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
"""

import numpy as np
import yaml


class Schedule(yaml.YAMLObject):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ExponentialScheduler(Schedule):
    def __init__(self, epsilon=1.0, decay=1e-4, minimum=0.01):
        self.epsilon = epsilon
        self.decay = decay
        self.minimum = minimum

    def value(self, t):
        explore_p = self.minimum + (self.epsilon - self.minimum) * np.exp(-self.decay * t)
        return explore_p


class ConstantSchedule(Schedule):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(Schedule):
    yaml_tag = u'!PiecewiseSchedule'
    yaml_loader = yaml.SafeLoader

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
        if outside_value is not None:
            self._outside_value = outside_value
        else:
            self._outside_value = endpoints[-1][1]  # by default, using the last value.
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

    @classmethod
    def from_yaml(cls, loader, node):
        dict_representation = loader.construct_mapping(node, deep=True)
        endpoints = dict_representation['endpoints']
        return cls(endpoints=endpoints)

    @classmethod
    def to_yaml(cls, dumper, data):
        dict_representation = {
            'endpoints': data._endpoints
        }
        node = dumper.represent_mapping(cls.yaml_tag, dict_representation)
        return node


class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)