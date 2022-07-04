import collections
from typing import Dict, Tuple

import gym
from gym.wrappers import LazyFrames
import numpy as np


class FixResetWrapper(gym.Wrapper):
    """Fix a reset mode to sample initial condition from starting grid or randomly over the track."""

    def __init__(self, env, mode):
        assert mode in ['grid', 'random', 'random_bidirectional']
        self._mode = mode
        super(FixResetWrapper, self).__init__(env)

    def reset(self):
        return super(FixResetWrapper, self).reset(mode=self._mode)


class FilterObservationWrapper(gym.Wrapper):
    """
    observation wrapper that filter a single observation and return is without dictionary,
    all the observable quantities are moved to the info as `state`
    """

    def __init__(self, env, obs_list=[]):
        super(FilterObservationWrapper, self).__init__(env)
        self._obs_list = obs_list
        self.observation_space = gym.spaces.Dict({obs: self.env.observation_space[obs] for obs in obs_list})

    def _filter_obs(self, original_obs):
        new_obs = {}
        for obs in self._obs_list:
            assert obs in original_obs
            new_obs[obs] = original_obs[obs]
        return new_obs

    def step(self, action):
        original_obs, reward, done, info = super().step(action)
        new_obs = self._filter_obs(original_obs)
        # add original state into the info
        new_info = info
        new_info['state'] = {name: value for name, value in original_obs.items()}
        return new_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        new_obs = self._filter_obs(obs)
        return new_obs


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


class FrameStackOnChannel(gym.Wrapper):
    r"""
    Observation wrapper that stacks the observations in a rolling manner.

    Implementation from gym.wrappers but squeeze observation (then removing channel dimension),
    and support dictionary
    """

    def __init__(self, env, num_stack):
        super(FrameStackOnChannel, self).__init__(env)
        self.num_stack = num_stack

        self.frames = {
            k: collections.deque(maxlen=num_stack) for k in self.observation_space.spaces
        }
        lows = {
            k: np.repeat(space.low, num_stack, axis=0) for k, space in
            self.observation_space.spaces.items()
        }
        highs = {
            k: np.repeat(space.high, num_stack, axis=0) for k, space in
            self.observation_space.spaces.items()
        }

        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.Box(lows[k], highs[k], dtype=self.observation_space.spaces[k].dtype)
                for k in self.observation_space.spaces
            }
        )

    def _get_observation(self):
        assert all([len(frames) == self.num_stack for k, frames in self.frames.items()]), self.frames
        # return {k: LazyFrames(list(self.frames[k]), self.lz4_compress) for k in self.frames}
        return {k: np.array(self.frames[k], dtype=np.float32) for k in self.frames}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        for k in self.observation_space.spaces:
            self.frames[k].append(np.squeeze(observation[k]))  # assume 1d channel dimension and remove it
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for k in observation:
            [self.frames[k].append(np.squeeze(observation[k])) for _ in range(self.num_stack)]
        return self._get_observation()


class FixSpeedControl(gym.ActionWrapper):
    """
    reduce the original action space, and fix the speed command to a constant value
    """

    def __init__(self, env, fixed_speed: float = 2.0):
        super(FixSpeedControl, self).__init__(env)
        assert type(fixed_speed) == float
        assert "speed" in self.action_space.spaces.keys(), "the action space does not have any 'speed' action"
        self._fixed_speed = np.array([fixed_speed])
        assert self.action_space["speed"].contains(
            self._fixed_speed), "the fixed speed is not in the original action space"
        self.action_space = gym.spaces.Dict(
            {a: space for a, space in self.action_space.spaces.items() if a != "speed"})

    def action(self, action):
        action["speed"] = self._fixed_speed
        return action


class NormalizeObservationWithMinMax(gym.ObservationWrapper):
    """
    Normalize Box observations with respect to given min and max values.
    The observations to be normalized are given as a dictionary: obs_name -> (minvalue, maxvalue)

    The values are clipped w.r.t. min/max values to avoid unexpected values (e.g., <-1 or >1)
    """

    def __init__(self, env, map_name2values: Dict[str, Tuple[float, float]]):
        super(NormalizeObservationWithMinMax, self).__init__(env)
        self._map_name2values = map_name2values
        for obs_name in self._map_name2values.keys():
            self.observation_space[obs_name] = gym.spaces.Box(low=-1, high=+1,
                                                              shape=self.observation_space[obs_name].shape)

    def observation(self, observation):
        for obs_name, (min_value, max_value) in self._map_name2values.items():
            new_obs = np.clip(observation[obs_name], min_value, max_value)  # clip it
            new_obs = (new_obs - min_value) / (max_value - min_value)  # norm in 0,1
            observation[obs_name] = -1 + 2 * new_obs  # finally, map it to -1..+1
        return observation
