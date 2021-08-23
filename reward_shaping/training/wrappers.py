
import gym
import numpy as np
from gym.spaces import Box, Dict


class FlattenFloatObservation(gym.Wrapper):
    # maybe not needed, see FlattenObservation
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self._obs_is_dict = isinstance(self.env.observation_space, Dict) 
        if not self._obs_is_dict:
            import warnings
            warnings.warning("observation space is not Dict, FlattenObservation wrapper does nothing")
        else:
            self._keys = [k for k, space in self.env.observation_space.spaces.items()]
            low = np.array([space.low for k, space in self.env.observation_space.spaces.items()])
            high = np.array([space.high for k, space in self.env.observation_space.spaces.items()])
            self.observation_space = Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = super(FlattenFloatObservation, self).step(action)
        if self._obs_is_dict:
            obs = np.array([[float(obs[k])] for k in self._keys], dtype=np.float32)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super(FlattenFloatObservation, self).reset()
        if self._obs_is_dict:
            obs = np.array([[float(obs[k])] for k in self._keys])
        return obs
