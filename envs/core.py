from abc import ABC, abstractmethod
from typing import Dict, Any

import gym

class RewardFunction(ABC):

    @abstractmethod
    def __call__(self, state, action=None, next_state=None) -> float:
        pass

class RewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, reward_fn: RewardFunction):
        super().__init__(env)
        self._state = None
        self._reward_fn = reward_fn

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._state = state
        return state

    def step(self, action: Any):
        next_state, _, done, info = self.env.step(action)
        reward = self._reward_fn(state=self._state, action=action, next_state=next_state)
        self._state = next_state
        return next_state, reward, done, info



