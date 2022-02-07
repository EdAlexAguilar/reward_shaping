import random
from typing import Callable, List

import gym

from reward_shaping.envs.f1tenth.core.single_agent_env import SingleAgentRaceEnv


class ChangingTrackRaceEnv(gym.Env):

    def __init__(self, map_names: List[str], order: str, every: int, **kwargs):
        super().__init__()
        self._map_names = map_names
        self._params = kwargs
        self._current_track_index = 0
        self._order = order  # order for changing track
        self._every = every  # nr episodes every which changing track
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(map_names)
        elif order == 'random':
            self._order_fn = lambda: random.choice(list(set(range(0, len(map_names))) - {self._current_track_index}))
        # first time create environments and check consistency obs spaces
        envs = [SingleAgentRaceEnv(map_name=map_name, **kwargs) for map_name in map_names]
        assert all(envs[0].action_space == env.action_space for env in envs)
        assert all(envs[0].observation_space == env.observation_space for env in envs)
        # init first env
        self._nr_resets = 0
        self._current_env = envs[0]
        self.action_space = self._current_env.action_space
        self.observation_space = self._current_env.observation_space

    def step(self, action):
        return self._current_env.step(action=action)

    def reset(self, **kwargs):
        self._nr_resets += 1
        if self._nr_resets % self._every == 0:
            self._current_track_index = self._order_fn()
            self._current_env.close()
            print(self._map_names[self._current_track_index])
            self._current_env = SingleAgentRaceEnv(map_name=self._map_names[self._current_track_index], **self._params)
        return self._current_env.reset(**kwargs)

    def render(self, **kwargs):
        return self._current_env.render(**kwargs)

    def close(self):
        self._current_env.close()
