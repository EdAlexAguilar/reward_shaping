import random
from typing import Callable, List

import gym

from reward_shaping.envs.f1tenth.core.single_agent_env import SingleAgentRaceEnv


class ChangingTrackRaceEnv(gym.Env):

    def __init__(self, map_names: List[str], order: str, **kwargs):
        super().__init__()
        self._envs = [SingleAgentRaceEnv(map_name=map_name, **kwargs) for map_name in map_names]
        self._current_track_index = 0
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(self._envs)
        elif order == 'random':
            self._order_fn = lambda: random.choice(list(set(range(0, len(self._envs))) - {self._current_track_index}))
        self._order = order

        assert all(self._envs[0].action_space == env.action_space for env in self._envs)
        assert all(self._envs[0].observation_space == env.observation_space for env in self._envs)
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space

    def step(self, action):
        return self._get_env().step(action=action)

    def reset(self, mode, **kwargs):
        self._current_track_index = self._order_fn()
        if self._get_env().renderer is not None:
            track = self._get_env()._track
            self._get_env().renderer.update_map(track.filepath, track.ext)
        return self._get_env().reset(mode, **kwargs)

    def render(self, **kwargs):
        return self._get_env().render(**kwargs)

    def close(self):
        for env in self._envs:
            env.close()

    def _get_env(self):
        return self._envs[self._current_track_index]
