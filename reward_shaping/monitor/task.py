from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import gym
import numpy as np

from reward_shaping.monitor.formula import Operator
from reward_shaping.monitor.monitor import Monitor


class RLTask(gym.Wrapper):
    def __init__(self, env: gym.Env, requirements: List[Tuple[str, Operator, Callable]]):
        super(RLTask, self).__init__(env)
        self._requirements = requirements
        self._monitors = {}
        assert len(requirements) == len(
            set([l for l, _, _ in requirements])), f"not unique labels {[l for l, _, _ in requirements]}"
        for i, (label, op, pred) in enumerate(requirements):
            self._monitors[i] = Monitor.from_spec(op, pred)

    def _get_monitor_infos(self, obs, info):
        infos = {}
        for i, monitor in self._monitors.items():
            mstate, mcounter = monitor.step(obs, info)
            infos[f"{self._requirements[i][0]}_state"] = mstate
            infos[f"{self._requirements[i][0]}_counter"] = mcounter
        return infos

    def reset(self, **kwargs):
        self._time = 0
        for i, monitor in self._monitors.items():
            monitor.reset()
        return super(RLTask, self).reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super(RLTask, self).step(action)
        monitor_infos = self._get_monitor_infos(obs, info)
        info.update(monitor_infos)  #inplace
        return obs, reward, done, info
