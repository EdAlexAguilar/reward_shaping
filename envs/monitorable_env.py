from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np

import gym


class MonitorableEnv(gym.Env, ABC):

    def __init__(self):
        self._episode = {var: [] for var in self.monitoring_variables}
        self._last_complete_episode = None
        self._last_train_spec = None
        self._last_eval_spec = None

    @property
    @abstractmethod
    def monitoring_variables(self) -> List[str]:
        """List of monitored variables (ie, the variables which occur in the stl spec)."""
        pass

    @property
    @abstractmethod
    def monitoring_types(self) -> List[str]:
        """List of variables types."""
        pass

    @property
    @abstractmethod
    def train_spec(self) -> str:
        """
        stl spec used for TRAINING of agents. e.g., it could use continuous signals, like distances to obst/target..
        """
        pass

    @property
    @abstractmethod
    def eval_spec(self) -> str:
        """
        stl spec used as reward for evaluation env, it aims to be an EVALUATION METRIC among different reward shapings.
        For example, we could be interested in remarking the valuation of safety properties and distinguish the episodes
        with safety violations from the others. This is not necessarly the aim of the `train_spec` which could instead
        use a continuous definition.

        Example: in the cartpole, for representing the learning curves, we use boolean signal for
        safety properties (e.g. +3 hold, -3 violated). In this way, we can clearly recognize the episodes where any
        safety violation occurs because the robustness value has a clear negative value.
        """
        pass

    @property
    def last_complete_episode(self) -> Dict[str, List]:
        """
        Entire last episode, as a dictionary mapping each monitoring variable to the signal
        """
        return self._last_complete_episode

    @property
    def last_train_spec(self) -> str:
        return self._last_train_spec

    @property
    def last_eval_spec(self) -> str:
        return self._last_eval_spec

    def compute_episode_robustness(self, episode: Dict[str, np.ndarray], monitoring_spec: str):
        import rtamt
        spec = rtamt.STLSpecification()
        for v, t in zip(self.monitoring_variables, self.monitoring_types):
            spec.declare_var(v, f'{t}')
        spec.spec = monitoring_spec
        try:
            spec.parse()
        except rtamt.STLParseException:
            return
        # preprocess format, evaluate, post process
        robustness_trace = spec.evaluate(episode)
        return robustness_trace[0][1]

    def expand_episode(self, monitored_state: Dict[str, Any], done: bool):
        for key, value in monitored_state.items():
            self._episode[key].append(value)
        if done:
            self._last_complete_episode = self._episode
            self._last_train_spec = self.train_spec
            self._last_eval_spec = self.eval_spec
            self._episode = {v: [] for v in self.monitoring_variables}

    @abstractmethod
    def get_monitored_state(self, obs, done, info) -> Dict[str, Any]:
        pass

    @abstractmethod
    def step(self, action):
        obs, reward, done, info = ...
        monitored_state = self.get_monitored_state(obs, done, info)
        self.expand_episode(monitored_state, done)
        return obs, reward, done, info
