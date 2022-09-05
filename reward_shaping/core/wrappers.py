from collections import deque
from typing import Any, Callable, List

import gym
import numpy as np

from reward_shaping.core.configs import TLRewardConfig, EvalConfig
from reward_shaping.core.reward import RewardFunction


class RewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, reward_fn: RewardFunction):
        super().__init__(env)
        self._state = None
        self._reward = 0.0
        self._return = 0.0
        self._reward_fn = reward_fn

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._state = state
        self._reward = 0.0
        self._return = 0.0
        return state

    def step(self, action: Any):
        next_state, _, done, info = self.env.step(action)
        reward = self._reward_fn(state=self._state, action=action, next_state=next_state, info=info)
        self._state = next_state
        self._reward = reward
        self._return += reward
        return next_state, reward, done, info


class CollectionWrapper(gym.Wrapper):
    """
    This wrapper collects the measurable variables obtainable from the env state through the `extractor_fn`
    and store them in a dictionary under the names `variables`.
    It stores the `window_len` most recent variables using a moving-window.

    @param: env: gym environment
    @param: variables: name of the collected variables
    @param: extractor_fn: callable function to extract the variables from the env state
    @param: window_len: size of the moving window, if None then stores the whole episode till termination
    """

    def __init__(self, env: gym.Env, variables: List[str], extractor_fn: Callable, window_len: int = None):
        super(CollectionWrapper, self).__init__(env)
        self.env = env
        self._variables = variables
        self._extractor_fn = extractor_fn
        self._window_len = window_len
        self._episode = {var: deque(maxlen=self._window_len) for var in self._variables}

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._episode = {var: deque(maxlen=self._window_len) for var in self._variables}
        return state

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # prepare monitoring variables, collect them
        monitored_state = self._extractor_fn(obs, done, info)
        for key, value in monitored_state.items():
            self._episode[key].append(value)
        # evaluate reward only in terminal states
        return obs, reward, done, info


class TLRewardWrapper(CollectionWrapper):
    """
    This is an 'episodic' wrapper which evaluate a spec using RTAMT monitor.
    It evaluates the episode (or a slice of it), and return the robustness of the specification.

    @param: eval_at_end: boolean indicating if evaluating only on terminal states or at each step
    """

    def __init__(self, env: gym.Env, tl_conf: TLRewardConfig, semantics: str = "stl", window_len: int = None,
                 eval_at_end=True):
        super(TLRewardWrapper, self).__init__(env, tl_conf.monitoring_variables, tl_conf.get_monitored_state,
                                              window_len)
        self._tl_conf = tl_conf  # tl-spec configuration
        self._eval_at_end = eval_at_end
        self._reward = 0.0
        self._return = 0.0
        # initialize monitor fn
        if semantics == "stl":
            from reward_shaping.core.helper_fns import monitor_stl_episode
            self._monitor = monitor_stl_episode
        elif semantics == "filtering":
            from reward_shaping.core.helper_fns import monitor_mtl_filtering_episode
            self._monitor = monitor_mtl_filtering_episode
        else:
            raise NotImplementedError(f"semantics {semantics} not implemented. available semantics: 'stl', 'filtering'")

    def reset(self, **kwargs):
        self._reward = 0.0
        self._return = 0.0
        state = super().reset(**kwargs)
        return state

    def _compute_episode_robustness(self, done):
        reward = 0.0
        if len(self._episode['time']) > 1 and (not self._eval_at_end or done):
            reward = self._monitor(self._tl_conf.spec, self._tl_conf.monitoring_variables,
                                   self._tl_conf.monitoring_types, self._episode)[0][1]
        return reward

    def get_monitored_episode(self):
        return self._episode

    def step(self, action):
        obs, _, done, info = super().step(action)
        reward = self._compute_episode_robustness(done)
        self._reward = reward
        self._return += reward
        return obs, reward, done, info


class EvaluationRewardWrapper(CollectionWrapper):
    """ This is an 'episodic' wrapper which evaluate a custom metric in the terminal states."""

    def __init__(self, env: gym.Env, conf: EvalConfig):
        super(EvaluationRewardWrapper, self).__init__(env, conf.monitoring_variables, conf.get_monitored_state)
        self._conf = conf
        self._reward = 0.0
        self._return = 0.0

    def reset(self, **kwargs):
        self._reward = 0.0
        self._return = 0.0
        state = super().reset(**kwargs)
        return state

    def step(self, action):
        obs, _, done, info = super().step(action)
        reward = self._conf.eval_episode(self._episode) if done else 0.0
        self._reward = reward
        self._return += reward
        return obs, reward, done, info