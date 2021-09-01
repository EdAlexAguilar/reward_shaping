from typing import Any
import gym

from reward_shaping.core.configs import STLRewardConfig
from reward_shaping.core.helper_fns import monitor_episode
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

    def render(self, mode='human', **kwargs):
        return super(RewardWrapper, self).render(mode=mode, info={'reward': self._reward, 'return': self._return})


class STLRewardWrapper(gym.Wrapper):
    """ This is an 'episodic' wrapper which evaluate a spec in the terminal states."""

    def __init__(self, env: gym.Env, stl_conf: STLRewardConfig):
        super(STLRewardWrapper, self).__init__(env)
        self._episode = {var: [] for var in stl_conf.monitoring_variables}
        self._stl_conf = stl_conf
        self._reward = 0.0
        self._return = 0.0

    def reset(self, **kwargs):
        self._episode = {var: [] for var in self._stl_conf.monitoring_variables}
        self._reward = 0.0
        self._return = 0.0
        state = self.env.reset(**kwargs)
        return state

    def _compute_episode_robustness(self):
        return monitor_episode(self._stl_conf.spec, self._stl_conf.monitoring_variables,
                               self._stl_conf.monitoring_types, self._episode)

    def get_monitored_episode(self):
        return self._episode

    def step(self, action):
        obs, _, done, info = super().step(action)
        # prepare monitoring variables, collect them
        monitored_state = self._stl_conf.get_monitored_state(obs, done, info)
        for key, value in monitored_state.items():
            self._episode[key].append(value)
        # evaluate reward only in terminal states
        reward = self._compute_episode_robustness() if done else 0.0
        self._reward = reward
        self._return += reward
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        return super(STLRewardWrapper, self).render(mode=mode,
                                                    info={'reward': self._reward, 'return': self._return})
