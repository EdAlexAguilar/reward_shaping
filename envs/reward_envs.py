from typing import Dict, List

import gym
import numpy as np


class HierarchicalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, hierarchy: Dict[str, List]):
        assert 'safety' in hierarchy
        assert 'target' in hierarchy
        assert 'comfort' in hierarchy
        super().__init__(env)
        self._env = env
        self._hierarchy = hierarchy

    def reward(self, rew):
        state = self.state
        # compute rewards for each hierarchy level
        safety_rewards = np.array([f(state) for f in self._hierarchy['safety']])
        target_rewards = np.array([f(state) for f in self._hierarchy['target']])
        comfort_rewards = np.array([f(state) for f in self._hierarchy['comfort']])
        # compute indicators: ind_k = AND_{r in H_k} (f>=0)
        safety_indicator = all(safety_rewards >= 0.0)
        target_indicator = all(target_rewards >= 0.0)
        # compute final indicator-based reward
        final_reward = np.sum(safety_rewards) + \
                       safety_indicator * np.sum(target_rewards) + \
                       target_indicator * np.sum(comfort_rewards)
        return final_reward
