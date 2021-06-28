from typing import Dict, List

import gym
import numpy as np


class HierarchicalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, hierarchy: Dict[str, List], clip_negative_rewards: bool = False,
                 shift_rewards: bool = False, unit_scaling: bool = False):
        assert 'safety' in hierarchy
        assert 'target' in hierarchy
        assert 'comfort' in hierarchy
        super().__init__(env)
        self._env = env
        self._hierarchy = hierarchy
        # bool parameters
        self._clip_negative = clip_negative_rewards  # `true` if we want to clip rewards to [0,+inf]
        self._shift_by_one = shift_rewards  # `true` if we want to shift each hierarchy by one
        self._unit_scaling = unit_scaling  # `true` if we want to scale the final reward in [0, 1]
        # reward for rendering
        self._reward = 0.0
        self._return = 0.0

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        self._reward = 0.0
        self._return = 0.0
        return obs

    def reward(self, rew):
        self._reward = self.reward_in_state(self.state)
        self._return += self._reward
        self._env._reward = self._reward
        self._env._return = self._return
        return self._env._reward

    def reward_in_state(self, state):
        if state is None:
            raise ValueError("eval reward in not initialized env, state is None")
        # compute rewards for each hierarchy level: each fun returns (value, bool)
        safety_results = np.array([f(state) for f in self._hierarchy['safety']])
        target_results = np.array([f(state) for f in self._hierarchy['target']])
        comfort_results = np.array([f(state) for f in self._hierarchy['comfort']])
        # compute indicators: ind_k = AND_{r in H_k} (f>=0)
        safety_ind = all(safety_results[:, 1]) if len(safety_results)>0 else False
        target_ind = all(target_results[:, 1]) if len(target_results)>0 else False
        safety_rewards = safety_results[:, 0] if len(safety_results)>0 else [0.0]
        target_rewards = target_results[:, 0] if len(target_results)>0 else [0.0]
        comfort_rewards = comfort_results[:, 0] if len(comfort_results)>0 else [0.0]
        # (optional) clip rewards
        if self._clip_negative:
            safety_rewards = np.clip(safety_rewards, 0.0, np.Inf)
            target_rewards = np.clip(target_rewards, 0.0, np.Inf)
            comfort_rewards = np.clip(comfort_rewards, 0.0, np.Inf)
        if self._shift_by_one:
            safety_rewards = 1 + safety_rewards
            target_rewards = 1 + target_rewards
            comfort_rewards = 1 + comfort_rewards
        # compute individual rewards
        tot_safety_reward = np.mean(safety_rewards) if len(safety_rewards) > 0 else 0.0
        tot_target_reward = np.mean(target_rewards) if len(target_rewards) > 0 and safety_ind else 0.0
        tot_comfort_reward = np.mean(comfort_rewards) if len(comfort_rewards) > 0 and safety_ind and target_ind else 0.0
        # compute final indicator-based reward
        if self._unit_scaling:
            final_reward = 1 / 3 * tot_safety_reward + 1 / 3 * tot_target_reward + 1 / 3 * tot_comfort_reward
        else:
            final_reward = tot_safety_reward + tot_target_reward + tot_comfort_reward
        return final_reward
