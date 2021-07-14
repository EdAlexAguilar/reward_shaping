from typing import Dict, List, Tuple

import numpy as np
import gym


class HierarchicalRewardWrapper(gym.RewardWrapper):
    """
    Hierarchical Reward Definition, using indicator functions.

    Assume to receive a hierarchy with reward functions grouped into `safety`, `target`, `comfort`.
    The reward is then defined as:
        r(s,a) := safety_score + safety_indicator * target_score + safety_indicator * target_indicator * comfort_score
    where:
        safety_score is an aggregated metric which evaluates safety requirements,
        safety_indicator is a binary indicator which is true when all the safety requirements are satisfied,
        similar for target and comfort.
    """

    def __init__(self, env, hierarchy: Dict[str, List[Tuple]], clip_to_positive: bool = False,
                 unit_scaling: bool = False):
        """
        Note: hierarchy is defined as a dictionary of lists (i.e. str -> list).
        Each element of a list is a tuple (function, indicator), where function returns a score, indicator returns bool.

        The reason of this choice is that function could return a normalized score,
        loosing the information on boolean satisfaction of the requirements because no more in the original range.
        """
        assert 'safety' in hierarchy
        assert 'target' in hierarchy
        assert 'comfort' in hierarchy
        super().__init__(env)
        self.hierarchy = hierarchy
        # bool parameters
        self.clip_to_positive = clip_to_positive  # `true` if we want to clip rewards to [0,+inf]
        self.unit_scaling = unit_scaling  # `true` if we want to scale the final reward in [0, 1]
        # aux variables, for rendering
        self.rew = 0.0
        self.ret = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.rew = 0.0
        self.ret = 0.0
        return obs

    def reward(self, rew):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        if state is None:
            raise ValueError("eval reward in not initialized env, state is None")
        # compute rewards for each hierarchy level: each fun returns (value, bool)
        safety_results = np.array([[f(state), ind(state)] for f, ind in self.hierarchy['safety']])
        target_results = np.array([[f(state), ind(state)] for f, ind in self.hierarchy['target']])
        comfort_results = np.array([[f(state), ind(state)] for f, ind in self.hierarchy['comfort']])
        # compute indicators: ind_k = AND_{r in H_k} (f>=0)
        safety_ind = all(safety_results[:, 1]) if len(safety_results) > 0 else False
        target_ind = all(target_results[:, 1]) if len(target_results) > 0 else False
        safety_rewards = safety_results[:, 0] if len(safety_results) > 0 else [0.0]
        target_rewards = target_results[:, 0] if len(target_results) > 0 else [0.0]
        comfort_rewards = comfort_results[:, 0] if len(comfort_results) > 0 else [0.0]
        # (optional) clip rewards
        if self.clip_to_positive:
            safety_rewards = np.clip(safety_rewards, 0.0, np.Inf)
            target_rewards = np.clip(target_rewards, 0.0, np.Inf)
            comfort_rewards = np.clip(comfort_rewards, 0.0, np.Inf)
        # compute individual rewards
        tot_safety_reward = np.mean(safety_rewards) if len(safety_rewards) > 0 else 0.0
        tot_target_reward = np.mean(target_rewards) if len(target_rewards) > 0 and safety_ind else 0.0
        tot_comfort_reward = np.mean(comfort_rewards) if len(comfort_rewards) > 0 and safety_ind and target_ind else 0.0
        # compute final indicator-based reward
        if self.unit_scaling:
            final_reward = 1 / 3 * tot_safety_reward + 1 / 3 * tot_target_reward + 1 / 3 * tot_comfort_reward
        else:
            final_reward = tot_safety_reward + tot_target_reward + tot_comfort_reward
        # this is a workaround to visualize the scores in rendering
        self.env.safety_tot = tot_safety_reward
        self.env.target_tot = tot_target_reward
        self.env.comfort_tot = tot_comfort_reward

        return final_reward
