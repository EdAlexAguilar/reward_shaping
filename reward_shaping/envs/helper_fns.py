import numpy as np

from reward_shaping.envs.core import RewardFunction


class NormalizedReward(RewardFunction):
    def __init__(self, reward_fn, min_reward, max_reward):
        assert max_reward > min_reward, f"unvalid normalization: min: {min_reward} >= max: {max_reward}"
        super().__init__()
        self.reward_fn = reward_fn
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        reward = self.reward_fn(state, action, next_state, info)
        reward = np.clip(reward, self.min_reward, self.max_reward)
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)


class ThresholdIndicator(RewardFunction):
    def __init__(self, reward_fun: RewardFunction, threshold=0.0, negate=False, include_zero=True, return_float=True):
        self.reward_fun = reward_fun
        self.threshold = threshold
        self.negate = negate
        self.include_zero = include_zero
        self.return_float = return_float

    def __call__(self, state, action, next_state=None, info=None):
        # (default) if `reverse` is False, then indicator returns True when reward >= 0.0
        # if `reverse` is True, then indicator returns True when reward < 0.0
        reward = self.reward_fun(state, action, next_state, info)
        indicator = reward >= self.threshold if self.include_zero else reward > self.threshold
        result = indicator if not self.negate else not indicator
        result = float(result) if self.return_float else result
        return result


class PotentialReward(RewardFunction):
    def __init__(self, reward_fn, potential_coeff=1.0):
        self._reward_fn = reward_fn
        self._potential_coeff = potential_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        reward = self._reward_fn(state=state, info=info)
        next_reward = self._reward_fn(state=next_state, info=info)
        return self._potential_coeff * (next_reward - reward)