from typing import List, Dict, Any

import numpy as np
import rtamt

from reward_shaping.core.reward import RewardFunction


class DefaultReward(RewardFunction):
    """
    this is a dummy reward for using the default reward of an environment,
    it assumes the default reward computed by the original implementation is passed in the info
    """
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'default_reward' in info
        return info['default_reward']


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
        # if `negate` is True, then indicator returns True when reward < 0.0
        reward = self.reward_fun(state, action, next_state, info)
        indicator = reward >= self.threshold if self.include_zero else reward > self.threshold
        result = indicator if not self.negate else not indicator
        result = float(result) if self.return_float else result
        return result


class MinAggregatorReward(RewardFunction):
    """
    given a list `fns` of individual score functions, it returns:
        - score=min{scores}
    """

    def __init__(self, fns: List[RewardFunction]):
        self._fns = fns

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        score = np.min([fn(state, action, next_state, info) for fn in self._fns])
        return score


class ProdAggregatorReward(RewardFunction):
    """
    given a list `fns` of individual score functions, it returns:
        - score=prod{scores}
    note: this is thought for binary indicator functions, in fact the product evaluates the AND of all indicators
    """

    def __init__(self, fns: List[RewardFunction]):
        self._fns = fns

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        score = np.prod([fn(state, action, next_state, info) for fn in self._fns])
        return score


class PotentialReward(RewardFunction):
    def __init__(self, reward_fn, potential_coeff=1.0):
        self._reward_fn = reward_fn
        self._potential_coeff = potential_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # assume: the reward fn depends only on the current state
        # reward = self._reward_fn(state=state, info=info)
        # next_reward = self._reward_fn(state=next_state, info=info)
        # not working
        # return self._potential_coeff * (next_reward - reward)
        return -1.0


def monitor_episode(stl_spec: str, vars: List[str], types: List[str], episode: Dict[str, Any]):
    spec = rtamt.STLSpecification()
    for v, t in zip(vars, types):
        spec.declare_var(v, f'{t}')
    spec.spec = stl_spec
    try:
        spec.parse()
    except rtamt.STLParseException:
        return
    # preprocess format, evaluate, post process
    robustness_trace = spec.evaluate(episode)
    return robustness_trace[0][1]