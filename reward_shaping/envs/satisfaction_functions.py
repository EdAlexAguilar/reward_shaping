from abc import ABC, abstractmethod
from collections import Callable

from reward_shaping.envs.core import RewardFunction


class IndicatorRewardFunction(RewardFunction):

    @abstractmethod
    def is_satisfied(self, state, action=None, next_state=None) -> bool:
        pass

class RewardThresholdIndicator(IndicatorRewardFunction):

    def __init__(self, reward_fn: RewardFunction, threshold: float, comparator: Callable[[float, float], bool] = lambda a, b: a > b):
        self._reward_fn = reward_fn
        self._threshold = threshold
        self._comparator = comparator

    def is_satisfied(self, state, action=None, next_state=None) -> bool:
        return self._comparator(self._reward_fn(state, action, next_state), self._threshold)

    def __call__(self, state, action=None, next_state=None) -> float:
        return self._reward_fn(state, action, next_state)



