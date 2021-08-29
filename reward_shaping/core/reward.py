from abc import ABC, abstractmethod
from typing import List


class RewardFunction(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        pass


class WeightedReward(RewardFunction):
    """
    reward(s,a) := w_s * sum([score in safeties]) + w_t * sum([score in targets]) + w_c * sum([score in comforts])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._safety_weight = None
        self._target_weight = None
        self._comfort_weight = None
        self._safety_rules = None
        self._target_rules = None
        self._comfort_rules = None

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert all([type(w) == float for w in [self._safety_weight, self._target_weight, self._comfort_weight]])
        assert all([type(l) == list for l in [self._safety_rules, self._target_rules, self._comfort_rules]])
        safety_score = sum([fn(state, action, next_state, info) for fn in self._safety_rules])
        target_score = sum([fn(state, action, next_state, info) for fn in self._target_rules])
        comfort_score = sum([fn(state, action, next_state, info) for fn in self._comfort_rules])
        return self._safety_weight * safety_score + \
               self._target_weight * target_score + \
               self._comfort_weight * comfort_score
