from abc import ABC, abstractmethod


class RewardFunction(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        pass
