from abc import ABC, abstractmethod
from typing import List


class RewardConfig(ABC):
    def __init__(self, env_params):
        self._env_params = env_params


class TLRewardConfig(RewardConfig):
    @property
    @abstractmethod
    def monitoring_variables(self) -> List[str]:
        """List of monitored variables (ie, the variables which occur in the stl spec)."""
        pass

    @property
    @abstractmethod
    def monitoring_types(self) -> List[str]:
        """List of variables types."""
        pass

    @property
    @abstractmethod
    def spec(self) -> str:
        """
        stl specification used for evaluation
        """
        pass

    @abstractmethod
    def get_monitored_state(self, state, done, info):
        """Given observed quantities (eg, state,info..), prepare the variable monitored in the stl spec."""
        pass


class EvalConfig(RewardConfig):
    @property
    @abstractmethod
    def monitoring_variables(self) -> List[str]:
        """List of monitored variables (ie, the variables which occur in the stl spec)."""
        pass

    @property
    @abstractmethod
    def monitoring_types(self) -> List[str]:
        """List of variables types."""
        pass

    @abstractmethod
    def get_monitored_state(self, state, done, info):
        """Given observed quantities (eg, state,info..), prepare the variable monitored in the stl spec."""
        pass

    @abstractmethod
    def eval_episode(self, episode) -> float:
        """custom method to eval episode """
        pass
