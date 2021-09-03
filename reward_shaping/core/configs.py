from abc import ABC, abstractmethod
from typing import List

from reward_shaping.core.graph_based import GraphBasedReward
from reward_shaping.core.reward import RewardFunction


class RewardConfig(ABC):
    def __init__(self, env_params):
        self._env_params = env_params


class STLRewardConfig(RewardConfig):
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


class GraphRewardConfig(RewardConfig):
    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def topology(self):
        pass


class BuildGraphReward:
    @staticmethod
    def from_conf(graph_config: GraphRewardConfig) -> RewardFunction:
        reward_fn = GraphBasedReward.from_collections(nodes=graph_config.nodes, topology=graph_config.topology)
        return reward_fn
