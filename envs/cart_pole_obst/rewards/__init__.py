import gym

from envs.cart_pole_obst.rewards.baselines import ContinuousReward, SparseReward, STLReward
from envs.cart_pole_obst.rewards.graph_based import GraphWithContinuousScore, PotentialGraphWithContinuousScore, \
    GraphWithContinuousTargetAndDiscreteSafety, PotentialGraphWithContinuousTargetAndDiscreteSafety

_registry = {}


def get_reward(name: str) -> gym.RewardWrapper:
    return _registry[name]


def register_reward(name: str, reward: gym.RewardWrapper):
    if name not in _registry.keys():
        _registry[name] = reward


register_reward('sparse', reward=SparseReward)
register_reward('continuous', reward=ContinuousReward)
register_reward('stl', reward=STLReward)
register_reward('cont_gh', reward=GraphWithContinuousScore)
register_reward('cont_gh_pot', reward=PotentialGraphWithContinuousScore)
register_reward('sdisc_gh', reward=GraphWithContinuousTargetAndDiscreteSafety)
register_reward('sdisc_gh_pot', reward=PotentialGraphWithContinuousTargetAndDiscreteSafety)