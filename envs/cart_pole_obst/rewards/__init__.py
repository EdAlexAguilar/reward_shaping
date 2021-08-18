import gym

from envs.cart_pole_obst.rewards.baselines import ContinuousReward, SparseReward, STLReward, BoolSTLReward
from envs.cart_pole_obst.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator, \
    GraphWithContinuousScoreContinuousIndicator, PotentialGraphWithContinuousScore, \
    GraphWithContinuousTargetAndDiscreteSafety, PotentialGraphWithContinuousTargetAndDiscreteSafety, GraphWithTwoTargets

_registry = {}


def get_reward(name: str) -> gym.RewardWrapper:
    return _registry[name]


def register_reward(name: str, reward: gym.RewardWrapper):
    if name not in _registry.keys():
        _registry[name] = reward


register_reward('sparse', reward=SparseReward)
register_reward('continuous', reward=ContinuousReward)
register_reward('stl', reward=STLReward)
register_reward('bool_stl', reward=BoolSTLReward)
register_reward('hier_binary_indicator', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('hier_cont_indicator', reward=GraphWithContinuousScoreContinuousIndicator)
register_reward('hier_disc', reward=GraphWithContinuousTargetAndDiscreteSafety)
