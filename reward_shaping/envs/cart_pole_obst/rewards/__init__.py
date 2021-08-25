import gym

from reward_shaping.envs.cart_pole_obst.rewards.baselines import SparseReward, ContinuousReward
from reward_shaping.envs.cart_pole_obst.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator, \
    GraphWithContinuousScoreContinuousIndicator, GraphWithProgressScoreBinaryIndicator, \
    GraphWithBinarySafetyScoreBinaryIndicator
from reward_shaping.envs.cart_pole_obst.rewards.stl_based import STLReward, BoolSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward

# Baselines
register_reward('sparse', reward=SparseReward)
register_reward('continuous', reward=ContinuousReward)
register_reward('stl', reward=STLReward)
register_reward('bool_stl', reward=BoolSTLReward)
# Graph-based (gb) formulations
register_reward('gb_cr_bi', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('gb_cr_ci', reward=GraphWithContinuousScoreContinuousIndicator)
# Progress
register_reward('gb_pcr_bi', reward=GraphWithProgressScoreBinaryIndicator)
register_reward('gb_bcr_bi', reward=GraphWithBinarySafetyScoreBinaryIndicator)

"""
register_reward('hier_cont', reward=GraphWithContinuousScore)
register_reward('hier_cont_pot', reward=PotentialGraphWithContinuousScore)
register_reward('hier_disc', reward=GraphWithContinuousTargetAndDiscreteSafety)
register_reward('hier_disc_pot', reward=PotentialGraphWithContinuousTargetAndDiscreteSafety)
register_reward('hier_binary_ind', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('hier_cont_ind', reward=GraphWithContinuousScoreContinuousIndicator)
"""