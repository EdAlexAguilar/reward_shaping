from reward_shaping.envs.highway_env.rewards.baselines import HighwaySparseTargetReward, HighwayEvalConfig
from reward_shaping.envs.highway_env.rewards.potential import HighwayHierarchicalPotentialShaping, \
    HighwayUniformScalarizedMultiObjectivization, HighwayDecreasingScalarizedMultiObjectivization
from reward_shaping.envs.highway_env.rewards.stl_based import HighwaySTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('default', reward=HighwaySparseTargetReward)

# TL-based
register_reward('tltl', reward=HighwaySTLReward)  # evaluation on complete episode
#register_reward('bhnr', reward=HighwaySTLReward)  # evaluation with a moving window

# Multi-objectivization solved via linear scalarization
#register_reward('morl_uni', reward=HighwayUniformScalarizedMultiObjectivization)
#register_reward('morl_dec', reward=HighwayDecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hprs', reward=HighwayHierarchicalPotentialShaping)

# Evaluation
register_reward('eval', reward=HighwayEvalConfig)