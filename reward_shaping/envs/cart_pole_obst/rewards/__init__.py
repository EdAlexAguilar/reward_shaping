from reward_shaping.envs.cart_pole_obst.rewards.baselines import CPOSparseReward, CPOEvalConfig
from reward_shaping.envs.cart_pole_obst.rewards.potential import CPOHierarchicalPotentialShaping, \
    CPOUniformScalarizedMultiObjectivization, CPODecreasingScalarizedMultiObjectivization
from reward_shaping.envs.cart_pole_obst.rewards.stl_based import CPOSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('default', reward=CPOSparseReward)

# TL-based
register_reward('tltl', reward=CPOSTLReward)  # evaluation on complete episode
register_reward('bhnr', reward=CPOSTLReward)  # evaluation with a moving window

# Multi-objectivization solved via linear scalarization
register_reward('morl_uni', reward=CPOUniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=CPODecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hprs', reward=CPOHierarchicalPotentialShaping)

# Evaluation
register_reward('eval', reward=CPOEvalConfig)
