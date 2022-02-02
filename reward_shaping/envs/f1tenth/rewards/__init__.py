from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.f1tenth.rewards.baselines import F110EvalConfig, MinActionReward
from reward_shaping.envs.f1tenth.rewards.potential import F110HierarchicalPotentialShaping, \
    F110UniformScalarizedMultiObjectivization, F110DecreasingScalarizedMultiObjectivization
from reward_shaping.envs.f1tenth.rewards.stl_based import F110STLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('default', reward=DefaultReward)
register_reward('min_action', reward=MinActionReward)

# TL-based
register_reward('tltl', reward=F110STLReward)  # evaluation on complete episode
register_reward('bhnr', reward=F110STLReward)  # evaluation with a moving window

# Multi-objectivization solved via linear scalarization
register_reward('morl_uni', reward=F110UniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=F110DecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hrs_pot', reward=F110HierarchicalPotentialShaping)

# Evaluation
register_reward('eval', reward=F110EvalConfig)
