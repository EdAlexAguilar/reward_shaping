from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.racecar.rewards.baselines import RacecarEvalConfig, RacecarMinActionReward
from reward_shaping.envs.racecar.rewards.potential import RacecarUniformScalarizedMultiObjectivization, \
    RacecarDecreasingScalarizedMultiObjectivization, RacecarHierarchicalPotentialShaping
from reward_shaping.envs.racecar.rewards.stl_based import RacecarSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('default', reward=DefaultReward)
register_reward('min_action', reward=RacecarMinActionReward)

# TL-based
register_reward('tltl', reward=RacecarSTLReward)  # evaluation on complete episode
register_reward('bhnr', reward=RacecarSTLReward)  # evaluation with a moving window

# Multi-objectivization solved via linear scalarization
register_reward('morl_uni', reward=RacecarUniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=RacecarDecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hrs_pot', reward=RacecarHierarchicalPotentialShaping)

# Evaluation
register_reward('eval', reward=RacecarEvalConfig)
