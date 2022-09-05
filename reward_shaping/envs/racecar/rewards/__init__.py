from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.racecar.rewards.baselines import RCEvalConfig
from reward_shaping.envs.racecar.rewards.potential import RCHierarchicalPotentialShaping, \
    RCUniformScalarizedMultiObjectivization, RCDecreasingScalarizedMultiObjectivization, \
    RCHierarchicalPotentialShapingNoComfort
from reward_shaping.envs.racecar.rewards.stl_based import RCSTLReward

_registry = {}

def get_reward(name: str):
    try:
        reward = _registry[name]
    except KeyError:
        raise KeyError(f"the reward {name} is not registered")
    return reward


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('default', reward=DefaultReward)

# TL-based
register_reward('tltl', reward=RCSTLReward)  # evaluate on complete episode
register_reward('bhnr', reward=RCSTLReward)  # evaluate with a moving window

# Multi-objectivization solved via linear scalarization
register_reward('morl_uni', reward=RCUniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=RCDecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hprs', reward=RCHierarchicalPotentialShaping)
register_reward('hprs_nocomf', reward=RCHierarchicalPotentialShapingNoComfort)

# Evaluation
register_reward('eval', reward=RCEvalConfig)