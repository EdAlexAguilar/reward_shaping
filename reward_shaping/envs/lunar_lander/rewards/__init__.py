from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.lunar_lander.rewards.baselines import LLEvalConfig
from reward_shaping.envs.lunar_lander.rewards.potential import LLHierarchicalShapingOnSparseTargetReward, \
    LLUniformScalarizedMultiObjectivization, LLDecreasingScalarizedMultiObjectivization
from reward_shaping.envs.lunar_lander.rewards.stl_based import LLSTLReward

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
register_reward('tltl', reward=LLSTLReward)  # evaluate on complete episode
register_reward('bhnr', reward=LLSTLReward)  # evaluate with a moving window

# Multi-objectivization solved via linear scalarization
register_reward('morl_uni', reward=LLUniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=LLDecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hrs_pot', reward=LLHierarchicalShapingOnSparseTargetReward)

# Evaluation
register_reward('eval', reward=LLEvalConfig)
