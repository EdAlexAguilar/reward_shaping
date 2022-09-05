from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.racecar2.rewards.baselines import RC2EvalConfig
from reward_shaping.envs.racecar2.rewards.potential import RC2UniformScalarizedMultiObjectivization, \
    RC2HierarchicalPotentialShaping, RC2DecreasingScalarizedMultiObjectivization, \
    RC2HierarchicalPotentialShapingNoComfort
from reward_shaping.envs.racecar2.rewards.stl_based import RC2STLReward

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
register_reward('tltl', reward=RC2STLReward)  # evaluate on complete episode
register_reward('bhnr', reward=RC2STLReward)  # evaluate with a moving window

# Multi-objectivization solved via linear scalarization
register_reward('morl_uni', reward=RC2UniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=RC2DecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hprs', reward=RC2HierarchicalPotentialShaping)
register_reward('hprs_nocomf', reward=RC2HierarchicalPotentialShapingNoComfort)

# Evaluation
register_reward('eval', reward=RC2EvalConfig)