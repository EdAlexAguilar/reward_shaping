from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.bipedal_walker.rewards.baselines import BWEvalConfig
from reward_shaping.envs.bipedal_walker.rewards.potential import BWHierarchicalPotentialShaping, \
    BWUniformScalarizedMultiObjectivization, BWDecreasingScalarizedMultiObjectivization
from reward_shaping.envs.bipedal_walker.rewards.stl_based import BWSTLReward

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
register_reward('tltl', reward=BWSTLReward)  # evaluation on complete episode
register_reward('bhnr', reward=BWSTLReward)  # evaluation with a moving window

# Multi-Objectivization solved via Linear Scalarization
register_reward('morl_uni', reward=BWUniformScalarizedMultiObjectivization)
register_reward('morl_dec', reward=BWDecreasingScalarizedMultiObjectivization)

# Hierarchical Potential Shaping
register_reward('hrs_pot', reward=BWHierarchicalPotentialShaping)

# Evaluation
register_reward('eval', reward=BWEvalConfig)
