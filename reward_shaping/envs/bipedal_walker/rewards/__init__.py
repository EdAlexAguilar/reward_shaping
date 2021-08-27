from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.bipedal_walker.rewards.baselines import WeightedBaselineReward
from reward_shaping.envs.bipedal_walker.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator
from reward_shaping.envs.bipedal_walker.rewards.stl_based import STLReward

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
register_reward('weighted', reward=WeightedBaselineReward)
# Graph-based
# note: the gbased reward already uses binary formulation for safety rules, then 'gb_cr_bi'=='gb_bcr_bi'
# so both of them are registered referring to the same implementation, just for consistency with other envs (eg, cpole)
register_reward('gb_cr_bi', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('gb_bcr_bi', reward=GraphWithContinuousScoreBinaryIndicator)
# STL-based
# note: also here, the safety requirements are not continuous (as instead cpole), so there is no distinction between
# stl and boolstl but both of them are registered for consistency with other envs
register_reward('stl', reward=STLReward)
register_reward('bool_stl', reward=STLReward)   # used for evaluation (the safety properties are already not continuous)
