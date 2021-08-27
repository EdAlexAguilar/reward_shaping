from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.bipedal_walker.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator
from reward_shaping.envs.bipedal_walker.rewards.stl_based import STLReward, BoolSTLReward

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
# Graph-based
register_reward('gb_cr_bi', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('stl', reward=STLReward)
register_reward('bool_stl', reward=STLReward)   # used for evaluation (the safety properties are already not continuous)