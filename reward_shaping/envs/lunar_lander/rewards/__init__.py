from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.lunar_lander.rewards.graph_based import LLGraphWithBinarySafetyBinaryIndicator
from reward_shaping.envs.lunar_lander.rewards.stl_based import STLReward

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
register_reward('stl', reward=STLReward)
# Graph-based
register_reward('gb_bcr_bi', reward=LLGraphWithBinarySafetyBinaryIndicator)