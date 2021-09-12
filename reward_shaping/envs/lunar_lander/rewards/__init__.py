from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.lunar_lander.rewards.baselines import LLEvalConfig, LLWeightedBaselineReward
from reward_shaping.envs.lunar_lander.rewards.graph_based import LLGraphWithBinarySafetyBinaryIndicator, LLChainGraph, \
    LLGraphWithBinarySafetyContinuousIndicator, LLGraphWithBinarySafetyProgressTimesDistanceTargetContinuousIndicator
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
register_reward('stl', reward=LLSTLReward)
register_reward('gb_chain', reward=LLChainGraph)
register_reward('weighted', reward=LLWeightedBaselineReward)
# Graph-based
register_reward('gb_bpdr_ci', reward=LLGraphWithBinarySafetyProgressTimesDistanceTargetContinuousIndicator)
# Evaluation
register_reward('eval', reward=LLEvalConfig)
