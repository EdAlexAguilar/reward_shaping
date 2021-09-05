from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.bipedal_walker.rewards.baselines import BWWeightedBaselineReward, BWEvalConfig
from reward_shaping.envs.bipedal_walker.rewards.graph_based import BWChainGraph, \
    BWGraphWithContinuousScoreBinaryIndicator, BWGraphWithBinarySafetyProgressTargetContinuousIndicator, \
    BWGraphWithBinarySafetyProgressTargetContinuousIndicatorNoComfort
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
register_reward('stl', reward=BWSTLReward)
register_reward('eval', reward=BWEvalConfig)
register_reward('weighted', reward=BWWeightedBaselineReward)
register_reward('default', reward=DefaultReward)
register_reward('gb_chain', reward=BWChainGraph)
# Graph-based
register_reward('gb_bpr_ci', reward=BWGraphWithBinarySafetyProgressTargetContinuousIndicator)
register_reward('gb_bpr_bi', reward=BWGraphWithContinuousScoreBinaryIndicator)
register_reward('gb_bpr_ci_noc', reward=BWGraphWithBinarySafetyProgressTargetContinuousIndicatorNoComfort)


