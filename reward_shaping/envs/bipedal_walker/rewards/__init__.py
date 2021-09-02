from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.bipedal_walker.rewards.baselines import BWWeightedBaselineReward
from reward_shaping.envs.bipedal_walker.rewards.graph_based import BWChainGraph, \
    BWGraphWithContinuousScoreBinaryIndicator, BWGraphWithBinarySafetyProgressTargetContinuousIndicator
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
register_reward('weighted', reward=BWWeightedBaselineReward)
register_reward('default', reward=DefaultReward)
register_reward('gb_chain', reward=BWChainGraph)
# Graph-based
register_reward('gb_bpr_ci', reward=BWGraphWithBinarySafetyProgressTargetContinuousIndicator)
#register_reward('gb_cpr_ci', reward=TODO)


# note: the gbased reward already uses binary formulation for safety rules, then 'gb_cr_bi'=='gb_bcr_bi'
# so both of them are registered referring to the same implementation, just for consistency with other envs (eg, cpole)
register_reward('gb_bcr_bi', reward=BWGraphWithContinuousScoreBinaryIndicator)

