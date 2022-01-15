from reward_shaping.envs.cart_pole_obst.rewards.baselines import CPOSparseReward, CPOContinuousReward, \
    CPOWeightedBaselineReward, CPOEvalConfig, CPOSparseTargetReward
from reward_shaping.envs.cart_pole_obst.rewards.graph_based import CPOGraphWithContinuousScoreBinaryIndicator, \
    CPOGraphWithContinuousScoreContinuousIndicator, CPOGraphWithProgressScoreBinaryIndicator, \
    CPOGraphWithBinarySafetyScoreBinaryIndicator, CPOChainGraph, CPOGraphBinarySafetyProgressTargetContinuousIndicator, \
    CPOGraphContinuousSafetyProgressTargetContinuousIndicator, \
    CPOGraphContinuousSafetyProgressDistanceTargetContinuousIndicator, \
    CPOGraphContinuousSafetyProgressMaxTargetContinuousIndicator, \
    CPOGraphBinarySafetyProgressDistanceTargetContinuousIndicator
from reward_shaping.envs.cart_pole_obst.rewards.stl_based import CPOSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('stl', reward=CPOSTLReward)
register_reward('weighted', reward=CPOWeightedBaselineReward)
register_reward('default', reward=CPOSparseReward)
register_reward('gb_chain', reward=CPOChainGraph)
# Hierarchical
register_reward('gb_bpdr_ci', reward=CPOGraphBinarySafetyProgressDistanceTargetContinuousIndicator)
# Evaluation
register_reward('eval', reward=CPOEvalConfig)

register_reward('sparse_target', reward=CPOSparseTargetReward)
