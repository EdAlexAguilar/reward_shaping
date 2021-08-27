from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.cart_pole_obst.rewards.baselines import CPOSparseReward, CPOContinuousReward, \
    CPOWeightedBaselineReward
from reward_shaping.envs.cart_pole_obst.rewards.graph_based import CPOGraphWithContinuousScoreBinaryIndicator, \
    CPOGraphWithContinuousScoreContinuousIndicator, CPOGraphWithProgressScoreBinaryIndicator, \
    CPOGraphWithBinarySafetyScoreBinaryIndicator, CPOChainGraph
from reward_shaping.envs.cart_pole_obst.rewards.stl_based import CPOSTLReward, CPOBoolSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward

# Baselines
register_reward('default', reward=DefaultReward)
register_reward('sparse', reward=CPOSparseReward)
register_reward('continuous', reward=CPOContinuousReward)
register_reward('stl', reward=CPOSTLReward)
register_reward('bool_stl', reward=CPOBoolSTLReward)
register_reward('weighted', reward=CPOWeightedBaselineReward)
# Graph-based (gb) formulations
register_reward('gb_cr_bi', reward=CPOGraphWithContinuousScoreBinaryIndicator)
register_reward('gb_cr_ci', reward=CPOGraphWithContinuousScoreContinuousIndicator)
# Graph-based with target score measuring progress (ie, closeness to target w.r.t. the prev step)
register_reward('gb_pcr_bi', reward=CPOGraphWithProgressScoreBinaryIndicator)
# Graph-based with binary score only for safety nodes
register_reward('gb_bcr_bi', reward=CPOGraphWithBinarySafetyScoreBinaryIndicator)
# Graph-based with 1 node for each level, evaluated as conjunction (eg, AND_{collision, falldown, outside})
register_reward('gb_chain', reward=CPOChainGraph)