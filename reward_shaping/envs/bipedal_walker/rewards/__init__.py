from reward_shaping.envs.bipedal_walker.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
# todo
# Graph-based
register_reward('gb_cr_bi', reward=GraphWithContinuousScoreBinaryIndicator)
