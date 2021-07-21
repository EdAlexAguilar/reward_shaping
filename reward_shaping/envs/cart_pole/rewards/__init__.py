import gym

from reward_shaping.envs.cart_pole.rewards.baselines import ContinuousReward, SparseReward, STLReward

_registry = {}


def get_reward(name: str) -> gym.RewardWrapper:
    return _registry[name]


def register_reward(name: str, reward: gym.RewardWrapper):
    if name not in _registry.keys():
        _registry[name] = reward


register_reward('sparse', reward=SparseReward)
register_reward('continuous', reward=ContinuousReward)
register_reward('stl', reward=STLReward)
