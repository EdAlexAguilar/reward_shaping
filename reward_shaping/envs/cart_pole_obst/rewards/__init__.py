import gym

from reward_shaping.envs import ContinuousReward, SparseReward, STLReward
from reward_shaping.envs import GraphWithContinuousScore, PotentialGraphWithContinuousScore, \
    GraphWithContinuousTargetAndDiscreteSafety, PotentialGraphWithContinuousTargetAndDiscreteSafety

_registry = {}


def get_reward(name: str) -> gym.RewardWrapper:
    return _registry[name]


def register_reward(name: str, reward: gym.RewardWrapper):
    if name not in _registry.keys():
        _registry[name] = reward


register_reward('sparse', reward=SparseReward)
register_reward('continuous', reward=ContinuousReward)
register_reward('stl', reward=STLReward)
register_reward('hier_cont', reward=GraphWithContinuousScore)
register_reward('hier_cont_pot', reward=PotentialGraphWithContinuousScore)
register_reward('hier_disc', reward=GraphWithContinuousTargetAndDiscreteSafety)
register_reward('hier_disc_pot', reward=PotentialGraphWithContinuousTargetAndDiscreteSafety)