from typing import List

import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.racecar.specs import get_all_specs

gamma = 1.0


def safety_collision_potential(state, info):
    assert "collision" in state
    return int(state["collision"] <= 0)


def dist_to_target(state, info):
    assert "progress" in state and "target_progress" in info
    return clip_and_norm(state["progress"], 0.0, info["target_progress"])


def comfort_dist2obst(state, info):
    assert "dist2obst" in state and "target_dist2obst" in info
    return clip_and_norm(state["dist2obst"], 0.0, info["target_dist2obst"])

def comfort_min_speed_cmd(state, info):
    assert "last_actions" in state and "min_speed_cmd" in info
    speed_cmd = state["last_actions"][-1][1]
    return clip_and_norm(speed_cmd, 0.0, info["target_dist2obst"])


def simple_base_reward(state, info):
    assert "progress" in state and "target_progress" in info
    base_reward = 1.0 if state["progress"] >= info["target_progress"] else 0.0
    return base_reward


class RCHierarchicalPotentialShaping(RewardFunction):

    @staticmethod
    def _safety_potential(state, info):
        return safety_collision_potential(state, info)

    @staticmethod
    def _target_potential(state, info):
        safety_w = safety_collision_potential(state, info)
        return safety_w * dist_to_target(state, info)

    @staticmethod
    def _comfort_potential(state, info):
        comfort_d2o = comfort_dist2obst(state, info)
        # hierarchical weights
        safety_w = safety_collision_potential(state, info)
        target_w = dist_to_target(state, info)
        return safety_w * target_w * (comfort_d2o)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # base reward
        base_reward = simple_base_reward(next_state, info)
        # shaping
        if info["done"]:
            return base_reward
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = gamma * self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort


class RCHierarchicalPotentialShapingNoComfort(RCHierarchicalPotentialShaping):

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # base reward
        base_reward = simple_base_reward(next_state, info)
        # shaping
        if info["done"]:
            return base_reward
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        return base_reward + shaping_safety + shaping_target


class RCScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_coll = gamma * safety_collision_potential(next_state, info) - safety_collision_potential(state, info)
        shaping_target = gamma * dist_to_target(next_state, info) - dist_to_target(state, info)
        shaping_comf_d20 = gamma * comfort_dist2obst(next_state, info) - comfort_dist2obst(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_coll, shaping_target, shaping_comf_d20]):
            reward += w * f
        return reward


class RCUniformScalarizedMultiObjectivization(RCScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(RCUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class RCDecreasingScalarizedMultiObjectivization(RCScalarizedMultiObjectivization):
    """
    weights selected considering a budget of 1.0 + 0.5 + 0.25 = 1.75, then:
        - the sum of safety weights is ~ 1.0/1.75
        - the sum of target weights is ~ 0.50/1.75
        - the sum of comfort weights is ~ 0.25/1.75
    """

    def __init__(self, **kwargs):
        weights = np.array([1.0, 0.5, 0.25/5, 0.25/5, 0.25/5, 0.25/5, 0.25/5])
        weights /= np.sum(weights)
        super(RCDecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)
