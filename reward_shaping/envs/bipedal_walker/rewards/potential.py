from typing import List

import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.bipedal_walker.specs import get_all_specs


def safety_collision_potential(state, info):
    assert "collision" in state
    return int(state["collision"] <= 0)


def dist_to_target(state, info):
    assert "x" in state
    return np.clip(state["x"], 0.0, 1.0)  # already normalize, safety clipping to avoid unexpected values


def comfort_vx_potential(state, info):
    assert "horizontal_speed" in state
    assert "speed_x_target" in info
    return clip_and_norm(state["horizontal_speed"], 0.0, info["speed_x_target"])


def comfort_vy_potential(state, info):
    assert "vertical_speed" in state
    assert "speed_y_limit" in info
    return 1.0 - clip_and_norm(abs(state["vertical_speed"]), info["speed_y_limit"], 1.0)


def comfort_angle_potential(state, info):
    assert "hull_angle" in state
    assert "angle_hull_limit" in info
    return 1.0 - clip_and_norm(abs(state["hull_angle"]), info["angle_hull_limit"], 1.0)


def comfort_ang_vel_potential(state, info):
    assert "hull_angle_speed" in state
    assert "angle_vel_limit" in info
    return 1.0 - clip_and_norm(abs(state["hull_angle_speed"]), info["angle_vel_limit"], 1.0)


def simple_base_reward(state, info):
    assert "x" in state and "norm_target_x" in info
    base_reward = 1.0 if state["x"] >= info["norm_target_x"] else 0.0
    return base_reward


class BWHierarchicalPotentialShaping(RewardFunction):

    @staticmethod
    def _safety_potential(state, info):
        return safety_collision_potential(state, info)

    def _target_potential(self, state, info):
        safety_w = safety_collision_potential(state, info)
        return safety_w * dist_to_target(state, info)

    def _comfort_potential(self, state, info):
        comfort_vx = comfort_vx_potential(state, info)
        comf_angle = comfort_angle_potential(state, info)
        comf_vy = comfort_vy_potential(state, info)
        comf_angle_vel = comfort_ang_vel_potential(state, info)
        # hierarchical weights
        safety_w = safety_collision_potential(state, info)
        target_w = dist_to_target(state, info)
        return safety_w * target_w * (comfort_vx + comf_vy + comf_angle + comf_angle_vel)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # base reward
        base_reward = simple_base_reward(next_state, info)
        # shaping
        if info["done"]:
            return base_reward
        shaping_safety = self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort


class BWScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_coll = safety_collision_potential(next_state, info) - safety_collision_potential(state, info)
        shaping_target = dist_to_target(next_state, info) - dist_to_target(state, info)
        shaping_comf_vx = comfort_vx_potential(next_state, info) - comfort_vx_potential(state, info)
        shaping_comf_vy = comfort_vy_potential(next_state, info) - comfort_vy_potential(state, info)
        shaping_comf_ang = comfort_angle_potential(next_state, info) - comfort_angle_potential(state, info)
        shaping_comf_angvel = comfort_ang_vel_potential(next_state, info) - comfort_ang_vel_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_coll, shaping_target, shaping_comf_vx,
                         shaping_comf_vy, shaping_comf_ang, shaping_comf_angvel]):
            reward += w * f
        return reward


class BWUniformScalarizedMultiObjectivization(BWScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = [0.167, 0.167, 0.167, 0.167, 0.167,  0.165]   # 0.165 to have sum = 1.0
        super(BWUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class BWDecreasingScalarizedMultiObjectivization(BWScalarizedMultiObjectivization):
    """
    weights selected considering a budget of 1.0 + 0.5 + 0.25 = 1.75, then:
        - the sum of safety weights is ~ 1.0/1.75
        - the sum of target weights is ~ 0.50/1.75
        - the sum of comfort weights is ~ 0.25/1.75
    """
    def __init__(self, **kwargs):
        weights = [0.56, 0.28, 0.04, 0.04, 0.04,  0.04]
        super(BWDecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)
