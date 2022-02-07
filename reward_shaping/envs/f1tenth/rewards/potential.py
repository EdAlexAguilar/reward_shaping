from typing import List

import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.f1tenth.specs import get_all_specs

gamma = 1.0


def safety_collision_potential(state, info):
    assert "collision" in state
    return int(state["collision"] <= 0)


def safety_reverse_potential(state, info):
    assert "reverse" in state
    return int(state["reverse"] <= 0)


def target_potential(state, info):
    assert "progress_meters" in state and "progress_target_meters" in info
    return clip_and_norm(state["progress_meters"], 0, info["progress_target_meters"])


def comfort_speed_potential(state, info):
    assert "velocity" in state and "comfortable_speed_limit" in info
    return 1.0 - clip_and_norm(state["velocity"][0], info["comfortable_speed_limit"],
                               info["max_speed"])  # 0 > threshold


def comfort_steering_potential(state, info):
    assert "steering_cmd" in state and "comfortable_steering" in info
    return 1.0 - clip_and_norm(abs(state["steering_cmd"][0]), info["comfortable_steering"], info["max_steering"])


def comfort_lane_potential(state, info):
    assert "dist_to_lane" in state and "favourite_lane" in info
    target = -0.25   # we aim to drive at 10cm from centerline, empirically choosen (InformatikLectureHall)
    dist_to_target = (state["dist_to_lane"] - target)**2
    return 1.0 - clip_and_norm(dist_to_target, 0.0, 0.60**2)    # assume max dist to centerline is ~60cm


def simple_base_reward(state, info):
    assert "progress_meters" in state and "progress_target_meters" in info
    base_reward = 1.0 if state["progress_meters"] >= info["progress_target_meters"] else 0.0
    return base_reward


class F110HierarchicalPotentialShaping(RewardFunction):

    @staticmethod
    def _safety_potential(state, info):
        collision_reward = safety_collision_potential(state, info)
        reverse_reward = safety_reverse_potential(state, info)
        return collision_reward + reverse_reward

    def _target_potential(self, state, info):
        safety_w = safety_collision_potential(state, info) * safety_reverse_potential(state, info)
        return safety_w * target_potential(state, info)

    def _comfort_potential(self, state, info):
        comfort_speed = comfort_speed_potential(state, info)
        comfort_steering = comfort_steering_potential(state, info)
        comfort_lane = comfort_lane_potential(state, info)
        # hierarchical weights
        safety_w = safety_collision_potential(state, info) * safety_reverse_potential(state, info)
        target_w = target_potential(state, info)
        return safety_w * target_w * (comfort_speed + comfort_steering + comfort_lane)

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


class F110ScalarizedMultiObjectivization(RewardFunction):

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
        shaping_reverse = gamma * safety_reverse_potential(next_state, info) - safety_reverse_potential(state, info)
        shaping_target = gamma * target_potential(next_state, info) - target_potential(state, info)
        shaping_comf_speed = gamma * comfort_speed_potential(next_state, info) - comfort_speed_potential(state, info)
        shaping_comf_steer = gamma * comfort_steering_potential(next_state, info) - comfort_steering_potential(state,
                                                                                                               info)
        shaping_comf_lane = gamma * comfort_lane_potential(next_state, info) - comfort_lane_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_coll, shaping_reverse, shaping_target, shaping_comf_speed, shaping_comf_steer, shaping_comf_lane]):
            reward += w * f
        return reward


class F110UniformScalarizedMultiObjectivization(F110ScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(F110UniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class F110DecreasingScalarizedMultiObjectivization(F110ScalarizedMultiObjectivization):
    """
    weights selected considering a budget of 1.0 + 0.5 + 0.25 = 1.75, then:
        - the sum of safety weights is ~ 1.0/1.75
        - the sum of target weights is ~ 0.50/1.75
        - the sum of comfort weights is ~ 0.25/1.75
    """

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 0.5, 0.25, 0.25, 0.25])
        weights /= np.sum(weights)
        super(F110DecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)
