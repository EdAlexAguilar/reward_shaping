from typing import List, Union

import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.racecar2.specs import get_all_specs

gamma = 1.0


def safety_collision_potential(state, info):
    assert "collision" in state
    return int(state["collision"] <= 0)


def safety_distance_potential(state, info):
    assert "dist_ego2npc" in state and "safety_distance" in info
    return int(state["dist_ego2npc"] <= info["safety_distance"])


def dist_to_target(state, info):
    assert "progress" in state and "target_progress" in info
    return clip_and_norm(state["progress"], 0.0, info["target_progress"])


def comfort_dist2obst(state, info):
    assert "dist2obst" in state and "target_dist2obst" in info
    return clip_and_norm(state["dist2obst"], 0.0, info["target_dist2obst"])


def comfort_small_steer(state, info):
    # assume target steering is 0
    assert "last_actions" in state
    abs_steering = abs(state["last_actions"][-1][0])
    return 1.0 - clip_and_norm(abs_steering, 0.0, info["comfort_max_steering"])


def comfort_min_velx(state, info):
    assert "velocity_x" in state and "min_velx" in info
    return clip_and_norm(state["velocity_x"][0], 0.0, info["min_velx"])


def comfort_max_velx(state, info):
    assert "velocity_x" in state and "max_velx" in info and "limit_velx" in info
    return 1 - clip_and_norm(state["velocity_x"][0], info["max_velx"], info["limit_velx"])


def comfort_smooth_control(state, info):
    assert "last_actions" in state and "comfort_max_norm" in info
    l2norm_action = np.linalg.norm(state["last_actions"][-1] - state["last_actions"][-2])
    max_l2norm = np.sqrt(8)  # assume action_1=[-1, -1], action_2=[1, 1]
    return 1.0 - clip_and_norm(l2norm_action, info["comfort_max_norm"], max_l2norm)


def comfort_min_comfort_dist(state, info):
    # note: we assume dist_ego2npc to be bounded by a large negative distance, e.g., -5.0 meters
    assert "dist_ego2npc" in state and "min_comfort_distance" in info
    return clip_and_norm(state["dist_ego2npc"], -5.0, info["min_comfort_distance"])


def comfort_max_comfort_dist(state, info):
    # note: dist_ego2npc can be at most be safety distance, because beyond that the episode terminates
    assert "dist_ego2npc" in state and "max_comfort_distance" in info
    return 1 - clip_and_norm(state["dist_ego2npc"], info["max_comfort_distance"], info["safety_distance"])


def simple_base_reward(state, info):
    assert "progress" in state and "target_progress" in info
    base_reward = 1.0 if state["progress"] >= info["target_progress"] else 0.0
    return base_reward


class RC2HierarchicalPotentialShaping(RewardFunction):

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
        comfort_steer = comfort_small_steer(state, info)
        comfort_minv = comfort_min_velx(state, info)
        comfort_maxv = comfort_max_velx(state, info)
        comfort_smooth = comfort_smooth_control(state, info)
        comfort_mindist = comfort_min_comfort_dist(state, info)
        comfort_maxdist = comfort_max_comfort_dist(state, info)
        # hierarchical weights
        safety_w = safety_collision_potential(state, info)
        target_w = dist_to_target(state, info)
        return safety_w * target_w * (comfort_d2o + comfort_steer + comfort_minv + comfort_maxv +
                                      comfort_smooth + comfort_mindist + comfort_maxdist)

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


class RC2HierarchicalPotentialShapingNoComfort(RC2HierarchicalPotentialShaping):

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # base reward
        base_reward = simple_base_reward(next_state, info)
        # shaping
        if info["done"]:
            return base_reward
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        return base_reward + shaping_safety + shaping_target


class RC2ScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: Union[np.ndarray, List[float]], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_coll = gamma * safety_collision_potential(next_state, info) - safety_collision_potential(state, info)
        shaping_safe_dist = gamma * safety_distance_potential(next_state, info) - safety_distance_potential(state, info)
        shaping_target = gamma * dist_to_target(next_state, info) - dist_to_target(state, info)
        shaping_comf_d20 = gamma * comfort_dist2obst(next_state, info) - comfort_dist2obst(state, info)
        shaping_comf_steer = gamma * comfort_small_steer(next_state, info) - comfort_small_steer(state, info)
        shaping_comf_minv = gamma * comfort_min_velx(next_state, info) - comfort_min_velx(state, info)
        shaping_comf_maxv = gamma * comfort_max_velx(next_state, info) - comfort_max_velx(state, info)
        shaping_comf_smooth = gamma * comfort_smooth_control(next_state, info) - comfort_smooth_control(state, info)
        shaping_comf_mindist = gamma * comfort_min_comfort_dist(next_state, info) - comfort_min_comfort_dist(state, info)
        shaping_comf_maxdist = gamma * comfort_max_comfort_dist(next_state, info) - comfort_max_comfort_dist(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_coll, shaping_safe_dist, shaping_target,
                         shaping_comf_d20, shaping_comf_steer,
                         shaping_comf_minv, shaping_comf_maxv,
                         shaping_comf_smooth, shaping_comf_mindist,
                         shaping_comf_maxdist]):
            reward += w * f
        return reward


class RC2UniformScalarizedMultiObjectivization(RC2ScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(RC2UniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class RC2DecreasingScalarizedMultiObjectivization(RC2ScalarizedMultiObjectivization):
    """
    weights selected according to the class:
        - safety reqs have weight 1.0
        - target req has weight 0.5
        - comfort reqs have weight 0.25
    """

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        weights /= np.sum(weights)
        super(RC2DecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)
