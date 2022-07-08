from typing import List

import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.bipedal_walker.specs import get_all_specs

gamma = 1.0


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
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = gamma * self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort


class BWHierarchicalPotentialShapingNoComfort(RewardFunction):

    @staticmethod
    def _safety_potential(state, info):
        return safety_collision_potential(state, info)

    def _target_potential(self, state, info):
        safety_w = safety_collision_potential(state, info)
        return safety_w * dist_to_target(state, info)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # base reward
        base_reward = simple_base_reward(next_state, info)
        # shaping
        if info["done"]:
            return base_reward
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        return base_reward + shaping_safety + shaping_target


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
        shaping_coll = gamma * safety_collision_potential(next_state, info) - safety_collision_potential(state, info)
        shaping_target = gamma * dist_to_target(next_state, info) - dist_to_target(state, info)
        shaping_comf_vx = gamma * comfort_vx_potential(next_state, info) - comfort_vx_potential(state, info)
        shaping_comf_vy = gamma * comfort_vy_potential(next_state, info) - comfort_vy_potential(state, info)
        shaping_comf_ang = gamma * comfort_angle_potential(next_state, info) - comfort_angle_potential(state, info)
        shaping_comf_angvel = gamma * comfort_ang_vel_potential(next_state, info) - comfort_ang_vel_potential(state,
                                                                                                              info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_coll, shaping_target, shaping_comf_vx,
                         shaping_comf_vy, shaping_comf_ang, shaping_comf_angvel]):
            reward += w * f
        return reward


class BWUniformScalarizedMultiObjectivization(BWScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(BWUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class BWDecreasingScalarizedMultiObjectivization(BWScalarizedMultiObjectivization):
    """
    weights selected considering a budget of 1.0 + 0.5 + 0.25 = 1.75, then:
        - the sum of safety weights is ~ 1.0/1.75
        - the sum of target weights is ~ 0.50/1.75
        - the sum of comfort weights is ~ 0.25/1.75
    """

    def __init__(self, **kwargs):
        weights = np.array([1.0, 0.5, 0.25, 0.25, 0.25, 0.25])
        weights /= np.sum(weights)
        super(BWDecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


#########################################################################
###              Scalarized Target vs Comfort Multi-Objective         ###
#########################################################################
class BWScalarizedMultiObjectiveTargetVSComfort(RewardFunction):

    def __init__(self, lmbda: float, **kwargs):
        """ Scalarized Reward to study the tradeoff between target and comfort.
        - lmbda:    the weight coefficient for the target
                    (1-lmbda) is the weight coefficient for the aggregated comforts
        """
        self._lambda = lmbda

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # define norm coefficient s.t. target and comfort sum up to 1 under optimal policy
        targ_coeff = dist_to_target(info["initial_state"], info)  # norm progress w.r.t. starting x
        comfort_coeff = (1 / 4) * (1 / info["max_steps"])  # norm comfort w.r.t. nr comfort reqs and time steps
        # compute target, comfort rewards
        target_rew = targ_coeff * dist_to_target(next_state, info) - dist_to_target(state, info)
        comfort_vx = float(state["horizontal_speed"] >= info["speed_x_target"])             # 0 or 1
        comfort_vy = float(abs(state["vertical_speed"]) <= info["speed_y_limit"])           # 0 or 1
        comfort_ang = float(abs(state["hull_angle"]) <= info["angle_hull_limit"])           # 0 or 1
        comfort_angvel = float(abs(state["hull_angle_speed"]) <= info["angle_vel_limit"])   # 0 or 1
        comfort_rew = comfort_coeff * (comfort_vx + comfort_vy + comfort_ang + comfort_angvel)  # avg and norm step
        # linear scalarization of the multi-objectivized requirements
        reward = self._lambda * target_rew + (1 - self._lambda) * comfort_rew
        return reward
