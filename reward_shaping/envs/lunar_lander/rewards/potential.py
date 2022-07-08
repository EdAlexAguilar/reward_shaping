from typing import List, Dict, Any

import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.lunar_lander.specs import get_all_specs

gamma = 1.0


def safety_collision_potential(state, info):
    assert "collision" in state
    return int(state["collision"] <= 0)


def safety_exit_potential(state, info):
    assert "x" in state and "x_limit" in info
    return int(abs(state["x"]) <= info["x_limit"])


def target_dist2goal_potential(state, info):
    dist_goal = np.linalg.norm([state["x"] - info["x_target"], state["y"] - info["y_target"]])
    return 1.0 - clip_and_norm(dist_goal, 0, 1.5)


def comfort_angle_potential(state, info):
    return 1 - clip_and_norm(abs(state["angle"]), info["angle_limit"], 1.0)


def comfort_angvel_potential(state, info):
    return 1 - clip_and_norm(abs(state["angle_speed"]), info["angle_speed_limit"], 1.0)


def simple_base_reward(state, info):
    dist_x = info["halfwidth_landing_area"] - abs(state["x"])
    dist_y = info["landing_height"] - abs(state["y"])
    return 1.0 if min(dist_x, dist_y) >= 0 else 0.0


#########################################################################
###           Hierachical Potential-based Reward Shaping (HPRS)       ###
#########################################################################
class LLHierarchicalShapingOnSparseTargetReward(RewardFunction):
    def _safety_potential(self, state, info):
        collision_reward = safety_collision_potential(state, info)
        exit_reward = safety_exit_potential(state, info)
        return collision_reward + exit_reward

    def _target_potential(self, state, info):
        target_reward = target_dist2goal_potential(state, info)
        # hierarchical weights
        safety_weight = safety_collision_potential(state, info) * safety_exit_potential(state, info)
        return safety_weight * target_reward

    def _comfort_potential(self, state, info):
        angle_reward = comfort_angle_potential(state, info)
        angvel_reward = comfort_angvel_potential(state, info)
        # hierarchical weights
        safety_weight = safety_collision_potential(state, info) * safety_exit_potential(state, info)
        target_weight = target_dist2goal_potential(state, info)
        return safety_weight * target_weight * (angle_reward + angvel_reward)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # hierarchical shaping
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = gamma * self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort


#########################################################################
###           Scalarized Potential-based Multi-objectivization        ###
#########################################################################
class LLScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_collision = gamma * safety_collision_potential(next_state, info) - safety_collision_potential(state,
                                                                                                              info)
        shaping_exit = gamma * safety_exit_potential(next_state, info) - safety_exit_potential(state, info)
        shaping_target = gamma * target_dist2goal_potential(next_state, info) - target_dist2goal_potential(state,
                                                                                                           info)
        shaping_comf_ang = gamma * comfort_angle_potential(next_state, info) - comfort_angle_potential(state, info)
        shaping_comf_angvel = gamma * comfort_angvel_potential(next_state, info) - comfort_angvel_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_collision, shaping_exit, shaping_target, shaping_comf_ang, shaping_comf_angvel]):
            reward += w * f
        return reward


class LLUniformScalarizedMultiObjectivization(LLScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(LLUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class LLDecreasingScalarizedMultiObjectivization(LLScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        """
        weights selected considering a budget of 1.0 + 0.5 + 0.25 = 1.75, then:
            - the sum of safety weights is ~ 1.0/1.75
            - the sum of target weights is ~ 0.50/1.75
            - the sum of comfort weights is ~ 0.25/1.75
        """
        weights = np.array([1.0, 1.0, 0.5, 0.25, 0.25])
        weights /= np.sum(weights)
        super(LLDecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


#########################################################################
###              Scalarized Target vs Comfort Multi-Objective         ###
#########################################################################
class LLScalarizedMultiObjectiveTargetVSComfort(RewardFunction):

    def __init__(self, env_params: Dict[str, Any], **kwargs):
        """ Scalarized Reward to study the tradeoff between target and comfort.
        - lambda:    the weight coefficient for the target
                    (1-lambda) is the weight coefficient for the aggregated comforts
        """
        assert "lambda" in env_params, "missing lambda in params"
        self._lambda = env_params["lambda"]

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # define norm coefficient s.t. target and comfort sum up to 1 under optimal policy
        targ_coeff = 1 / (1 - target_dist2goal_potential(info["initial_state"], info))  # norm progress w.r.t. starting x
        comfort_coeff = (1 / 2) * (1 / info["max_steps"])  # norm comfort w.r.t. nr comfort reqs and time steps
        # compute target, comfort rewards
        target_rew = targ_coeff * (target_dist2goal_potential(next_state, info) - target_dist2goal_potential(state, info))
        comfort_angle = float(abs(state["angle"]) <= info["angle_limit"])                   # 0 or 1
        comfort_angvel = float(abs(state["angle_speed"]) <= info["angle_speed_limit"])      # 0 or 1
        comfort_rew = comfort_coeff * (comfort_angle + comfort_angvel)                      # avg and norm step
        # linear scalarization of the multi-objectivized requirements
        reward = self._lambda * target_rew + (1 - self._lambda) * comfort_rew
        return reward
