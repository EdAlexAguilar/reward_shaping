from typing import Union, List, Dict, Any

from reward_shaping.core.reward import RewardFunction
import numpy as np

from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.cart_pole_obst.specs import get_all_specs

gamma = 1.0


def safety_falldown_potential(state, info):
    assert "theta" in state and "theta_limit" in info
    falldown = (state["theta"] <= info["theta_limit"])
    return float(falldown)


def safety_exit_potential(state, info):
    assert "x" in state and "x_limit" in info
    outside = (state["x"] <= info["x_limit"])
    return float(outside)


def safety_collision_potential(state, info):
    assert "collision" in state
    collision = (state["collision"] <= 0)
    return float(collision)


def target_dist2goal_potential(state, info):
    assert "x" in state and "theta" in state and "x_target" in info
    assert "axle_y" in info and "pole_length" in info
    x, theta = state['x'], state['theta']
    pole_x, pole_y = x + info['pole_length'] * np.sin(theta), info['axle_y'] + info['pole_length'] * np.cos(theta)
    goal_x, goal_y = info['x_target'], info['axle_y'] + info['pole_length']
    dist_goal = np.linalg.norm([goal_x - pole_x, goal_y - pole_y])
    target_reward = 1 - np.clip(dist_goal, 0, 2.5) / 2.5
    return target_reward


def comfort_balance_potential(state, info):
    assert "theta" in state and "theta_limit" in info and "theta_target_tol" in info
    comfort_reward = 1 - clip_and_norm(abs(state["theta"]), info["theta_target_tol"], info["theta_limit"])
    return comfort_reward


def simple_base_reward(state, info):
    """
    sparse reward which returns +1 when the target configuration has been reached,
    otherwise 0.
    """
    pole_x = state["x"] + info['pole_length'] * np.sin(state["theta"])
    pole_y = info['axle_y'] + info['pole_length'] * np.cos(state["theta"])
    goal_x, goal_y = info['x_target'], info['axle_y'] + info['pole_length']
    check_goal = np.linalg.norm([goal_x - pole_x, goal_y - pole_y]) <= info["dist_target_tol"]
    return 1.0 if check_goal else 0.0


#########################################################################
###        Hierarchical Potential-based Reward Shaping (HPRS)         ###
#########################################################################
class CPOHierarchicalPotentialShaping(RewardFunction):

    def _safety_potential(self, state, info):
        falldown_reward = safety_falldown_potential(state, info)
        exit_reward = safety_exit_potential(state, info)
        collision_reward = safety_collision_potential(state, info)
        return falldown_reward + exit_reward + collision_reward

    def _target_potential(self, state, info):
        """
        idea: since the task is to conquer the origin, the potential of a state depends on two factors:
            - the distance to the target (if not reached yet), and the persistence on the target (once reached)
        """
        target_reward = target_dist2goal_potential(state, info)
        # hierarchical weights
        falldown_reward = safety_falldown_potential(state, info)
        exit_reward = safety_exit_potential(state, info)
        collision_reward = safety_collision_potential(state, info)
        safety_w = falldown_reward * exit_reward * collision_reward
        return safety_w * target_reward

    def _comfort_potential(self, state, info):
        comfort_reward = comfort_balance_potential(state, info)
        # hierarchical weights
        falldown_reward = safety_falldown_potential(state, info)
        exit_reward = safety_exit_potential(state, info)
        collision_reward = safety_collision_potential(state, info)
        safety_w = falldown_reward * exit_reward * collision_reward
        target_w = target_dist2goal_potential(state, info)
        return safety_w * target_w * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # hierarchical shaping function
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = gamma * self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort


#########################################################################
###         Scalarized Potential-based Multi-Objectivization          ###
#########################################################################
class CPOScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_falldown = gamma * safety_falldown_potential(next_state, info) - safety_falldown_potential(state, info)
        shaping_exit = gamma * safety_exit_potential(next_state, info) - safety_exit_potential(state, info)
        shaping_coll = gamma * safety_collision_potential(next_state, info) - safety_collision_potential(state, info)
        shaping_target = gamma * target_dist2goal_potential(next_state, info) - target_dist2goal_potential(state,
                                                                                                           info)
        shaping_comfort = gamma * comfort_balance_potential(next_state, info) - comfort_balance_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights, [shaping_falldown, shaping_exit, shaping_coll, shaping_target, shaping_comfort]):
            reward += w * f
        return reward


class CPOUniformScalarizedMultiObjectivization(CPOScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(CPOUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class CPODecreasingScalarizedMultiObjectivization(CPOScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        """
        weights selected considering a budget of 1.0 + 0.5 + 0.25 = 1.75, then:
            - the sum of safety weights is ~ 1.0/1.75
            - the sum of target weights is ~ 0.50/1.75
            - the sum of comfort weights is ~ 0.25/1.75
        """
        weights = np.array([1.0, 1.0, 1.0, 0.5, 0.25])
        weights /= np.sum(weights)
        super(CPODecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


#########################################################################
###              Scalarized Target vs Comfort Multi-Objective         ###
#########################################################################
class CPOScalarizedMultiObjectiveTargetVSComfort(RewardFunction):

    def __init__(self, env_params: Dict[str, Any], **kwargs):
        """ Scalarized Reward to study the tradeoff between target and comfort.
        - lambda:    the weight coefficient for the target
                    (1-lambda) is the weight coefficient for the aggregated comforts
        """
        assert "lambda" in env_params, "missing lambda in params"
        self._lambda = env_params["lambda"]

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # define norm coefficient s.t. target and comfort sum up to 1 under optimal policy
        targ_coeff = 1 / target_dist2goal_potential(info["initial_state"], info)  # norm progress w.r.t. starting x
        comfort_coeff = 1 / info["max_steps"]  # norm comfort w.r.t. nr time steps
        # compute target, comfort rewards
        target_rew = targ_coeff * (target_dist2goal_potential(next_state, info) - target_dist2goal_potential(state, info))
        comfort_rew = comfort_coeff * float(abs(state["theta"]) <= info["theta_target_tol"])
        # linear scalarization of the multi-objectivized requirements
        reward = self._lambda * target_rew + (1 - self._lambda) * comfort_rew
        return reward
