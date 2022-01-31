from typing import Union, List

from reward_shaping.core.reward import RewardFunction
import numpy as np

from reward_shaping.envs.cart_pole_obst.specs import get_all_specs


def clip_and_norm(v: Union[int, float], minv: Union[int, float], maxv: Union[int, float]):
    """
    utility function which returns the normalized value v' in [0, 1].

    @params: value `v` before normalization,
    @params: `minv`, `maxv` extreme values of the domain.
    """
    return (np.clip(v, minv, maxv) - minv) / (maxv - minv)


def simple_base_reward(state, info):
    """
    sparse reward which returns +1 when the target configuration has been reached,
    otherwise 0.
    """
    assert all([s in state for s in ["x", "theta"]])
    assert all([i in info for i in ["x_target", "x_target_tol", "theta_target_tol"]])
    check_goal = abs(state['x'] - info['x_target']) <= info['x_target_tol'] and \
                 abs(state['theta']) <= info["theta_target_tol"]
    return 1.0 if check_goal else 0.0


def safety_falldown_potential(state, info):
    assert "theta" in state and "theta_limit" in info
    falldown = (state["theta"] <= info["theta_limit"])
    return int(falldown)


def safety_exit_potential(state, info):
    assert "x" in state and "x_limit" in info
    outside = (state["x"] <= info["x_limit"])
    return int(outside)


def safety_collision_potential(state, info):
    assert "collision" in state
    collision = (state["collision"] <= 0)
    return int(collision)


def target_dist_to_goal_potential(state, info):
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


class CPOHierarchicalPotentialShaping(RewardFunction):

    def _safety_potential(self, state, info):
        falldown = safety_falldown_potential(state, info)
        exit = safety_exit_potential(state, info)
        collision = safety_collision_potential(state, info)
        return int(falldown) + int(exit) + int(collision)

    def _target_potential(self, state, info):
        """
        idea: since the task is to conquer the origin, the potential of a state depends on two factors:
            - the distance to the target (if not reached yet), and the persistence on the target (once reached)
        """
        target_reward = target_dist_to_goal_potential(state, info)
        safety_w = self._safety_potential(state, info)
        return safety_w * target_reward

    def _comfort_potential(self, state, info):
        comfort_reward = comfort_balance_potential(state, info)
        safety_w, target_w = self._safety_potential(state, info), self._target_potential(state, info)
        return safety_w * target_w * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # hierarchical shaping function
        shaping_safety = self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort


class CPOScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert sum(weights) == 1.0, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_falldown = safety_falldown_potential(next_state, info) - safety_falldown_potential(state, info)
        shaping_exit = safety_exit_potential(next_state, info) - safety_exit_potential(state, info)
        shaping_coll = safety_collision_potential(next_state, info) - safety_collision_potential(state, info)
        shaping_target = target_dist_to_goal_potential(next_state, info) - target_dist_to_goal_potential(state, info)
        shaping_comfort = comfort_balance_potential(next_state, info) - comfort_balance_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights, [shaping_falldown, shaping_exit, shaping_coll, shaping_target, shaping_comfort]):
            reward += w * f
        return reward


class CPOUniformScalarizedMultiObjectivization(CPOScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        super(CPOUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class CPODecreasingScalarizedMultiObjectivization(CPOScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = [0.25, 0.25, 0.25, 0.15, 0.10]
        super(CPODecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)
