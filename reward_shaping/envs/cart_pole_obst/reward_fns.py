from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from reward_shaping.envs.core import RewardFunction


class CollisionReward(RewardFunction):
    def __init__(self, collision_penalty=0.0, no_collision_bonus=0.0):
        super().__init__()
        self.collision_penalty = collision_penalty
        self.no_collision_bonus = no_collision_bonus

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'collision' in state
        collision = state['collision'] == 1
        return self.no_collision_bonus if not collision else self.collision_penalty

class FalldownReward(RewardFunction):
    def __init__(self, theta_limit, falldown_penalty=0.0, no_falldown_bonus=0.0):
        super().__init__()
        self.theta_limit = theta_limit
        self.falldown_penalty = falldown_penalty
        self.no_falldown_bonus = no_falldown_bonus

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'theta' in state
        theta = state['theta']
        return self.falldown_penalty if (abs(theta) > self.theta_limit) else self.no_falldown_bonus


class ContinuousFalldownReward(RewardFunction):
    """
    psi := abs(theta) <= theta_limit
    rho := theta_limit - abs(theta)
    """

    def __init__(self, theta_limit):
        super().__init__()
        self.theta_limit = theta_limit

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'theta' in state
        theta = state['theta']
        return self.theta_limit - abs(theta)


class OutsideReward(RewardFunction):
    def __init__(self, x_limit, exit_penalty=0.0, no_exit_bonus=0.0):
        super().__init__()
        self.x_limit = x_limit
        self.exit_penalty = exit_penalty
        self.no_exit_bonus = no_exit_bonus

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'x' in state
        x = state['x']
        return self.exit_penalty if (abs(x) > self.x_limit) else self.no_exit_bonus


class ContinuousOutsideReward(RewardFunction):
    """
    psi := abs(x) <= x_limit
    rho := x_limit - abs(x)
    """
    def __init__(self, x_limit):
        super().__init__()
        self.x_limit = x_limit

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'x' in state
        x = state['x']
        return self.x_limit - abs(x)


class ReachTargetReward(RewardFunction):
    def __init__(self, x_target, x_target_tol):
        self.x_target = x_target
        self.x_target_tol = x_target_tol

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'x' in state
        x = state['x']
        return self.x_target_tol - abs(x - self.x_target)


class SparseReachTargetReward(RewardFunction):
    def __init__(self, x_target, x_target_tol, target_reward=5.0):
        self.target_reward = target_reward
        self.x_target = x_target
        self.x_target_tol = x_target_tol

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'x' in state
        x = state['x']
        rho = self.x_target_tol - abs(x - self.x_target)
        return self.target_reward if rho >= 0.0 else 0.0


class ProgressReachTargetReward(RewardFunction):
    def __init__(self, x_target, x_target_tol, progress_coeff=1.0):
        self.x_target = x_target
        self.x_target_tol = x_target_tol
        self.progress_coeff = progress_coeff
        self._previous_distance = None

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'x' in state and 'x' in next_state
        x = state['x']
        next_x = next_state['x']
        distance = self.x_target_tol - abs(x - self.x_target)
        next_distance = self.x_target_tol - abs(next_x - self.x_target)
        return self.progress_coeff * (distance - next_distance)


class BalanceReward(RewardFunction):
    def __init__(self, theta_target, theta_target_tol):
        self.theta_target = theta_target
        self.theta_target_tol = theta_target_tol

    def __call__(self, state, action=None, next_state=None) -> float:
        assert 'theta' in state
        theta = state['theta']
        return self.theta_target_tol - abs(theta - self.theta_target)

class CheckOvercomingFeasibility(RewardFunction):
    """
    psi := obst_bottom_y - axle_y >= feasibility_height
    rho := obst_bottom_y - axle_y - feasibility_height
    """
    def __init__(self, obstacle_y: float, axle_y: float, feasible_height: float):
        self.obstacle_y = obstacle_y
        self.axle_y = axle_y
        self.feasible_height = feasible_height

    def __call__(self, state, action=None, next_state=None) -> float:
        return self.obstacle_y - self.axle_y - self.feasible_height

class Indicator(RewardFunction):
    def __init__(self, reward_fn: RewardFunction, negate=False):
        self.negate = negate
        self.reward_fn = reward_fn

    def __call__(self, state, action=None, next_state=None) -> float:
        if self.negate:
            return self.reward_fn(state, action, next_state) < 0
        else:
            return self.reward_fn(state, action, next_state) > 0


