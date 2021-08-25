from abc import ABC, abstractmethod
import numpy as np

from reward_shaping.core.reward import RewardFunction


class CollisionReward(RewardFunction):
    def __init__(self, collision_penalty=0.0, no_collision_bonus=0.0):
        super().__init__()
        self.collision_penalty = collision_penalty
        self.no_collision_bonus = no_collision_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'collision' in state
        collision = state['collision'] == 1
        return self.no_collision_bonus if not collision else self.collision_penalty


class ContinuousCollisionReward(RewardFunction):
    """
    psi := not((obs_left_x <= pole_x and pole_x <= obs_right_x) and (obs_bottom_y <= pole_y and pole_y <= obs_top_y))

    rewriting
    psi := not(obs_left_x <= pole_x and pole_x <= obs_right_x) or not(obs_bottom_y <= pole_y and pole_y <= obs_top_y)
        := not(obs_left_x <= pole_x) or not(pole_x <= obs_right_x)
           or not(obs_bottom_y <= pole_y) or not(pole_y <= obs_top_y)
        := not(pole_x - obs_left_x >= 0) or not(obs_right_x - pole_x >= 0)
           or not(pole_y - obs_bottom_y >= 0) or not(obs_top_y - pole_y >= 0)
    rho(psi, state) := max(-(pole_x-obs_left_x), -(obs_right_x-pole_x), -(pole_y-obs_bottom_y), -(obs_top_y-pole_y))
    """

    def __call__(self, state, action, next_state, info):
        assert 'x' in state and 'theta' in state
        assert 'axle_y' in info and 'pole_length' in info
        assert 'obstacle_left' in state and 'obstacle_right' in state
        assert 'obstacle_bottom' in state and 'obstacle_top' in state
        x, theta = state['x'], state['theta']
        pole_x = x + np.sin(theta) * info['pole_length']
        pole_y = info['axle_y'] + np.cos(theta) * info['pole_length']
        rho = max(-(pole_x - state['obstacle_left']), -(state['obstacle_right'] - pole_x),
                  -(pole_y - state['obstacle_bottom']), -(state['obstacle_top'] - pole_y))
        return rho


class FalldownReward(RewardFunction):
    def __init__(self, theta_limit, falldown_penalty=0.0, no_falldown_bonus=0.0):
        super().__init__()
        self.theta_limit = theta_limit
        self.falldown_penalty = falldown_penalty
        self.no_falldown_bonus = no_falldown_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'theta' in state
        theta = state['theta']
        return self.falldown_penalty if (abs(theta) > self.theta_limit) else self.no_falldown_bonus


class ContinuousFalldownReward(RewardFunction):
    """
    psi := abs(theta) <= theta_limit
    rho := theta_limit - abs(theta)
    """
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'theta' in state and 'theta_limit' in info
        theta = state['theta']
        return info['theta_limit'] - abs(theta)


class OutsideReward(RewardFunction):
    def __init__(self, exit_penalty=0.0, no_exit_bonus=0.0):
        super().__init__()
        self.exit_penalty = exit_penalty
        self.no_exit_bonus = no_exit_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in state and 'x_limit' in info
        x = state['x']
        return self.exit_penalty if (abs(x) > info['x_limit']) else self.no_exit_bonus


class ContinuousOutsideReward(RewardFunction):
    """
    psi := abs(x) <= x_limit
    rho := x_limit - abs(x)
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in state and 'x_limit' in info
        return info['x_limit'] - abs(state['x'])


class ReachTargetReward(RewardFunction):
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in state and 'x_target' in info and 'x_target_tol' in info
        x = state['x']
        return info['x_target_tol'] - abs(x - info['x_target'])


class ProgressToTargetReward(RewardFunction):
    def __init__(self, progress_coeff=1.0):
        self._progress_coeff=progress_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in state and 'x_target' in info and 'x_target_tol' in info
        if next_state is not None:
            closeness = 1 - (abs(state['x'] - info['x_target']) / abs(info['x_limit'] - info['x_target']))
            next_closeness = 1 - (abs(next_state['x'] - info['x_target']) / abs(info['x_limit'] - info['x_target']))
            return self._progress_coeff * (next_closeness - closeness)
        else:
            return info['x_target_tol'] - abs(state['x'] - info['x_target'])



class SparseReachTargetReward(RewardFunction):
    def __init__(self, target_reward=5.0):
        self.target_reward = target_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in state and 'x_target' in info and 'x_target_tol' in info
        x = state['x']
        rho = info['x_target_tol'] - abs(x - info['x_target'])
        return self.target_reward if rho >= 0.0 else 0.0


class ProgressReachTargetReward(RewardFunction):
    def __init__(self, progress_coeff=1.0):
        self.progress_coeff = progress_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in state and 'x' in next_state
        assert 'x_target' in info and 'x_target_tol' in info
        x = state['x']
        next_x = next_state['x']
        distance = info['x_target_tol'] - abs(x - info['x_target'])
        next_distance = info['x_target_tol'] - abs(next_x - info['x_target'])
        return self.progress_coeff * (distance - next_distance)


class BalanceReward(RewardFunction):

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'theta' in state
        theta = state['theta']
        return info['theta_target_tol'] - abs(theta - info['theta_target'])


class CheckOvercomingFeasibility(RewardFunction):
    """
    psi := obst_bottom_y - axle_y >= feasibility_height
    rho := obst_bottom_y - axle_y - feasibility_height
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'obstacle_bottom' in state
        assert 'axle_y' in info and 'feasible_height' in info
        return state['obstacle_bottom'] - info['axle_y'] - info['feasible_height']
