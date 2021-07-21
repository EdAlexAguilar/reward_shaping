from abc import ABC, abstractmethod
import numpy as np


class TaskReward(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state):
        pass


class CollisionReward(TaskReward):
    def __init__(self, env, collision_penalty=0.0, no_collision_bonus=0.0):
        super().__init__()
        self.env = env  # we need it to call the `intersect` method
        self.collision_penalty = collision_penalty
        self.no_collision_bonus = no_collision_bonus

    def __call__(self, state):
        x, theta = state[0], state[2]
        collision = self.env.obstacle.intersect(x, theta)
        return self.no_collision_bonus if not collision else self.collision_penalty


class ContinuousCollisionReward(TaskReward):
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

    def __init__(self, env):
        super().__init__()
        self.env = env  # we need it to access the obstacle coordinates

    def __call__(self, state):
        x, theta = state[0], state[2]
        obs = self.env.obstacle
        pole_x = x + np.sin(theta) * self.env.pole_length
        pole_y = self.env.axle_y + np.cos(theta) * self.env.pole_length
        rho = max(-(pole_x - obs.left_x), -(obs.right_x - pole_x), -(pole_y - obs.bottom_y), -(obs.top_y - pole_y))
        return rho


class FalldownReward(TaskReward):
    def __init__(self, theta_limit, falldown_penalty=0.0, no_falldown_bonus=0.0):
        super().__init__()
        self.theta_limit = theta_limit
        self.falldown_penalty = falldown_penalty
        self.no_falldown_bonus = no_falldown_bonus

    def __call__(self, state):
        theta = state[2]
        return self.falldown_penalty if (abs(theta) > self.theta_limit) else self.no_falldown_bonus


class ContinuousFalldownReward(TaskReward):
    """
    psi := abs(theta) <= theta_limit
    rho := theta_limit - abs(theta)
    """

    def __init__(self, theta_limit):
        super().__init__()
        self.theta_limit = theta_limit

    def __call__(self, state):
        theta = state[2]
        return self.theta_limit - abs(theta)


class OutsideReward(TaskReward):
    def __init__(self, x_limit, exit_penalty=0.0, no_exit_bonus=0.0):
        super().__init__()
        self.x_limit = x_limit
        self.exit_penalty = exit_penalty
        self.no_exit_bonus = no_exit_bonus

    def __call__(self, state):
        x = state[0]
        return self.exit_penalty if (abs(x) > self.x_limit) else self.no_exit_bonus


class ContinuousOutsideReward(TaskReward):
    """
    psi := abs(x) <= x_limit
    rho := x_limit - abs(x)
    """
    def __init__(self, x_limit):
        super().__init__()
        self.x_limit = x_limit

    def __call__(self, state):
        x = state[0]
        return self.x_limit - abs(x)


class ReachTargetReward(TaskReward):
    def __init__(self, x_target, x_target_tol):
        super().__init__()
        self.x_target = x_target
        self.x_target_tol = x_target_tol

    def __call__(self, state):
        x = state[0]
        return self.x_target_tol - abs(x - self.x_target)


class SparseReachTargetReward(ReachTargetReward):
    def __init__(self, x_target, x_target_tol, target_reward=5.0):
        super().__init__(x_target, x_target_tol)
        self.target_reward = target_reward

    def __call__(self, state):
        rho = super(SparseReachTargetReward, self).__call__(state)
        return self.target_reward if rho >= 0.0 else 0.0


class ProgressReachTargetReward(ReachTargetReward):
    def __init__(self, env, x_target, x_target_tol, progress_coeff=1.0):
        super().__init__(x_target, x_target_tol)
        self.env = env
        self.progress_coeff = progress_coeff

    def __call__(self, state):
        reward = super(ProgressReachTargetReward, self).__call__(state)
        if self.env.last_state is not None:
            last_reward = super(ProgressReachTargetReward, self).__call__(self.env.last_state)
        else:
            raise Exception("last_state not found, probably env not reset")
        return self.progress_coeff * (reward - last_reward)


class BalanceReward(TaskReward):
    def __init__(self, theta_target, theta_target_tol):
        self.theta_target = theta_target
        self.theta_target_tol = theta_target_tol

    def __call__(self, state):
        theta = state[2]
        return self.theta_target_tol - abs(theta - self.theta_target)


class CheckOvercomingFeasibility(TaskReward):
    """
    psi := obst_bottom_y - axle_y >= feasibility_height
    rho := obst_bottom_y - axle_y - feasibility_height
    """
    def __init__(self, env):
        self.env = env

    def __call__(self, state):
        return self.env.obstacle.bottom_y - self.env.axle_y - self.env.feasible_height


class NormalizedReward(TaskReward):
    def __init__(self, reward, min_reward, max_reward):
        assert max_reward > min_reward, f"unvalid normalization: min: {min_reward} >= max: {max_reward}"
        super().__init__()
        self.reward = reward
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(self, reward):
        reward = self.reward(reward)
        reward = np.clip(reward, self.min_reward, self.max_reward)
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)


class TaskIndicator(TaskReward):
    def __init__(self, reward_fun, reverse=False, include_zero=True):
        self.reward_fun = reward_fun
        self.reverse = reverse
        self.include_zero=include_zero

    def __call__(self, state):
        # (default) if `reverse` is False, then indicator returns True when reward >= 0.0
        # if `reverse` is True, then indicator returns True when reward < 0.0
        reward = self.reward_fun(state)
        sat = reward >= 0.0 if self.include_zero else reward > 0.0
        result = sat if not self.reverse else not sat
        return result
