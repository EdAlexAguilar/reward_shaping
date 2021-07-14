# hierarchy with 4 rules: T={reach origin}, S={no collision, no falldown}, C={keep balance}
# SAFETY 1: no collision
from abc import ABC, abstractmethod

import gym


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


class FalldownReward(TaskReward):
    def __init__(self, theta_limit, falldown_penalty=-10):
        super().__init__()
        self.theta_limit = theta_limit
        self.falldown_penalty = falldown_penalty

    def __call__(self, state):
        theta = state[2]
        return self.falldown_penalty * (abs(theta) > self.theta_limit)


class OutsideReward(TaskReward):
    def __init__(self, x_limit, exit_penalty=-10):
        super().__init__()
        self.x_limit = x_limit
        self.exit_penalty = exit_penalty

    def __call__(self, state):
        x = state[0]
        return self.exit_penalty * (abs(x) > self.x_limit)


class ReachTargetReward(TaskReward):
    def __init__(self, x_target, x_target_tol):
        super().__init__()
        self.x_target = x_target
        self.x_target_tol = x_target_tol
        # define lambda fun for later reuse
        self.reward_fun = lambda x: self.x_target_tol - abs(x - self.x_target)

    def __call__(self, state):
        x = state[0]
        return self.reward_fun(x)


class SparseReachTargetReward(ReachTargetReward):
    def __init__(self, x_target, x_target_tol, target_reward=5.0):
        super().__init__(x_target, x_target_tol)
        self.target_reward = target_reward

    def __call__(self, state):
        reward = super(SparseReachTargetReward, self).__call__(state)
        return self.target_reward if reward >= 0.0 else 0.0


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


class NormalizedReward(TaskReward):
    def __init__(self, reward, min_reward, max_reward):
        super().__init__()
        self.reward = reward
        self.min_reward = min_reward
        self.max_reward = max_reward

    def __call__(self, reward):
        reward = self.reward(reward)
        return (reward - self.min_reward) / (self.max_reward - self.min_reward)


class TaskIndicator(TaskReward):
    def __init__(self, reward_fun):
        self.reward_fun = reward_fun

    def __call__(self, state):
        reward = self.reward_fun(state)
        return reward >= 0.0
