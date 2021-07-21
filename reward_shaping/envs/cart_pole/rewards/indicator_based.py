import numpy as np

from reward_shaping.hierarchy import HierarchicalRewardWrapper
from reward_shaping.envs import CollisionReward, FalldownReward, OutsideReward, \
    SparseReachTargetReward, ProgressReachTargetReward
from reward_shaping.envs import NormalizedReward, TaskIndicator, ReachTargetReward, BalanceReward


class IndicatorWithContinuousTargetReward(HierarchicalRewardWrapper):
    def __init__(self, env, clip_to_positive=False, unit_scaling=False):
        # define safety rules
        safety_penalty = -10
        safety_functions = []
        # collision
        fun = CollisionReward(env, collision_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)
        # falldown
        fun = FalldownReward(theta_limit=env.theta_threshold_radians, falldown_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)
        # outside
        fun = OutsideReward(x_limit=env.x_threshold, exit_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)

        # define target rules
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        fun_ind_pair = NormalizedReward(fun, min_r, max_r), TaskIndicator(fun)
        target_functions = [fun_ind_pair]

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        fun_ind_pair = NormalizedReward(fun, min_r, max_r), TaskIndicator(fun)
        comfort_functions = [fun_ind_pair]

        hierarchy = {'safety': safety_functions, 'target': target_functions, 'comfort': comfort_functions}
        super(IndicatorWithContinuousTargetReward, self).__init__(env, hierarchy, clip_to_positive, unit_scaling)


class IndicatorWithSparseTargetReward(HierarchicalRewardWrapper):
    def __init__(self, env, clip_to_positive=False, unit_scaling=False):
        # define safety rules
        safety_penalty = -10
        safety_functions = []
        # collision
        fun = CollisionReward(env, collision_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)
        # falldown
        fun = FalldownReward(theta_limit=env.theta_threshold_radians, falldown_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)
        # outside
        fun = OutsideReward(x_limit=env.x_threshold, exit_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)

        # define target rules
        fun = SparseReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        fun_ind_pair = NormalizedReward(fun, min_r, max_r), TaskIndicator(fun)
        target_functions = [fun_ind_pair]

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        fun_ind_pair = NormalizedReward(fun, min_r, max_r), TaskIndicator(fun)
        comfort_functions = [fun_ind_pair]

        hierarchy = {'safety': safety_functions, 'target': target_functions, 'comfort': comfort_functions}
        super(IndicatorWithSparseTargetReward, self).__init__(env, hierarchy, clip_to_positive, unit_scaling)


class IndicatorWithProgressTargetReward(HierarchicalRewardWrapper):
    def __init__(self, env, clip_to_positive=False, unit_scaling=False):
        # define safety rules
        safety_penalty = -10
        safety_functions = []
        # collision
        fun = CollisionReward(env, collision_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)
        # falldown
        fun = FalldownReward(theta_limit=env.theta_threshold_radians, falldown_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)
        # outside
        fun = OutsideReward(x_limit=env.x_threshold, exit_penalty=safety_penalty)
        fun_ind_pair = fun, TaskIndicator(fun)
        safety_functions.append(fun_ind_pair)

        # define target rules
        fun = ProgressReachTargetReward(env=env, x_target=env.x_target, x_target_tol=env.x_target_tol,
                                        progress_coeff=10.0)
        fun_ind_pair = fun, TaskIndicator(fun)
        target_functions = [fun_ind_pair]

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        fun_ind_pair = NormalizedReward(fun, min_r, max_r), TaskIndicator(fun)
        comfort_functions = [fun_ind_pair]

        hierarchy = {'safety': safety_functions, 'target': target_functions, 'comfort': comfort_functions}
        super(IndicatorWithProgressTargetReward, self).__init__(env, hierarchy, clip_to_positive, unit_scaling)