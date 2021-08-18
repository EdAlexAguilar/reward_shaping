from envs.cart_pole_obst.rewards.subtask_rewards import ContinuousCollisionReward, TaskIndicator, \
    ContinuousFalldownReward, ContinuousOutsideReward, NormalizedReward, ReachTargetReward, BalanceReward, \
    CheckOvercomingFeasibility, CollisionReward, FalldownReward, OutsideReward
from hierarchy.graph import HierarchicalGraph
from hierarchy.graph_hierarchical_reward import HierarchicalGraphRewardWrapper
import numpy as np


class GraphWithContinuousScoreBinaryIndicator(HierarchicalGraphRewardWrapper):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} sigma(r')) * rho(r)
    with sigma returns binary value {0,1}
    """

    def __init__(self, env, use_potential=False):
        labels, score_functions, indicators = [], [], []
        # define safety rules
        # collision
        fun = ContinuousCollisionReward(env)
        # note: defining the min/max robustness bound is not straightforward because depend on the obstacle position
        #       and it is randomly choosen at each episode. Then, manually define robustness bounds
        min_r, max_r = -0.5, 2.5
        labels.append("S_coll")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # falldown
        fun = ContinuousFalldownReward(theta_limit=env.theta_threshold_radians)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = fun(min_r_state), fun(max_r_state)
        labels.append("S_fall")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))
        # outside
        fun = ContinuousOutsideReward(x_limit=env.x_threshold)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = fun(min_r_state), fun(max_r_state)
        labels.append("S_exit")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define target rules
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        labels.append("T_orig")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        balance_fun = NormalizedReward(fun, min_r, max_r)
        balance_ind = TaskIndicator(fun)
        labels.append("T_bal")
        score_functions.append(balance_fun)
        indicators.append(balance_ind)

        # define conditional statement
        if env.task == "fixed_height":
            edges = [("S_coll", "T_orig"), ("S_fall", "T_orig"), ("S_exit", "T_orig"),
                     ("T_orig", "T_bal")]
        elif env.task == "random_height":
            fun = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            ind_true = TaskIndicator(CheckOvercomingFeasibility(env))
            labels.append("H_feas")
            score_functions.append(fun)
            indicators.append(ind_true)

            edges = [("S_coll", "T_bal"), ("S_fall", "T_bal"), ("S_exit", "T_bal"),
                     ("S_coll", "H_feas"), ("S_fall", "H_feas"), ("S_exit", "H_feas"),
                     ("H_feas", "T_orig")]
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")

        # define graph-based hierarchy
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithContinuousScore, self).__init__(env, hierarchy, use_potential=use_potential)


class GraphWithContinuousScoreContinuousIndicator(HierarchicalGraphRewardWrapper):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} rho(r')) * rho(r)
    """
    def __init__(self, env, use_potential=False):
        labels, score_functions, indicators = [], [], []
        # define safety rules
        # collision
        fun = ContinuousCollisionReward(env)
        # note: defining the min/max robustness bound is not straightforward because depend on the obstacle position
        #       and it is randomly choosen at each episode. Then, manually define robustness bounds
        min_r, max_r = -0.5, 2.5
        labels.append("S_coll")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(NormalizedReward(fun, min_r, max_r))

        # falldown
        fun = ContinuousFalldownReward(theta_limit=env.theta_threshold_radians)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = fun(min_r_state), fun(max_r_state)
        labels.append("S_fall")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(NormalizedReward(fun, min_r, max_r))
        # outside
        fun = ContinuousOutsideReward(x_limit=env.x_threshold)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = fun(min_r_state), fun(max_r_state)
        labels.append("S_exit")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(NormalizedReward(fun, min_r, max_r))

        # define target rules
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        labels.append("T_orig")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(NormalizedReward(fun, min_r, max_r))

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        labels.append("T_bal")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(NormalizedReward(fun, min_r, max_r))

        # define conditional statement
        if env.task == "fixed_height":
            edges = [("S_coll", "T_orig"), ("S_fall", "T_orig"), ("S_exit", "T_orig"),
                     ("T_orig", "T_bal")]
        elif env.task == "random_height":
            fun = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            ind_true = TaskIndicator(CheckOvercomingFeasibility(env))
            labels.append("H_feas")
            score_functions.append(fun)
            indicators.append(ind_true)

            edges = [("S_coll", "T_bal"), ("S_fall", "T_bal"), ("S_exit", "T_bal"),
                     ("S_coll", "H_feas"), ("S_fall", "H_feas"), ("S_exit", "H_feas"),
                     ("H_feas", "T_orig")]
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")

        # define graph-based hierarchy
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithContinuousScore, self).__init__(env, hierarchy, use_potential=use_potential)


class GraphWithContinuousTargetAndDiscreteSafety(HierarchicalGraphRewardWrapper):
    """
    s1, s2, s3
    """

    def __init__(self, env, use_potential=False):
        labels, score_functions, indicators = [], [], []
        # define safety rules
        # collision
        fun = CollisionReward(env, collision_penalty=0.0, no_collision_bonus=1.0)
        labels.append("S_coll")
        score_functions.append(fun)
        indicators.append(TaskIndicator(fun, include_zero=False))
        # falldown
        fun = FalldownReward(theta_limit=env.theta_threshold_radians, falldown_penalty=0.0, no_falldown_bonus=1.0)
        labels.append("S_fall")
        score_functions.append(fun)
        indicators.append(TaskIndicator(fun, include_zero=False))
        # outside
        fun = OutsideReward(x_limit=env.x_threshold, exit_penalty=0.0, no_exit_bonus=1.0)
        labels.append("S_exit")
        score_functions.append(fun)
        indicators.append(TaskIndicator(fun, include_zero=False))

        # define target rules
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        labels.append("T_orig")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        balance_fun = NormalizedReward(fun, min_r, max_r)
        balance_ind = TaskIndicator(fun)
        labels.append("T_bal")
        score_functions.append(balance_fun)
        indicators.append(balance_ind)

        # define conditional statement
        if env.task == "fixed_height":
            edges = [("S_coll", "T_orig"), ("S_fall", "T_orig"), ("S_exit", "T_orig"),
                     ("T_orig", "T_bal")]
        elif env.task == "random_height":
            fun = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            ind_true = TaskIndicator(CheckOvercomingFeasibility(env))
            labels.append("H_feas")
            score_functions.append(fun)
            indicators.append(ind_true)

            edges = [("S_coll", "T_bal"), ("S_fall", "T_bal"), ("S_exit", "T_bal"),
                     ("S_coll", "H_feas"), ("S_fall", "H_feas"), ("S_exit", "H_feas"),
                     ("H_feas", "T_orig")]
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")

        # define graph-based hierarchy
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithContinuousTargetAndDiscreteSafety, self).__init__(env, hierarchy, use_potential=use_potential)


class GraphWithContinuousTargetAndDiscreteSafety(HierarchicalGraphRewardWrapper):
    """
    s1, s2, s3
    """

    def __init__(self, env, use_potential=False):
        labels, score_functions, indicators = [], [], []
        # define safety rules
        # collision
        fun = CollisionReward(env, collision_penalty=0.0, no_collision_bonus=1.0)
        labels.append("S_coll")
        score_functions.append(fun)
        indicators.append(TaskIndicator(fun, include_zero=False))
        # falldown
        fun = FalldownReward(theta_limit=env.theta_threshold_radians, falldown_penalty=0.0, no_falldown_bonus=1.0)
        labels.append("S_fall")
        score_functions.append(fun)
        indicators.append(TaskIndicator(fun, include_zero=False))
        # outside
        fun = OutsideReward(x_limit=env.x_threshold, exit_penalty=0.0, no_exit_bonus=1.0)
        labels.append("S_exit")
        score_functions.append(fun)
        indicators.append(TaskIndicator(fun, include_zero=False))

        # define target rules
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        labels.append("T_orig")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        # Note: C_bal != T_bal, look at the ContReward definition for a longer explaination
        labels.append("C_bal")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))
        labels.append("T_bal")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define conditional statement
        if env.task == "fixed_height":
            edges = [("S_coll", "T_orig"), ("S_fall", "T_orig"), ("S_exit", "T_orig"),
                     ("T_orig", "C_bal")]
        elif env.task == "random_height":
            fun = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            ind_true = TaskIndicator(CheckOvercomingFeasibility(env))
            labels.append("H_feas")
            score_functions.append(fun)
            indicators.append(ind_true)

            ind_false = TaskIndicator(CheckOvercomingFeasibility(env), reverse=True)
            labels.append("H_nfeas")
            score_functions.append(fun)
            indicators.append(ind_false)

            edges = [("S_coll", "H_feas"), ("S_fall", "H_feas"), ("S_exit", "H_feas"),
                     ("S_coll", "H_nfeas"), ("S_fall", "H_nfeas"), ("S_exit", "H_nfeas"),
                     ("H_feas", "T_orig"), ("T_orig", "C_bal"),
                     ("H_nfeas", "T_bal")]
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")

        # define graph-based hierarchy
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithContinuousTargetAndDiscreteSafety, self).__init__(env, hierarchy, use_potential=use_potential)


class GraphWithTwoTargets(HierarchicalGraphRewardWrapper):
    """
    s1, s2, s3
    """

    def __init__(self, env, use_potential=False):
        labels, score_functions, indicators = [], [], []
        # define safety rules
        # collision
        fun = ContinuousCollisionReward(env)
        # note: defining the min/max robustness bound is not straightforward because depend on the obstacle position
        #       and it is randomly choosen at each episode. Then, manually define robustness bounds
        min_r, max_r = -0.5, 2.5
        labels.append("S_coll")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # falldown
        fun = ContinuousFalldownReward(theta_limit=env.theta_threshold_radians)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = fun(min_r_state), fun(max_r_state)
        labels.append("S_fall")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))
        # outside
        fun = ContinuousOutsideReward(x_limit=env.x_threshold)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = fun(min_r_state), fun(max_r_state)
        labels.append("S_exit")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define target rules
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        labels.append("T_orig")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        # Note: two nodes with the same rule, why?
        # C_bal != T_bal, because one is a comfort rule when reaching the origin is feasible
        # T_bal instead is a target requirement when the origin in not reachable.
        # For this reason we have to use 2 distinct nodes, otherwise it will be impossible to enable any of them
        # because H_feas and H_nfeas are never both satisfied.
        labels.append("C_bal")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))
        labels.append("T_bal")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define conditional statement
        if env.task == "fixed_height":
            edges = [("S_coll", "T_orig"), ("S_fall", "T_orig"), ("S_exit", "T_orig"),
                     ("T_orig", "C_bal")]
        elif env.task == "random_height":
            fun = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            ind_true = TaskIndicator(CheckOvercomingFeasibility(env))
            labels.append("H_feas")
            score_functions.append(fun)
            indicators.append(ind_true)

            edges = [("S_coll", "T_bal"), ("S_fall", "T_bal"), ("S_exit", "T_bal"),
                     ("S_coll", "H_feas"), ("S_fall", "H_feas"), ("S_exit", "H_feas"),
                     ("H_feas", "T_orig"), ("T_orig", "C_bal")]
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")

        # define graph-based hierarchy
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithTwoTargets, self).__init__(env, hierarchy, use_potential=use_potential)


class PotentialGraphWithContinuousScore(GraphWithContinuousScore):
    def __init__(self, env):
        super(PotentialGraphWithContinuousScore, self).__init__(env, use_potential=True)


class PotentialGraphWithContinuousTargetAndDiscreteSafety(GraphWithContinuousTargetAndDiscreteSafety):
    def __init__(self, env):
        super(PotentialGraphWithContinuousTargetAndDiscreteSafety, self).__init__(env, use_potential=True)
