from envs.cart_pole.rewards.subtask_rewards import TaskIndicator, \
    ContinuousFalldownReward, ContinuousOutsideReward, NormalizedReward, ReachTargetReward, BalanceReward, \
    CollisionReward, FalldownReward, OutsideReward, ContinuousCollisionReward
from hierarchy.graph import HierarchicalGraph
from hierarchy.graph_hierarchical_reward import HierarchicalGraphRewardWrapper
import numpy as np


class GraphWithContinuousScore(HierarchicalGraphRewardWrapper):
    """
    s1, s2, s3
    """

    def __init__(self, env, use_potential=False):
        labels, score_functions, indicators = [], [], []
        # define safety rules
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
        target_score = NormalizedReward(fun, min_r, max_r)
        target_indicator = TaskIndicator(fun)

        # define comfort rules
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        balance_score = NormalizedReward(fun, min_r, max_r)
        balance_indicator = TaskIndicator(fun)
        labels.append("T_bal")
        score_functions.append(balance_score)
        indicators.append(balance_indicator)

        if env.task == "balance":
            edges = [("S_fall", "T_bal"), ("S_exit", "T_bal")]
        elif env.task == "target":
            # target rule
            labels.append("T_orig")
            score_functions.append(target_score)
            indicators.append(target_indicator)
            edges = [("S_fall", "T_orig"), ("S_exit", "T_orig"),
                     ("S_fall", "T_bal"), ("S_exit", "T_bal")]
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")

        # define graph-based hierarchy
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithContinuousScore, self).__init__(env, hierarchy, use_potential=use_potential)


class PotentialGraphWithContinuousScore(GraphWithContinuousScore):
    def __init__(self, env):
        super(PotentialGraphWithContinuousScore, self).__init__(env, use_potential=True)
