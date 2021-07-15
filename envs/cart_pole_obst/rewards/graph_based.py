from envs.cart_pole_obst.rewards.subtask_rewards import ContinuousCollisionReward, TaskIndicator, \
    ContinuousFalldownReward, ContinuousOutsideReward, NormalizedReward, ReachTargetReward, BalanceReward, \
    CheckOvercomingFeasibility
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
        # collision
        fun = ContinuousCollisionReward(env)
        # note: defining the min/max robustness bound is not straightforward because depend on the obstacle position
        #       and it is randomly choosen at each episode. Then, manually define robustness bounds
        min_r, max_r = -0.5, 2.5
        assert max_r>min_r
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

        # define conditional statement
        fun = lambda _: 0.0     # this is a static condition, do not score for it (depends on the env)
        ind_true = TaskIndicator(CheckOvercomingFeasibility(env))
        labels.append("IF_feas")
        score_functions.append(fun)
        indicators.append(ind_true)

        ind_false = TaskIndicator(CheckOvercomingFeasibility(env), reverse=True)
        labels.append("IF_nfeas")
        score_functions.append(fun)
        indicators.append(ind_false)

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
        labels.append("T_bal")
        score_functions.append(NormalizedReward(fun, min_r, max_r))
        indicators.append(TaskIndicator(fun))

        # define graph-based hierarchy
        edges = [("S_coll", "IF_feas"), ("S_fall", "IF_feas"), ("S_exit", "IF_feas"),
                 ("S_coll", "IF_nfeas"), ("S_fall", "IF_nfeas"), ("S_exit", "IF_nfeas"),
                 ("IF_feas", "T_orig"),
                 ("IF_nfeas", "T_bal")]
        hierarchy = HierarchicalGraph(labels, score_functions, indicators, edges)
        super(GraphWithContinuousScore, self).__init__(env, hierarchy, use_potential=use_potential)


class PotentialGraphWithContinuousScore(GraphWithContinuousScore):
    def __init__(self, env):
        super(PotentialGraphWithContinuousScore, self).__init__(env, use_potential=True)
