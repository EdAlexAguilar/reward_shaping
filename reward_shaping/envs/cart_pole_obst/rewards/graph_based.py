import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
from reward_shaping.core.configs import GraphRewardConfig
from reward_shaping.core.helper_fns import ThresholdIndicator, MinAggregatorReward, \
    ProdAggregatorReward
import numpy as np

from reward_shaping.core.utils import get_normalized_reward

PROGCOEFF = 1.0


def get_cartpole_topology(task):
    # just to avoid to rewrite it all the times
    if task == "fixed_height":
        topology = {
            'S_coll': ['T_origin'],
            'S_fall': ['T_origin'],
            'S_exit': ['T_origin'],
            'T_origin': ['T_bal'],
        }
    elif task == "random_height":
        topology = {
            'S_coll': ['H_feas', 'H_nfeas'],
            'S_fall': ['H_feas', 'H_nfeas'],
            'S_exit': ['H_feas', 'H_nfeas'],
            'H_feas': ['T_origin'],
            'H_nfeas': ['T_bal'],
            'T_origin': ['C_bal'],
        }
    else:
        raise NotImplemented(f"no reward-topology for task {task}")
    return topology


class CPOGraphWithContinuousScoreBinaryIndicator(GraphRewardConfig):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} sigma(r')) * rho(r)
    with sigma returns binary value {0,1}
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision
        fun = fns.ContinuousCollisionReward()
        # note: defining the min/max robustness bounds depend on the obstacle position (not known a priori)
        #       Then, the reward is normalized with approx. bounds
        min_r, max_r = -0.5, 2.5
        nodes["S_coll"] = get_normalized_reward(fun, min_r, max_r, info=info)

        # falldown
        fun = fns.ContinuousFalldownReward()
        min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': 0.0}
        nodes["S_fall"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)

        # outside
        fun = fns.ContinuousOutsideReward()
        min_r_state, max_r_state = {'x': info['x_limit']}, {'x': 0.0}
        nodes["S_exit"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)

        # define target rules
        fun = fns.ReachTargetReward()
        min_r_state, max_r_state = {'x': info['x_limit']}, {'x': info['x_target']}
        nodes["T_origin"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)

        # define comfort rules
        fun = fns.BalanceReward()
        min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': info['theta_target']}
        nodes["T_bal"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)

        if self._env_params['task'] == "random_height":
            # for random env, additional comfort node
            fun = fns.BalanceReward()
            min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': info['theta_target']}
            nodes["C_bal"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)
            # conditional nodes (ie, to check env conditions)
            zero_fn = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            feas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nfeas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility(), negate=True)
            nodes["H_feas"] = (zero_fn, feas_ind)
            nodes["H_nfeas"] = (zero_fn, nfeas_ind)
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphWithContinuousScoreContinuousIndicator(GraphRewardConfig):
    """
        rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} rho(r')) * rho(r)
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision, note: defining the min/max robustness bounds depend on the obstacle position (not known a priori)
        # Then, the reward is normalized with approx. bounds
        fn_coll, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.5, max_r=2.5)
        nodes["S_coll"] = (fn_coll, fn_coll)

        # falldown
        fn_fall, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                           min_r_state={'theta': info['theta_limit']},
                                           max_r_state={'theta': 0.0}, info=info)
        nodes["S_fall"] = (fn_fall, fn_fall)

        # outside
        fn_out, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                          min_r_state={'x': info['x_limit']},
                                          max_r_state={'x': 0.0}, info=info)
        nodes["S_exit"] = (fn_out, fn_out)

        # define target rules
        fn_target, _ = get_normalized_reward(fns.ReachTargetReward(),
                                             min_r_state={'x': info['x_limit']},
                                             max_r_state={'x': info['x_target']}, info=info)
        nodes["T_origin"] = (fn_target, fn_target)

        # define comfort rules
        fn_balance, _ = get_normalized_reward(fns.BalanceReward(),
                                              min_r_state={'theta': info['theta_limit']},
                                              max_r_state={'theta': info['theta_target']}, info=info)
        nodes["T_bal"] = (fn_balance, fn_balance)

        if self._env_params['task'] == "random_height":
            # for random env, additional comfort node
            nodes["C_bal"] = (fn_balance, fn_balance)
            # conditional nodes (ie, to check env conditions)
            zero_fn = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            feas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nfeas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility(), negate=True)
            nodes["H_feas"] = (zero_fn, feas_ind)
            nodes["H_nfeas"] = (zero_fn, nfeas_ind)
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphWithProgressScoreBinaryIndicator(GraphRewardConfig):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} sigma(r')) * rho(r)
    with sigma returns binary value {0,1}
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision
        fun = fns.CollisionReward(collision_penalty=-1.0, no_collision_bonus=0.0)
        nodes["S_coll"] = (fun, ThresholdIndicator(fun))

        # falldown
        fun = fns.FalldownReward(falldown_penalty=-1.0, no_falldown_bonus=0.0)
        nodes["S_fall"] = (fun, ThresholdIndicator(fun))

        # outside
        fun = fns.OutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0)
        nodes["S_exit"] = (fun, ThresholdIndicator(fun))

        # define target rules
        fun = fns.ProgressToTargetReward(progress_coeff=PROGCOEFF)
        nodes["T_origin"] = (fun, ThresholdIndicator(fun, include_zero=False))

        # define comfort rules
        fun = fns.BalanceReward()
        min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': info['theta_target']}
        nodes["T_bal"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)

        if self._env_params['task'] == "random_height":
            # for random env, additional comfort node
            fun = fns.BalanceReward()
            min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': info['theta_target']}
            nodes["C_bal"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)
            # conditional nodes (ie, to check env conditions)
            zero_fn = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            feas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nfeas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility(), negate=True)
            nodes["H_feas"] = (zero_fn, feas_ind)
            nodes["H_nfeas"] = (zero_fn, nfeas_ind)
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphWithBinarySafetyScoreBinaryIndicator(GraphRewardConfig):
    """
    the safety properties return -1 (violation) or 0 (sat)
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision
        fun = fns.CollisionReward(collision_penalty=-1.0, no_collision_bonus=0.0)
        nodes["S_coll"] = (fun, ThresholdIndicator(fun))

        # falldown
        fun = fns.FalldownReward(falldown_penalty=-1.0, no_falldown_bonus=0.0)
        nodes["S_fall"] = (fun, ThresholdIndicator(fun))

        # outside
        fun = fns.OutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0)
        nodes["S_exit"] = (fun, ThresholdIndicator(fun))

        # define target rules
        fun = fns.ReachTargetReward()
        min_r_state, max_r_state = {'x': info['x_limit']}, {'x': info['x_target']}
        nodes["T_origin"] = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state, info=info)

        # define comfort rules
        cont_balance_fn = fns.BalanceReward()
        nodes["T_bal"] = get_normalized_reward(cont_balance_fn, min_r_state={'theta': info['theta_limit']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)

        if self._env_params['task'] == "random_height":
            # for random env, additional comfort node
            cont_balance_fn = fns.BalanceReward()
            min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': info['theta_target']}
            nodes["C_bal"] = get_normalized_reward(cont_balance_fn, min_r_state={'theta': info['theta_limit']},
                                                   max_r_state={'theta': info['theta_target']},
                                                   info=info)
            # conditional nodes (ie, to check env conditions)
            zero_fn = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            feas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nfeas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility(), negate=True)
            nodes["H_feas"] = (zero_fn, feas_ind)
            nodes["H_nfeas"] = (zero_fn, nfeas_ind)
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphBinarySafetyProgressTargetContinuousIndicator(GraphRewardConfig):
    """
        rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} rho(r')) * custom_rho(r)

        where:
            - rho(.) is the robustness evaluation (ie, normalized degree of satisfaction)
            - custom_rho(.) is our semantics which is binary for safety, progress-based for target
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision
        binary_coll_fn = fns.CollisionReward(collision_penalty=-1.0, no_collision_bonus=0.0)
        # note: defining the min/max robustness bounds depend on the obstacle position (not known a priori)
        #       We approx. normalize the reward assuming range in +-0.5
        cont_coll_fn, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0)
        nodes["S_coll"] = (binary_coll_fn, cont_coll_fn)

        # falldown
        binary_fall_fn = fns.FalldownReward(falldown_penalty=-1.0, no_falldown_bonus=0.0)
        cont_fall_fn, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                                min_r_state={'theta': info['theta_limit']},
                                                max_r_state={'theta': 0.0},
                                                info=info)
        nodes["S_fall"] = (binary_fall_fn, cont_fall_fn)

        # outside
        binary_exit_fn = fns.OutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0)
        cont_exit_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                min_r_state={'x': info['x_limit']},
                                                max_r_state={'x': 0.0},
                                                info=info)
        nodes["S_exit"] = (binary_exit_fn, cont_exit_fn)

        # define target rules
        # define target rules
        # note: progress is computed as progress/time, bound it to have approx same scale
        progress_fn, _ = get_normalized_reward(fns.ProgressToTargetReward(progress_coeff=PROGCOEFF),
                                               min_r=-1.0, max_r=1.0)
        target_fun, _ = get_normalized_reward(fns.ReachTargetReward(),
                                              min_r_state={'x': info['x_limit']},
                                              max_r_state={'x': info['x_target'] - info['x_target_tol']},
                                              info=info)
        nodes["T_origin"] = (progress_fn, target_fun)

        # define comfort rules
        # balance: theta_tol - |theta - theta_target|, rob range: [-1.25, 0.4] (considering all theta domain +-pi/2)
        # Q: better to normalize over the rob of theta domain, or the rob of the comfort domain?
        nodes["T_bal"] = get_normalized_reward(fns.BalanceReward(),
                                               min_r_state={'theta': info['theta_target'] - info['theta_target_tol']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphContinuousSafetyProgressTargetContinuousIndicator(GraphRewardConfig):
    """
        rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} rho(r')) * custom_rho(r)

        where:
            - rho(.) is the robustness evaluation (ie, normalized degree of satisfaction)
            - custom_rho(.) is our semantics which is binary for safety, progress-based for target
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision: not(pole_coordinates in obstacle_area), rob range ~ [-0.05, 4.0]
        # note: clip the upperbound to a lower value in order to consider safe all the state that are
        # far enough from the obstacle, otherwise it would be incentivated to extreme values in x, theta (falldown)
        collision_fn, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0,
                                                info=info)
        nodes["S_coll"] = (collision_fn, collision_fn)

        # falldown: theta_limit - |theta|, rob range: [0.0, 1.55]
        cont_fall_fn, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                                min_r_state={'theta': info['theta_limit']},
                                                max_r_state={'theta': 0.0},
                                                info=info)
        nodes["S_fall"] = (cont_fall_fn, cont_fall_fn)

        # outside: x_limit - |x|, rob range: [0.0, 2.5]
        cont_exit_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                min_r_state={'x': info['x_limit']},
                                                max_r_state={'x': 0.0},
                                                info=info)
        nodes["S_exit"] = (cont_exit_fn, cont_exit_fn)

        # define target rules
        # note: progress is computed as progress/time, bound it to have approx same scale
        progress_fn, _ = get_normalized_reward(fns.ProgressToTargetReward(progress_coeff=PROGCOEFF),
                                               min_r=-1.0, max_r=1.0)
        target_fun, _ = get_normalized_reward(fns.ReachTargetReward(),
                                              min_r_state={'x': info['x_limit']},
                                              max_r_state={'x': info['x_target'] - info['x_target_tol']},
                                              info=info)
        nodes["T_origin"] = (progress_fn, target_fun)

        # define comfort rules
        # balance: theta_tol - |theta - theta_target|, rob range: [-1.25, 0.4] (considering all theta domain +-pi/2)
        # Q: better to normalize over the rob of theta domain, or the rob of the comfort domain?
        nodes["T_bal"] = get_normalized_reward(fns.BalanceReward(),
                                               min_r_state={'theta': info['theta_target'] - info['theta_target_tol']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)

        if self._env_params['task'] == "random_height":
            raise NotImplemented("task random height not implemented")
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphContinuousSafetyProgressDistanceTargetContinuousIndicator(GraphRewardConfig):

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision: not(pole_coordinates in obstacle_area), rob range ~ [-0.05, 4.0]
        # note: clip the upperbound to a lower value in order to consider safe all the state that are
        # far enough from the obstacle, otherwise it would be incentivated to extreme values in x, theta (falldown)
        collision_fn, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0,
                                                info=info)
        nodes["S_coll"] = (collision_fn, collision_fn)

        # falldown: theta_limit - |theta|, rob range: [0.0, 1.55]
        cont_fall_fn, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                                min_r_state={'theta': info['theta_limit']},
                                                max_r_state={'theta': 0.0},
                                                info=info)
        nodes["S_fall"] = (cont_fall_fn, cont_fall_fn)

        # outside: x_limit - |x|, rob range: [0.0, 2.5]
        cont_exit_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                min_r_state={'x': info['x_limit']},
                                                max_r_state={'x': 0.0},
                                                info=info)
        nodes["S_exit"] = (cont_exit_fn, cont_exit_fn)

        # define target rules
        # note: progress is computed as progress/time, bound it to have approx same scale
        progress_fn = fns.ProgressTimesDistanceToTargetReward()
        nodes["T_origin"] = (progress_fn, progress_fn)

        # define comfort rules
        # balance: theta_tol - |theta - theta_target|, rob range: [-1.25, 0.4] (considering all theta domain +-pi/2)
        # Q: better to normalize over the rob of theta domain, or the rob of the comfort domain?
        nodes["T_bal"] = get_normalized_reward(fns.BalanceReward(),
                                               min_r_state={'theta': info['theta_target'] - info['theta_target_tol']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)

        if self._env_params['task'] == "random_height":
            raise NotImplemented("task random height not implemented")
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphBinarySafetyProgressDistanceTargetContinuousIndicator(GraphRewardConfig):

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision: not(pole_coordinates in obstacle_area), rob range ~ [-0.05, 4.0]
        # note: clip the upperbound to a lower value in order to consider safe all the state that are
        # far enough from the obstacle, otherwise it would be incentivated to extreme values in x, theta (falldown)
        collision_fn = fns.CollisionReward(collision_penalty=-1.0, no_collision_bonus=0.0)
        collision_sat, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0,
                                                 info=info)
        nodes["S_coll"] = (collision_fn, collision_sat)

        # falldown: theta_limit - |theta|, rob range: [0.0, 1.55]
        falldown_fn = fns.FalldownReward(falldown_penalty=-1.0, no_falldown_bonus=0.0)
        cont_fall_sat, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                                 min_r_state={'theta': info['theta_limit']},
                                                 max_r_state={'theta': 0.0},
                                                 info=info)
        nodes["S_fall"] = (falldown_fn, cont_fall_sat)

        # outside: x_limit - |x|, rob range: [0.0, 2.5]
        exit_fn = fns.OutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0)
        cont_exit_sat, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                 min_r_state={'x': info['x_limit']},
                                                 max_r_state={'x': 0.0},
                                                 info=info)
        nodes["S_exit"] = (exit_fn, cont_exit_sat)

        # define target rules
        # note: progress is computed as progress/time, bound it to have approx same scale
        progress_fn = fns.ProgressTimesDistanceToTargetReward()
        nodes["T_origin"] = (progress_fn, progress_fn)

        # define comfort rules
        # balance: theta_tol - |theta - theta_target|, rob range: [-1.25, 0.4] (considering all theta domain +-pi/2)
        # Q: better to normalize over the rob of theta domain, or the rob of the comfort domain?
        nodes["T_bal"] = get_normalized_reward(fns.BalanceReward(),
                                               min_r_state={'theta': info['theta_target'] - info['theta_target_tol']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)

        if self._env_params['task'] == "random_height":
            raise NotImplemented("task random height not implemented")
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOGraphContinuousSafetyProgressMaxTargetContinuousIndicator(GraphRewardConfig):

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision: not(pole_coordinates in obstacle_area), rob range ~ [-0.05, 4.0]
        # note: clip the upperbound to a lower value in order to consider safe all the state that are
        # far enough from the obstacle, otherwise it would be incentivated to extreme values in x, theta (falldown)
        collision_fn, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0,
                                                info=info)
        nodes["S_coll"] = (collision_fn, collision_fn)

        # falldown: theta_limit - |theta|, rob range: [0.0, 1.55]
        cont_fall_fn, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                                min_r_state={'theta': info['theta_limit']},
                                                max_r_state={'theta': 0.0},
                                                info=info)
        nodes["S_fall"] = (cont_fall_fn, cont_fall_fn)

        # outside: x_limit - |x|, rob range: [0.0, 2.5]
        cont_exit_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                min_r_state={'x': info['x_limit']},
                                                max_r_state={'x': 0.0},
                                                info=info)
        nodes["S_exit"] = (cont_exit_fn, cont_exit_fn)

        # define target rules
        # note: progress is computed as progress/time, bound it to have approx same scale
        progress_fn = fns.ProgressAndStayToTargetReward()
        target_fun, _ = get_normalized_reward(fns.ReachTargetReward(),
                                              min_r_state={'x': info['x_limit']},
                                              max_r_state={'x': info['x_target'] - info['x_target_tol']},
                                              info=info)
        nodes["T_origin"] = (progress_fn, target_fun)

        # define comfort rules
        # balance: theta_tol - |theta - theta_target|, rob range: [-1.25, 0.4] (considering all theta domain +-pi/2)
        # Q: better to normalize over the rob of theta domain, or the rob of the comfort domain?
        nodes["T_bal"] = get_normalized_reward(fns.BalanceReward(),
                                               min_r_state={'theta': info['theta_target'] - info['theta_target_tol']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)

        if self._env_params['task'] == "random_height":
            raise NotImplemented("task random height not implemented")
        return nodes

    @property
    def topology(self):
        topology = get_cartpole_topology(self._env_params['task'])
        return topology


class CPOChainGraph(GraphRewardConfig):
    """
    all the safety requirements are evaluated as a single conjunction
    """

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'x_limit': self._env_params['x_limit'],
                'x_target': self._env_params['x_target'],
                'x_target_tol': self._env_params['x_target_tol'],
                'theta_limit': np.deg2rad(self._env_params['theta_limit']),
                'theta_target': np.deg2rad(self._env_params['theta_target']),
                'theta_target_tol': np.deg2rad(self._env_params['theta_target_tol'])}

        # define safety rules
        # collision
        # note: the rob values depend on the obstacle size
        collision_fn, collision_sat = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0,
                                                            info=info)

        # falldown
        fun = fns.ContinuousFalldownReward()
        min_r_state, max_r_state = {'theta': info['theta_limit']}, {'theta': 0.0}
        falldown_fn, falldown_sat = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state,
                                                          info=info)

        # outside
        fun = fns.ContinuousOutsideReward()
        min_r_state, max_r_state = {'x': info['x_limit']}, {'x': 0.0}
        outside_fn, outside_sat = get_normalized_reward(fun, min_r_state=min_r_state, max_r_state=max_r_state,
                                                        info=info)

        # define single safety rule as conjunction of the three
        funs = [collision_fn, falldown_fn, outside_fn]
        sats = [collision_sat, falldown_sat, outside_sat]
        nodes["S_all"] = (MinAggregatorReward(funs), ProdAggregatorReward(sats))

        # define target rules
        # note: progress is computed as progress/time, bound it to have approx same scale
        progress_fn, _ = get_normalized_reward(fns.ProgressToTargetReward(progress_coeff=PROGCOEFF),
                                               min_r=-1.0, max_r=1.0)
        target_fun, _ = get_normalized_reward(fns.ReachTargetReward(),
                                              min_r_state={'x': info['x_limit']},
                                              max_r_state={'x': info['x_target'] - info['x_target_tol']},
                                              info=info)
        nodes["T_origin"] = (progress_fn, target_fun)

        # define comfort rules
        nodes["T_bal"] = get_normalized_reward(fns.BalanceReward(),
                                               min_r_state={'theta': info['theta_target'] - info['theta_target_tol']},
                                               max_r_state={'theta': info['theta_target']},
                                               info=info)
        return nodes

    @property
    def topology(self):
        # just to avoid to rewrite it all the times
        if self._env_params['task'] == "fixed_height":
            topology = {
                'S_all': ['T_origin'],
                'T_origin': ['T_bal'],
            }
        else:
            raise NotImplemented(f"no reward-topology for task {self._env_params['task']}")
        return topology
