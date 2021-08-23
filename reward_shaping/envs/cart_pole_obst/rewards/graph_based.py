from abc import ABC, abstractmethod

import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
from reward_shaping.envs.helper_fns import ThresholdIndicator, NormalizedReward
import numpy as np


class GraphRewardConfig(ABC):

    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def topology(self):
        pass


class GraphWithContinuousScoreBinaryIndicator(GraphRewardConfig):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} sigma(r')) * rho(r)
    with sigma returns binary value {0,1}
    """

    def __init__(self, env_params):
        self._env_params = env_params
        self._task = env_params['task']

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
        nodes["S_coll"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        # falldown
        fun = fns.ContinuousFalldownReward()
        min_r_state = {'theta': info['theta_limit']}
        max_r_state = {'theta': 0.0}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["S_fall"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        # outside
        fun = fns.ContinuousOutsideReward()
        min_r_state = {'x': info['x_limit']}
        max_r_state = {'x': 0.0}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["S_exit"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        # define target rules
        fun = fns.ReachTargetReward()
        min_r_state = {'x': info['x_limit']}
        max_r_state = {'x': info['x_target']}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["T_origin"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        # define comfort rules
        fun = fns.BalanceReward()
        min_r_state = {'theta': info['theta_limit']}
        max_r_state = {'theta': info['theta_target']}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        balance_reward_fn = NormalizedReward(fun, min_r, max_r)
        balance_ind_fn = ThresholdIndicator(fun)
        nodes["T_bal"] = (balance_reward_fn, balance_ind_fn)

        if self._task == "random_height":
            # for random env, additional comfort node
            nodes["C_bal"] = (balance_reward_fn, balance_ind_fn)
            # conditional nodes (ie, to check env conditions)
            zero_fn = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            feas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nfeas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nodes["H_feas"] = (zero_fn, feas_ind)
            nodes["H_nfeas"] = (zero_fn, nfeas_ind)
        return nodes

    @property
    def topology(self):
        if self._task == "fixed_height":
            topology = {
                'S_coll': ['T_origin'],
                'S_fall': ['T_origin'],
                'S_exit': ['T_origin'],
                'T_origin': ['T_bal'],
            }
        elif self._task == "random_height":
            topology = {
                'S_coll': ['H_feas', 'H_nfeas'],
                'S_fall': ['H_feas', 'H_nfeas'],
                'S_exit': ['H_feas', 'H_nfeas'],
                'H_feas': ['T_origin'],
                'H_nfeas': ['T_bal'],
                'T_origin': ['C_bal'],
            }
        else:
            raise NotImplemented(f"no reward-topology for task {self._task}")
        return topology


class GraphWithContinuousScoreContinuousIndicator(GraphRewardConfig):
    """
        rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} rho(r')) * rho(r)
    """

    def __init__(self, env_params):
        self._env_params = env_params
        self._task = env_params['task']

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
        nodes["S_coll"] = (NormalizedReward(fun, min_r, max_r), NormalizedReward(fun, min_r, max_r))

        # falldown
        fun = fns.ContinuousFalldownReward()
        min_r_state = {'theta': info['theta_limit']}
        max_r_state = {'theta': 0.0}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["S_fall"] = (NormalizedReward(fun, min_r, max_r), NormalizedReward(fun, min_r, max_r))

        # outside
        fun = fns.ContinuousOutsideReward()
        min_r_state = {'x': info['x_limit']}
        max_r_state = {'x': 0.0}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["S_exit"] = (NormalizedReward(fun, min_r, max_r), NormalizedReward(fun, min_r, max_r))

        # define target rules
        fun = fns.ReachTargetReward()
        min_r_state = {'x': info['x_limit']}
        max_r_state = {'x': info['x_target']}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["T_origin"] = (NormalizedReward(fun, min_r, max_r), NormalizedReward(fun, min_r, max_r))

        # define comfort rules
        fun = fns.BalanceReward()
        min_r_state = {'theta': info['theta_limit']}
        max_r_state = {'theta': info['theta_target']}
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        balance_reward_fn = NormalizedReward(fun, min_r, max_r)
        balance_ind_fn = ThresholdIndicator(fun)
        nodes["T_bal"] = (balance_reward_fn, balance_reward_fn)

        if self._task == "random_height":
            # for random env, additional comfort node
            nodes["C_bal"] = (balance_reward_fn, balance_reward_fn)
            # conditional nodes (ie, to check env conditions)
            zero_fn = lambda _: 0.0  # this is a static condition, do not score for it (depends on the env)
            feas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nfeas_ind = ThresholdIndicator(fns.CheckOvercomingFeasibility())
            nodes["H_feas"] = (zero_fn, feas_ind)
            nodes["H_nfeas"] = (zero_fn, nfeas_ind)
        return nodes

    @property
    def topology(self):
        if self._task == "fixed_height":
            topology = {
                'S_coll': ['T_origin'],
                'S_fall': ['T_origin'],
                'S_exit': ['T_origin'],
                'T_origin': ['T_bal'],
            }
        elif self._task == "random_height":
            topology = {
                'S_coll': ['H_feas', 'H_nfeas'],
                'S_fall': ['H_feas', 'H_nfeas'],
                'S_exit': ['H_feas', 'H_nfeas'],
                'H_feas': ['T_origin'],
                'H_nfeas': ['T_bal'],
                'T_origin': ['C_bal'],
            }
        else:
            raise NotImplemented(f"no reward-topology for task {self._task}")
        return topology
