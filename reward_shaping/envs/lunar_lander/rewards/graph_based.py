from abc import ABC, abstractmethod
import reward_shaping.envs.lunar_lander.rewards.subtask_rewards as fns
import numpy as np

from reward_shaping.core.helper_fns import ThresholdIndicator, NormalizedReward


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

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'FPS': self._env_params['FPS'],
                'theta_limit': self._env_params['theta_limit'],
                'fuel': self._env_params['fuel'],
                'theta_dot_limit': self._env_params['theta_dot_limit']}

        # SAFETY RULES
        fun = fns.MinimizeYVelocity()
        max_r_state = [0, 1.5, 0, 0.5, 0, 0, 0, 0]
        max_r = fun(max_r_state, info=info)
        min_r = 0.0
        nodes["S_crash_y"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        fun = fns.MinimizeCraftAngle()
        max_r_state = [0]*8
        max_r = fun(max_r_state, info=info)
        min_r = 0
        nodes["S_theta"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        fun = fns.MinimizeFuelConsumption()
        max_r = 1
        min_r = 0
        node["S_fuel"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        # TARGET RULES
        fun = fns.MinimizeDistanceToLandingArea()
        max_r_state = [1.0, 1.5] + [0]*6
        max_r = fun(max_r_state, info=info)
        min_r = 0
        node["T_origin"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        # COMFORT RULES
        fun = fns.MinimizeXVelocity()
        max_r_state = [1.5, 0, 0.5, 0, 0, 0, 0, 0]
        max_r = fun(max_r_state, info=info)
        min_r = 0.0
        nodes["C_x_dot"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        fun = fns.MinimizeAngleVelocity()
        max_r_state = [0] * 8
        max_r = fun(max_r_state, info=info)
        min_r = 0
        nodes["C_theta_dot"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))
        return nodes

    @property
    def topology(self):
        """
        Safety: crash_y  \
        Safety: angle   -- Target  -- Comfort: Angular Velocity
        Safety: fuel     /          \ Comfort: Horizontal Speed
        """
        topology = {
            'S_crash_y': ['T_origin'],
            'S_theta': ['T_origin'],
            'S_fuel': ['T_origin'],
            'T_origin': ['C_theta_dot', 'C_x_dot']
        }
        return topology