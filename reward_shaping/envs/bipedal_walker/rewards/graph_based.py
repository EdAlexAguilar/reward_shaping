from abc import ABC, abstractmethod
import reward_shaping.envs.bipedal_walker.rewards.subtask_rewards as fns
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
        info = {'angle_hull_limit': self._env_params['angle_hull_limit'],
                'speed_y_limit': self._env_params['speed_y_limit'],
                'angle_vel_limit': self._env_params['angle_vel_limit']}

        # safety rules
        fun = fns.BinaryFalldownReward()
        nodes["S_fall"] = (fun, ThresholdIndicator(fun, include_zero=False))

        # define target rule
        fun = fns.SpeedTargetReward()       # this is already normalized in +-1
        nodes["T_move"] = (NormalizedReward(fun, 0, +1), ThresholdIndicator(fun, threshold=0.1))

        # define comfort rules
        # note: for comfort rules, the indicators does not necessarly reflect the satisfaction
        # since they are last layer in the hierarchy, we do not care (simplicity)
        fun = fns.ContinuousHullAngleReward()
        min_r_state = [info['angle_hull_limit']] + [0.0]*23
        max_r_state = [0.0] + [0.0]*23
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["C_angle"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        fun = fns.ContinuousVerticalSpeedReward()
        min_r_state = [0.0]*3 + [info['speed_y_limit']] + [0.0] * 20
        max_r_state = [0.0] * 24
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["C_v_y"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))

        fun = fns.ContinuousHullAngleVelocityReward()
        min_r_state = [0.0] + [info['angle_vel_limit']] + [0.0] * 22
        max_r_state = [0.0] * 24
        min_r, max_r = fun(min_r_state, info=info), fun(max_r_state, info=info)
        nodes["C_angle_vel"] = (NormalizedReward(fun, min_r, max_r), ThresholdIndicator(fun))
        return nodes

    @property
    def topology(self):
        """
                          / Comfort: Hull Angle
        Safety -- Target  - Comfort: Hull Angle Vel.
                          \ Comfort: Vertical Speed
        """
        topology = {
            'S_fall': ['T_move'],
            'T_move': ['C_angle', 'C_angle_vel', 'C_v_y']
        }
        return topology