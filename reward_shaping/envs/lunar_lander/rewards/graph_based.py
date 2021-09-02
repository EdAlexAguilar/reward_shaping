from abc import ABC, abstractmethod
import reward_shaping.envs.lunar_lander.rewards.subtask_rewards as fns
import numpy as np

from reward_shaping.core.helper_fns import ThresholdIndicator, NormalizedReward, MinAggregatorReward, \
    ProdAggregatorReward
from reward_shaping.core.utils import get_normalized_reward


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


class LLGraphWithBinarySafetyBinaryIndicator(GraphRewardConfig):
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
                'theta_dot_limit': self._env_params['theta_dot_limit']}

        # SAFETY RULES
        safe_landing_fn = fns.BinarySlowLandingReward(slow_bonus=0.0, crash_penalty=-1.0)
        nodes["S_crash"] = (safe_landing_fn, ThresholdIndicator(safe_landing_fn))

        fuel_fn = fns.BinaryFuelReward(still_fuel_bonus=0.0, no_fuel_penalty=-1.0)
        nodes["S_fuel"] = (fuel_fn, ThresholdIndicator(fuel_fn))

        coll_fn = fns.CollisionReward(no_collision_bonus=0.0, collision_penalty=-1.0)
        nodes["S_coll"] = (coll_fn, ThresholdIndicator(coll_fn))

        exit_fn = fns.OutsideReward(no_exit_bonus=0.0, exit_penalty=-1.0)
        nodes["S_exit"] = (exit_fn, ThresholdIndicator(exit_fn))

        # TARGET RULES
        progress_fn = fns.ProgressToOriginReward(progress_coeff=1.0)
        nodes["T_origin"] = (progress_fn, ThresholdIndicator(progress_fn, include_zero=False))

        # COMFORT RULES
        theta_limit = self._env_params['theta_limit']
        thetadot_limit = self._env_params['theta_dot_limit']
        nodes["C_angle"] = get_normalized_reward(fns.MinimizeCraftAngle(),
                                                 min_r_state=[0] * 4 + [theta_limit] + [0] * 9,
                                                 max_r_state=[0] * 14,
                                                 info=info)

        nodes["C_angvel"] = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                  min_r_state=[0] * 5 + [thetadot_limit] + [0] * 8,
                                                  max_r_state=[0] * 14,
                                                  info=info)
        return nodes

    @property
    def topology(self):
        """
        S_crash \               / Comfort: Theta angle
        S_coll  | __ Target  --|
        S_fuel  |              \ Comfort: Ang Velocity
        S_exit  /
        """
        topology = {
            'S_crash': ['T_origin'],
            'S_coll': ['T_origin'],
            'S_fuel': ['T_origin'],
            'S_exit': ['T_origin'],
            'T_origin': ['C_angle', 'C_angvel']
        }
        return topology


class LLChainGraph(GraphRewardConfig):
    """
    all the safety and comfort requirements are evaluated as a single reqiurement
    """

    def __init__(self, env_params):
        self._env_params = env_params

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'FPS': self._env_params['FPS'],
                'theta_limit': self._env_params['theta_limit'],
                'theta_dot_limit': self._env_params['theta_dot_limit'],
                'x_high_limit': self._env_params['x_high_limit'], 'half_width': 600 / 30 / 2}

        # SAFETY RULES
        safe_landing_fn, safe_landing_sat = get_normalized_reward(fns.SlowLandingReward(), min_r=-0.1, max_r=1.5)

        fuel_fn = fns.FuelReward()
        fuel_sat = ThresholdIndicator(fuel_fn, include_zero=False)

        coll_fn, coll_sat = get_normalized_reward(fns.CollisionReward(no_collision_bonus=0.0, collision_penalty=-1.0),
                                                  min_r=-1.0, max_r=0.0)

        exit_fn, exit_sat = get_normalized_reward(fns.OutsideReward(), min_r_state=[1.0], max_r_state=[0.0], info=info)

        safety_funs = [safe_landing_fn, fuel_fn, coll_fn, exit_fn]
        safety_sats = [safe_landing_sat, fuel_sat, coll_sat, exit_sat]
        nodes["S_all"] = (MinAggregatorReward(safety_funs), ProdAggregatorReward(safety_sats))

        # TARGET RULES
        progress_fn = fns.ProgressToOriginReward(progress_coeff=1.0)
        nodes["T_origin"] = (progress_fn, ThresholdIndicator(progress_fn, include_zero=False))

        # COMFORT RULES
        theta_limit = self._env_params['theta_limit']
        thetadot_limit = self._env_params['theta_dot_limit']
        angle_fn, angle_sat = get_normalized_reward(fns.MinimizeCraftAngle(),
                                                    min_r_state=[0] * 4 + [theta_limit] + [0] * 9,
                                                    max_r_state=[0] * 14,
                                                    info=info)

        anglevel_fn, anglevel_sat = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                          min_r_state=[0] * 5 + [theta_dot_limit] + [0] * 8,
                                                          max_r_state=[0] * 14,
                                                          info=info)
        comfort_funs = [angle_fn, anglevel_fn]
        comfort_sats = [angle_sat, anglevel_sat]
        nodes["C_all"] = (MinAggregatorReward(comfort_funs), ProdAggregatorReward(comfort_sats))

        return nodes

    @property
    def topology(self):
        # just to avoid to rewrite it all the times
        topology = {
            'S_all': ['T_origin'],
            'T_origin': ['C_all'],
        }
        return topology
