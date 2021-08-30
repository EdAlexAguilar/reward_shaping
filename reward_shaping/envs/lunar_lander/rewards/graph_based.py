from abc import ABC, abstractmethod
import reward_shaping.envs.lunar_lander.rewards.subtask_rewards as fns
import numpy as np

from reward_shaping.core.helper_fns import ThresholdIndicator, NormalizedReward
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
                'fuel': self._env_params['fuel'],
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
        progress_fn = fns.ProgressToTargetReward(progress_coeff=1.0)
        nodes["T_origin"] = (progress_fn, ThresholdIndicator(progress_fn, include_zero=False))

        # COMFORT RULES
        # todo define min max to these!
        nodes["C_angle"] = get_normalized_reward(fns.MinimizeCraftAngle(), min_r_state=[0] * 8,
                                                 max_r_state=[0] * 8, info=info)

        nodes["C_angvel"] = get_normalized_reward(fns.MinimizeAngleVelocity(), min_r_state=[0] * 8,
                                                 max_r_state=[0] * 8, info=info)
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
