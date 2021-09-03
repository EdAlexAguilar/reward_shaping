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
                'angle_limit': self._env_params['angle_limit'],
                'angle_speed_limit': self._env_params['angle_speed_limit'],
                "x_target": self._env_params['x_target'],
                "y_target": self._env_params['y_target'],
                "halfwidth_landing_area": self._env_params['halfwidth_landing_area'],
                }

        # SAFETY RULES
        binary_fuel = fns.get_subtask_reward("binary_fuel")
        fuel_sat = fns.get_subtask_reward("fuel_sat")
        nodes["S_fuel"] = (binary_fuel, fuel_sat)

        binary_collision = fns.get_subtask_reward("binary_collision")
        collision_sat = fns.get_subtask_reward("collision_sat")
        nodes["S_coll"] = (binary_collision, collision_sat)

        binary_exit = fns.get_subtask_reward("binary_exit")
        exit_sat = fns.get_subtask_reward("exit_sat")
        nodes["S_exit"] = (binary_exit, exit_sat)

        # TARGET RULES
        progress_fn = fns.get_subtask_reward("continuous_progress")
        target_sat = fns.get_subtask_reward("target_sat")
        nodes["T_origin"] = (progress_fn, target_sat)

        # COMFORT RULES
        angle_limit = self._env_params['angle_limit']
        angle_fn, angle_sat = get_normalized_reward(fns.MinimizeCraftAngle(),
                                                    min_r_state={'angle': angle_limit},
                                                    max_r_state={'angle': 0.0},
                                                    info=info)
        nodes["C_angle"] = (angle_fn, angle_sat)

        angle_speed_limit = self._env_params['angle_speed_limit']
        angle_speed_fn, angle_speed_sat = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                                min_r_state={'angle_speed': angle_speed_limit},
                                                                max_r_state={'angle_speed': 0.0},
                                                                info=info)
        nodes["C_angle_speed"] = (angle_speed_fn, angle_speed_sat)
        return nodes

    @property
    def topology(self):
        """
                                / Comfort: angle
        S_coll  | __ Target  --|
        S_fuel  |              \ Comfort: Ang speed
        S_exit  /
        """
        topology = {
            'S_coll': ['T_origin'],
            'S_fuel': ['T_origin'],
            'S_exit': ['T_origin'],
            'T_origin': ['C_angle', 'C_angle_speed']
        }
        return topology


class LLGraphWithBinarySafetyContinuousIndicator(GraphRewardConfig):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} sigma(r')) * rho(r)
    with sigma returns continuous value in [0,1]
    """

    def __init__(self, env_params):
        self._env_params = env_params

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'FPS': self._env_params['FPS'],
                'angle_limit': self._env_params['angle_limit'],
                'angle_speed_limit': self._env_params['angle_speed_limit'],
                "x_target": self._env_params['x_target'],
                "y_target": self._env_params['y_target'],
                "halfwidth_landing_area": self._env_params['halfwidth_landing_area'],
                }

        # SAFETY RULES
        binary_fuel = fns.get_subtask_reward("binary_fuel")
        continuous_fuel = fns.get_subtask_reward("continuous_fuel")
        nodes["S_fuel"] = (binary_fuel, continuous_fuel)

        binary_collision = fns.get_subtask_reward("binary_collision")
        continuous_collision = fns.get_subtask_reward("continuous_collision")
        nodes["S_coll"] = (binary_collision, continuous_collision)

        binary_exit = fns.get_subtask_reward("binary_exit")
        continuous_exit = fns.get_subtask_reward("continuous_exit")
        nodes["S_exit"] = (binary_exit, continuous_exit)

        # TARGET RULES
        progress_fn = fns.get_subtask_reward("continuous_progress")
        target_fn, _ = get_normalized_reward(fns.MinimizeDistanceToLandingArea(),
                                             min_r_state={'x': 1.0, 'y': 1.0},
                                             max_r_state={'x': 0.0, 'y': 0.0},
                                             info=info)
        nodes["T_origin"] = (progress_fn, target_fn)

        # COMFORT RULES
        angle_limit = self._env_params['angle_limit']
        angle_fn, _ = get_normalized_reward(fns.MinimizeCraftAngle(),
                                            min_r_state={'angle': angle_limit},
                                            max_r_state={'angle': 0.0},
                                            info=info)
        nodes["C_angle"] = (angle_fn, angle_fn)

        angle_speed_limit = self._env_params['angle_speed_limit']
        angle_speed_fn, _ = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                  min_r_state={'angle_speed': angle_speed_limit},
                                                  max_r_state={'angle_speed': 0.0},
                                                  info=info)
        nodes["C_angle_speed"] = (angle_speed_fn, angle_speed_fn)
        return nodes

    @property
    def topology(self):
        """
                                / Comfort: angle
        S_coll  | __ Target  --|
        S_fuel  |              \ Comfort: Ang speed
        S_exit  /
        """
        topology = {
            'S_coll': ['T_origin'],
            'S_fuel': ['T_origin'],
            'S_exit': ['T_origin'],
            'T_origin': ['C_angle', 'C_angle_speed']
        }
        return topology



class LLGraphWithBinarySafetyProgressTimesDistanceTargetContinuousIndicator(GraphRewardConfig):
    """
    rew(R) = Sum_{r in R} (Product_{r' in R st. r' <= r} sigma(r')) * rho(r)
    with sigma returns continuous value in [0,1]
    """

    def __init__(self, env_params):
        self._env_params = env_params

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'FPS': self._env_params['FPS'],
                'angle_limit': self._env_params['angle_limit'],
                'angle_speed_limit': self._env_params['angle_speed_limit'],
                "x_target": self._env_params['x_target'],
                "y_target": self._env_params['y_target'],
                "halfwidth_landing_area": self._env_params['halfwidth_landing_area'],
                }

        # SAFETY RULES
        binary_fuel = fns.get_subtask_reward("binary_fuel")
        continuous_fuel = fns.get_subtask_reward("continuous_fuel")
        nodes["S_fuel"] = (binary_fuel, continuous_fuel)

        binary_collision = fns.get_subtask_reward("binary_collision")
        continuous_collision = fns.get_subtask_reward("continuous_collision")
        nodes["S_coll"] = (binary_collision, continuous_collision)

        binary_exit = fns.get_subtask_reward("binary_exit")
        continuous_exit = fns.get_subtask_reward("continuous_exit")
        nodes["S_exit"] = (binary_exit, continuous_exit)

        # TARGET RULES
        progress_fn = fns.get_subtask_reward("continuous_progress")
        target_fn = fns.get_subtask_reward("progress_x_distance")
        nodes["T_origin"] = (progress_fn, target_fn)

        # COMFORT RULES
        angle_limit = self._env_params['angle_limit']
        angle_fn, _ = get_normalized_reward(fns.MinimizeCraftAngle(),
                                            min_r_state={'angle': angle_limit},
                                            max_r_state={'angle': 0.0},
                                            info=info)
        nodes["C_angle"] = (angle_fn, angle_fn)

        angle_speed_limit = self._env_params['angle_speed_limit']
        angle_speed_fn, _ = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                  min_r_state={'angle_speed': angle_speed_limit},
                                                  max_r_state={'angle_speed': 0.0},
                                                  info=info)
        nodes["C_angle_speed"] = (angle_speed_fn, angle_speed_fn)
        return nodes

    @property
    def topology(self):
        """
                                / Comfort: angle
        S_coll  | __ Target  --|
        S_fuel  |              \ Comfort: Ang speed
        S_exit  /
        """
        topology = {
            'S_coll': ['T_origin'],
            'S_fuel': ['T_origin'],
            'S_exit': ['T_origin'],
            'T_origin': ['C_angle', 'C_angle_speed']
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
                'angle_limit': self._env_params['angle_limit'],
                'angle_speed_limit': self._env_params['angle_speed_limit'],
                "x_target": self._env_params['x_target'],
                "y_target": self._env_params['y_target'],
                "halfwidth_landing_area": self._env_params['halfwidth_landing_area'],
                }

        # SAFETY RULES
        binary_fuel = fns.get_subtask_reward("binary_fuel")
        fuel_sat = fns.get_subtask_reward("fuel_sat")

        binary_collision = fns.get_subtask_reward("binary_collision")
        collision_sat = fns.get_subtask_reward("collision_sat")

        binary_exit = fns.get_subtask_reward("binary_exit")
        exit_sat = fns.get_subtask_reward("exit_sat")

        safety_funs = [binary_fuel, binary_collision, binary_exit]
        safety_sats = [fuel_sat, collision_sat, exit_sat]
        nodes["S_all"] = (MinAggregatorReward(safety_funs), ProdAggregatorReward(safety_sats))

        # TARGET RULES
        progress_fn = fns.get_subtask_reward("continuous_progress")
        target_fn, _ = get_normalized_reward(fns.MinimizeDistanceToLandingArea(),
                                             min_r_state={'x': 1.0, 'y': 1.0},
                                             max_r_state={'x': 0.0, 'y': 0.0},
                                             info=info)
        nodes["T_origin"] = (progress_fn, target_fn)

        # COMFORT RULES
        angle_limit = self._env_params['angle_limit']
        angle_fn, angle_sat = get_normalized_reward(fns.MinimizeCraftAngle(),
                                                    min_r_state={'angle': angle_limit},
                                                    max_r_state={'angle': 0.0},
                                                    info=info)
        angle_speed_limit = self._env_params['angle_speed_limit']
        angle_speed_fn, angle_speed_sat = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                                min_r_state={'angle_speed': angle_speed_limit},
                                                                max_r_state={'angle_speed': 0.0},
                                                                info=info)
        comfort_funs = [angle_fn, angle_speed_fn]
        comfort_sats = [angle_sat, angle_speed_sat]
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
