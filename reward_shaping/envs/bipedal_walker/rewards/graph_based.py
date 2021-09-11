from abc import ABC, abstractmethod

import reward_shaping.envs.bipedal_walker.rewards.subtask_rewards as fns
from reward_shaping.core.helper_fns import ThresholdIndicator, MinAggregatorReward, \
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


class BWGraphWithContinuousScoreBinaryIndicator(GraphRewardConfig):
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
                'angle_vel_limit': self._env_params['angle_vel_limit'],
                'speed_x_target': self._env_params['speed_x_target']}

        # safety rules
        fun = fns.get_subtask_reward("binary_falldown")
        nodes["S_fall"] = (fun, ThresholdIndicator(fun))

        # define target rule: speed_x >= speed__xtarget
        nodes["T_move"] = get_normalized_reward(fns.SpeedTargetReward(),  # this is already normalized in +-1
                                                min_r_state={'horizontal_speed': info['speed_x_target']},
                                                max_r_state={'horizontal_speed': 1.0},
                                                info=info,
                                                threshold=info['speed_x_target'])

        # define comfort rules
        # note: for comfort rules, the indicators do not necessarly need to reflect the satisfaction
        # since they are last layer in the hierarchy, we do not care (for simplicity)
        nodes["C_angle"] = get_normalized_reward(fns.ContinuousHullAngleReward(),
                                                 min_r_state={'hull_angle': info['angle_hull_limit']},
                                                 max_r_state={'hull_angle': 0.0},
                                                 info=info,
                                                 threshold=info['angle_hull_limit'])
        nodes["C_v_y"] = get_normalized_reward(fns.ContinuousVerticalSpeedReward(),
                                               min_r_state={'vertical_speed': info['speed_y_limit']},
                                               max_r_state={'vertical_speed': 0.0},
                                               info=info,
                                               threshold=info['speed_y_limit'])
        nodes["C_angle_vel"] = get_normalized_reward(fns.ContinuousHullAngleVelocityReward(),
                                                     min_r_state={'hull_angle_speed': info['angle_vel_limit']},
                                                     max_r_state={'hull_angle_speed': 0.0},
                                                     info=info,
                                                     threshold=info['angle_vel_limit'])

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


class BWGraphWithBinarySafetyProgressTargetContinuousIndicator(GraphRewardConfig):
    """
    This reward uses CONTINUOUS INDICATORS, meaning that it weights the hierarchy with the sat degree of ancestors
    In particular, this reward uses:
        - BINARY score for safety nodes, norm safety-rho as satisfaction
        - PROGRESS score for target nodes, dist to target as satisfaction
        - normalized continuous score and sat for comfort nodes
    """

    def __init__(self, env_params):
        self._env_params = env_params

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'dist_hull_limit': self._env_params['dist_hull_limit'],
                'angle_hull_limit': self._env_params['angle_hull_limit'],
                'speed_y_limit': self._env_params['speed_y_limit'],
                'angle_vel_limit': self._env_params['angle_vel_limit'],
                'speed_x_target': self._env_params['speed_x_target'],
                'norm_target_x': 1.0, 'collision': False}

        # safety rules
        binary_fall_fun = fns.get_subtask_reward("binary_falldown")
        cont_fall_fun = fns.get_subtask_reward("continuous_falldown")
        nodes["S_fall"] = (binary_fall_fun, cont_fall_fun)

        # define target rule: speed_x >= speed__xtarget
        progress_fn, _ = get_normalized_reward(fns.SpeedTargetReward(),  # this is already normalized in +-1
                                               min_r_state={'horizontal_speed': info['speed_x_target']},
                                               max_r_state={'horizontal_speed': 1.0},
                                               info=info)
        nodes["T_move"] = (progress_fn, progress_fn)

        # define comfort rules
        # note: for comfort rules, the indicators do not necessarly need to reflect the satisfaction
        # since they are last layer in the hierarchy, we do not care (for simplicity)
        nodes["C_angle"] = get_normalized_reward(fns.ContinuousHullAngleReward(),
                                                 min_r_state={'hull_angle': info['angle_hull_limit']},
                                                 max_r_state={'hull_angle': 0.0},
                                                 info=info,
                                                 threshold=info['angle_hull_limit'])
        nodes["C_v_y"] = get_normalized_reward(fns.ContinuousVerticalSpeedReward(),
                                               min_r_state={'vertical_speed': info['speed_y_limit']},
                                               max_r_state={'vertical_speed': 0.0},
                                               info=info,
                                               threshold=info['speed_y_limit'])
        nodes["C_angle_vel"] = get_normalized_reward(fns.ContinuousHullAngleVelocityReward(),
                                                     min_r_state={'hull_angle_speed': info['angle_vel_limit']},
                                                     max_r_state={'hull_angle_speed': 0.0},
                                                     info=info,
                                                     threshold=info['angle_vel_limit'])

        return nodes

    @property
    def topology(self):
        """
                          / Comfort: Hull Angle
        Safety -- Target  - Comfort: Hull Angle Vel.
                          \ Comfort: Vertical Speedinfo['angle_vel_limit']
        """
        topology = {
            'S_fall': ['T_move'],
            'T_move': ['C_angle', 'C_angle_vel', 'C_v_y']
        }
        return topology


class BWGraphWithBinarySafetyProgressTargetContinuousIndicatorNoComfort(GraphRewardConfig):
    """
    As bpr_ci but without any comfort node. In order to perform an ablation study on the effect of hierarchy
    """

    def __init__(self, env_params):
        self._env_params = env_params

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'dist_hull_limit': self._env_params['dist_hull_limit'],
                'angle_hull_limit': self._env_params['angle_hull_limit'],
                'speed_y_limit': self._env_params['speed_y_limit'],
                'angle_vel_limit': self._env_params['angle_vel_limit'],
                'speed_x_target': self._env_params['speed_x_target'],
                'norm_target_x': 1.0, 'collision': False}

        # safety rules
        binary_fall_fun = fns.get_subtask_reward("binary_falldown")
        cont_fall_fun = fns.get_subtask_reward("continuous_falldown")
        nodes["S_fall"] = (binary_fall_fun, cont_fall_fun)

        # define target rule: speed_x >= speed__xtarget
        progress_fn, _ = get_normalized_reward(fns.SpeedTargetReward(),  # this is already normalized in +-1
                                               min_r_state={'horizontal_speed': info['speed_x_target']},
                                               max_r_state={'horizontal_speed': 1.0},
                                               info=info)
        nodes["T_move"] = (progress_fn, progress_fn)

        return nodes

    @property
    def topology(self):
        """
        Safety -- Target
        """
        topology = {
            'S_fall': ['T_move'],
        }
        return topology


class BWChainGraph(GraphRewardConfig):
    """
    graph-based with 1 node for each level of the hierarchy
    """

    def __init__(self, env_params):
        self._env_params = env_params

    @property
    def nodes(self):
        nodes = {}
        # prepare env info
        info = {'angle_hull_limit': self._env_params['angle_hull_limit'],
                'speed_y_limit': self._env_params['speed_y_limit'],
                'angle_vel_limit': self._env_params['angle_vel_limit'],
                'speed_x_target': self._env_params['speed_x_target']}

        # safety rules
        nodes["S_fall"] = (fns.get_subtask_reward("binary_falldown"), fns.get_subtask_reward("binary_falldown"))

        # define target rule: speed_x >= speed__xtarget
        progress_fn, _ = get_normalized_reward(fns.SpeedTargetReward(),  # this is already normalized in +-1
                                               min_r_state={'horizontal_speed': info['speed_x_target']},
                                               max_r_state={'horizontal_speed': 1.0},
                                               info=info,
                                               threshold=info['speed_x_target'])
        nodes["T_move"] = (progress_fn, progress_fn)

        # define comfort rules
        # note: for comfort rules, the indicators do not necessarly need to reflect the satisfaction
        # since they are last layer in the hierarchy, we do not care (for simplicity)
        angle_fn, angle_sat = get_normalized_reward(fns.ContinuousHullAngleReward(),
                                                    min_r_state={'hull_angle': info['angle_hull_limit']},
                                                    max_r_state={'hull_angle': 0.0},
                                                    info=info,
                                                    threshold=info['angle_hull_limit'])
        vy_fn, vy_sat = get_normalized_reward(fns.ContinuousVerticalSpeedReward(),
                                              min_r_state={'vertical_speed': info['speed_y_limit']},
                                              max_r_state={'vertical_speed': 0.0},
                                              info=info,
                                              threshold=info['speed_y_limit'])
        angle_vel_fn, angle_vel_sat = get_normalized_reward(fns.ContinuousHullAngleVelocityReward(),
                                                            min_r_state={'hull_angle_speed': info['angle_vel_limit']},
                                                            max_r_state={'hull_angle_speed': 0.0},
                                                            info=info,
                                                            threshold=info['angle_vel_limit'])

        # define single comfort rule as conjunction of the three
        funs = [angle_fn, vy_fn, angle_vel_fn]
        sats = [angle_sat, vy_sat, angle_vel_sat]
        nodes["C_all"] = (MinAggregatorReward(funs), ProdAggregatorReward(sats))

        return nodes

    @property
    def topology(self):
        """
        Safety -- Target  - Comfort: Hull Angle AND Hull Angle Vel AND Vertical Speed
        """
        topology = {
            'S_fall': ['T_move'],
            'T_move': ['C_all']
        }
        return topology
