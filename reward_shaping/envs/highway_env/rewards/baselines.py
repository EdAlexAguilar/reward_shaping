import math
import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.highway_env import highway_utils

from typing import List
from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_stl_episode


class HighwaySparseTargetReward(RewardFunction):
    """
        reward(s,a) := bonus, if target is reached
        reward(s,a) := penalty for crash
    """

    def reached_target(self, state, info):
        assert 'distance_to_target' in state and 'TARGET_DISTANCE' in info
        return 1 - clip_and_norm(state['distance_to_target'], 0, info['TARGET_DISTANCE'])

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'observation' in state and 'done' in info
        assert 'collision' in state
        if info['done']:
            if state['collision'] == 1:
                return -1.0
            elif self.reached_target(state, info):
                return 1.0
        return 0


class HighwayProgressTargetReward(RewardFunction):
    """
    reward(s, a, s') := target(s') - target(s)
    """

    def target_potential(self, state, info):
        assert 'distance_to_target' in state and 'TARGET_DISTANCE' in info
        return 1 - clip_and_norm(state['distance_to_target'], 0, info['TARGET_DISTANCE'])

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'TARGET_DISTANCE' in info
        progress = self.target_potential(next_state, info) - self.target_potential(state, info)
        return progress


class HighwayEvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super(HighwayEvalConfig, self).__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self) -> List[str]:
        return ['time', 'MAX_STEPS',
                'ego_x', 'ego_y', 'ego_vx', 'ego_vy',
                'collision',
                'violated_safe_distance',
                'violated_hard_speed_limit', 'HARD_SPEED_LIMIT', 'SOFT_SPEED_LIMIT', 'SPEED_LOWER_BOUND',
                'road_progress', 'distance_to_target', 'TARGET_DISTANCE', 'TARGET_DISTANCE_TOL',
                'max_velocity_difference_to_left',
                'X_LIMIT', 'Y_LIMIT', 'VX_LIMIT', 'VY_LIMIT']

    @property
    def monitoring_types(self) -> List[str]:
        return ['int', 'int',
                'float', 'float', 'float', 'float',
                'float',
                'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info):
        monitored_state = {
            'time': state['step_count'],
            'MAX_STEPS': info['MAX_STEPS'],
            'ego_x': state['ego_x'],
            'ego_y': state['ego_y'],
            'ego_vx': state['ego_vx'],
            'ego_vy': state['ego_vy'],
            'collision': state['collision'],
            'violated_safe_distance': state['violated_safe_distance'],
            'violated_hard_speed_limit': state['violated_hard_speed_limit'],
            'HARD_SPEED_LIMIT': info['HARD_SPEED_LIMIT'],
            'SOFT_SPEED_LIMIT': info['SOFT_SPEED_LIMIT'],
            'SPEED_LOWER_BOUND': info['SPEED_LOWER_BOUND'],
            'road_progress': state['road_progress'],
            'distance_to_target': state['distance_to_target'],
            'TARGET_DISTANCE': info['TARGET_DISTANCE'],
            'TARGET_DISTANCE_TOL': info['TARGET_DISTANCE_TOL'],
            'max_velocity_difference_to_left': state['max_velocity_difference_to_left'],
            'X_LIMIT': info['X_LIMIT'],
            'Y_LIMIT': info['Y_LIMIT'],
            'VX_LIMIT': info['VX_LIMIT'],
            'VY_LIMIT': info['VY_LIMIT']
        }
        self._max_episode_len = info['MAX_STEPS']
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always((violated_safe_distance==0) and (violated_hard_speed_limit==0))"
        safety_rho = monitor_stl_episode(stl_spec=safety_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        target_spec = "eventually((distance_to_target <= TARGET_DISTANCE_TOL))"
        target_rho = monitor_stl_episode(stl_spec=target_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        comfort_softlim_spec = "(ego_vx<=SOFT_SPEED_LIMIT)"
        comfort_lowlim_spec = "(ego_vx>=SPEED_LOWER_BOUND)"
        comfort_slwlft_spec = "(max_velocity_difference_to_left<=0)"
        comfort_metrics = []
        for comfort_spec in [comfort_softlim_spec, comfort_lowlim_spec, comfort_slwlft_spec]:
            comfort_trace = monitor_stl_episode(stl_spec=comfort_spec,
                                                vars=self.monitoring_variables, types=self.monitoring_types,
                                                episode=episode)
            # Ask Luigi : Where
            comfort_trace = comfort_trace + [[-1, -1] for _ in
                                             range((self._max_episode_len - len(comfort_trace)))]
            comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
            comfort_metrics.append(comfort_mean)
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * comfort_mean
        return tot_score
