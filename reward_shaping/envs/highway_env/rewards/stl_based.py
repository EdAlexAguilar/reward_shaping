from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


def _get_highway_default_monitoring_variables():
    return ['time', 'MAX_STEPS',
            'ego_x', 'ego_y', 'ego_vx', 'ego_vy',
            'collision',
            'violated_safe_distance',
            'violated_hard_speed_limit', 'HARD_SPEED_LIMIT', 'SOFT_SPEED_LIMIT', 'SPEED_LOWER_BOUND',
            'road_progress', 'distance_to_target', 'TARGET_DISTANCE', 'TARGET_DISTANCE_TOL',
            'max_velocity_difference_to_left',
            'X_LIMIT', 'Y_LIMIT', 'VX_LIMIT', 'VY_LIMIT']

def _get_highway_default_monitoring_types():
    return ['int', 'int',
            'float', 'float', 'float', 'float',
            'float',
            'float',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float',
            'float',
            'float', 'float', 'float', 'float']


def _get_highway_default_monitoring_procedure(state, done, info):
    step_count_norm = state['step_count'] / info['MAX_STEPS']
    max_steps_norm = info['MAX_STEPS'] / info['MAX_STEPS']
    x_norm = state['ego_x'] / info['X_LIMIT']  # allow normalized value to be out-of-bound to have neg robustness
    y_norm = state['ego_y'] / info['Y_LIMIT']
    vx_norm = state['ego_vx'] / info['VX_LIMIT']
    vy_norm = state['ego_vy'] / info['VY_LIMIT']
    hard_speed_limit_norm = info['HARD_SPEED_LIMIT'] / info['HARD_SPEED_LIMIT']
    soft_speed_limit_norm = info['SOFT_SPEED_LIMIT'] / info['HARD_SPEED_LIMIT']
    speed_lower_bound_norm = info['SPEED_LOWER_BOUND'] / info['HARD_SPEED_LIMIT']
    road_progress_norm = state['road_progress']/info['X_LIMIT']
    distance_to_target_norm = state['distance_to_target']/info['X_LIMIT']
    target_distance_norm = info['TARGET_DISTANCE']/info['X_LIMIT']
    target_distance_tol_norm = info['TARGET_DISTANCE_TOL']/info['X_LIMIT']
    max_velocity_difference_to_left_norm = state['max_velocity_difference_to_left'] / info['VX_LIMIT']
    x_limit_norm = info['X_LIMIT'] / info['X_LIMIT']
    y_limit_norm = info['Y_LIMIT'] / info['Y_LIMIT']
    vx_limit_norm = info['VX_LIMIT'] / info['VX_LIMIT']
    vy_limit_norm = info['VY_LIMIT'] / info['VY_LIMIT']

    # compute monitoring variables
    monitored_state = {
        'violated_safe_distance': 1.0 if state['violated_safe_distance'] > 0 else -1.0,
        'violated_hard_speed_limit': 1.0 if state['violated_hard_speed_limit'] > 0 else -1.0,
        'distance_to_target': distance_to_target_norm,
        'TARGET_DISTANCE_TOL': target_distance_tol_norm,

        'time': step_count_norm,
        'max_steps': max_steps_norm,

        'ego_x': x_norm,
        'ego_y': y_norm,
        'ego_vx': vx_norm,
        'ego_vy': vy_norm,
        'collision': state['collision'],

        'HARD_SPEED_LIMIT': hard_speed_limit_norm,
        'SOFT_SPEED_LIMIT': soft_speed_limit_norm,
        'SPEED_LOWER_BOUND': speed_lower_bound_norm,
        'road_progress': road_progress_norm,
        'TARGET_DISTANCE': target_distance_norm,
        'max_velocity_difference_to_left': max_velocity_difference_to_left_norm,
        'X_LIMIT': x_limit_norm,
        'Y_LIMIT': y_limit_norm,
        'VX_LIMIT': vx_limit_norm,
        'VY_LIMIT': vy_limit_norm
    }
    return monitored_state


class HighwaySTLReward(TLRewardConfig):
    _safe_distance = "always(violated_safe_distance<0)"
    _safe_speed = "always(violated_hard_speed_limit<0)"
    _reach_target = "eventually(distance_to_target <= TARGET_DISTANCE_TOL)"

    @property
    def spec(self) -> str:
        safety_requirements = f"({self._safe_distance}) and ({self._safe_speed})"
        target_requirement = self._reach_target
        spec = f"({safety_requirements}) and ({target_requirement})"
        return spec

    @property
    def requirements_dict(self):
        return {'safe_distance': self._safe_distance,
                'safe_speed': self._safe_speed,
                'reach_target': self._reach_target}

    @property
    def monitoring_variables(self):
        return _get_highway_default_monitoring_variables()

    @property
    def monitoring_types(self):
        return _get_highway_default_monitoring_types()

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        return _get_highway_default_monitoring_procedure(state, done, info)
