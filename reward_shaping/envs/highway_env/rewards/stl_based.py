from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


def _get_highway_default_monitoring_variables():
    return ['time', 'max_steps',
            'violated_safe_distance',
            'violated_hard_speed_limit',
            'distance_to_target', 'TARGET_DISTANCE_TOL',
            ]


def _get_highway_default_monitoring_types():
    return ['int', 'int',
            'float', 'float', 'float', 'float']


def _get_highway_default_monitoring_procedure(state, done, info):
    step_count_norm = state['step_count'] / info['MAX_STEPS']
    max_steps_norm = info['MAX_STEPS'] / info['MAX_STEPS']
    distance_to_target_norm = state['distance_to_target'] / info['X_LIMIT']
    target_distance_tol_norm = info['TARGET_DISTANCE_TOL'] / info['X_LIMIT']

    # compute monitoring variables
    monitored_state = {
        'time': step_count_norm,
        'max_steps': max_steps_norm,

        'violated_safe_distance': 1.0 if state['violated_safe_distance'] > 0 else -1.0,
        'violated_hard_speed_limit': 1.0 if state['violated_hard_speed_limit'] > 0 else -1.0,
        'distance_to_target': distance_to_target_norm,
        'TARGET_DISTANCE_TOL': target_distance_tol_norm,

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
