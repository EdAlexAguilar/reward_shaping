from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


def _get_cpo_default_monitoring_variables():
    return ['time',
            'x', 'x_limit', 'x_target', 'x_target_tol',
            'theta', 'theta_limit', 'theta_target', 'theta_target_tol',
            'collision', 'dist_target_x', 'dist_target_theta']


def _get_cpo_default_monitoring_types():
    return ['int',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float']


def _get_cpo_default_monitoring_procedure(state, done, info):
    x_norm = state['x'] / info['x_limit']  # allow normalized value to be out-of-bound to have neg robustness
    x_target_norm = np.clip(info['x_target'], -info['x_limit'], info['x_limit']) / info['x_limit']
    theta_norm = state['theta'] / info['theta_limit']
    theta_target_norm = np.clip(info['theta_target'], -info['theta_limit'], info['theta_limit']) / info[
        'theta_limit']
    # compute monitoring variables
    monitored_state = {
        'time': info['time'],
        'x': x_norm,
        'x_limit': info['x_limit'] / info['x_limit'],  # 1.0, just for clarity
        'x_target': x_target_norm,
        'x_target_tol': np.clip(info['x_target_tol'], -info['x_limit'], info['x_limit']) / info['x_limit'],
        'theta': theta_norm,
        'theta_limit': info['theta_limit'] / info['theta_limit'],  # 1.0
        'theta_target': theta_target_norm,
        'theta_target_tol': np.clip(info['theta_target_tol'], -info['theta_limit'], info['theta_limit']) / info[
            'theta_limit'],
        'collision': 1.0 if info['collision'] else 0.0,
        'dist_target_x': abs(x_norm - x_target_norm),
        'dist_target_theta': abs(theta_norm - theta_target_norm),
    }
    return monitored_state


class CPOSTLReward(TLRewardConfig):
    _no_falldown = "always(abs(theta) <= theta_limit)"
    _no_outside = "always(abs(x) <= x_limit)"
    _no_collision = "always(collision <= 0.0)"
    _reach_origin = "eventually(always((dist_target_x <= x_target_tol) and (abs(theta) <= theta_target_tol)))"

    @property
    def spec(self) -> str:
        safety_requirements = f"({self._no_falldown}) and ({self._no_outside}) and ({self._no_collision})"
        target_requirement = self._reach_origin
        # All toghether: Safeties and BalanceReq and (env_feasible->TargetReq)
        spec = f"({safety_requirements}) and ({target_requirement})"
        return spec

    @property
    def requirements_dict(self):
        return {'no_collision': self._no_collision,
                'no_outside': self._no_outside,
                'no_falldown': self._no_falldown,
                'reach_origin': self._reach_origin,
                'balance': self._balance}

    @property
    def monitoring_variables(self):
        return _get_cpo_default_monitoring_variables()

    @property
    def monitoring_types(self):
        return _get_cpo_default_monitoring_types()

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        return _get_cpo_default_monitoring_procedure(state, done, info)
