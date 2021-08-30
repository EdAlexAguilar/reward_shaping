from typing import List, Dict, Any
import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class CPOSTLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1: G(|theta|<=theta_limit)
        no_falldown = "always(abs(theta) <= theta_limit)"
        # Safety 2: G(|x|<=x_limit)
        no_outside = "always(abs(x) <= x_limit)"
        # Safety 3: G(no collision)
        no_collision = f"always(collision <= 0.0)"
        # All Safeties: AND_{i=1,2,3} Safety_i
        safety_requirements = f"({no_falldown}) and ({no_outside}) and ({no_collision})"
        # Target 1: F(G(|x-x_target|<=x_target_tolerance))
        target_requirement = "eventually(always(dist_target_x <= x_target_tol))"
        # Target 2: F(G(|theta-theta_target|<=theta_target_tolerance))
        balance_requirement = "eventually(always(dist_target_theta <= theta_target_tol))"
        # All toghether: Safeties and BalanceReq and (env_feasible->TargetReq)
        spec = f"({safety_requirements}) and ((is_feasible>0)->{target_requirement}) and ({balance_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time',
                'x', 'x_limit', 'x_target', 'x_target_tol',
                'theta', 'theta_limit', 'theta_target', 'theta_target_tol',
                'collision', 'dist_target_x', 'dist_target_theta', 'is_feasible']

    @property
    def monitoring_types(self):
        return ['int',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
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
            'is_feasible': 1.0 if info['is_feasible'] else 0.0
        }
        return monitored_state


class CPOBoolSTLReward(STLRewardConfig):

    @property
    def spec(self) -> str:
        # safety specs bool
        no_falldown_bool = f"always(falldown <= 0.0)"
        no_outside_bool = f"always(outside <= 0)"
        no_collision_bool = f"always(collision <= 0)"
        safety_requirements = f"({no_falldown_bool}) and ({no_outside_bool}) and ({no_collision_bool})"
        # target spec
        target_requirement = f"eventually(always(dist_target_x <= x_target_tol))"
        balance_requirement = f"eventually(always(dist_target_theta <= theta_target_tol))"
        # all together
        spec = f"({safety_requirements}) and ((is_feasible>0)->{target_requirement}) and ({balance_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'dist_target_x', 'dist_target_theta',
                'x_target_tol', 'theta_target_tol',
                'collision', 'falldown', 'outside', 'is_feasible']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float',
                'float', 'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'x_target_tol': info['x_target_tol'],
            'theta_target_tol': info['theta_target_tol'],
            'collision': 3.0 if info['collision'] else -3.0,
            'outside': 3.0 if info['outside'] else -3.0,
            'falldown': 3.0 if info['falldown'] else -3.0,
            'dist_target_x': abs(state['x'] - info['x_target']),
            'dist_target_theta': abs(state['theta'] - info['theta_target']),
            'is_feasible': float(info['is_feasible'])
        }
        return monitored_state
