from typing import List, Dict, Any
import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class STLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1 : Head away from floor and obstacles as measured by lidar
        no_falldown = f"always(min(lidar) >= dist_hull_limit)"
        safety_requirement = f"({no_falldown})"
        # Target 1: Keep moving in x
        target_requirement = f"always(v_x >= 0)"
        # Comfort 1: Small Hull Angle
        hull_angle = f"always(abs(phi) <= angle_hull_limit)"
        # Comfort 2: Vertical Speed
        vertical_speed = f"always(abs(v_y) <= speed_y_hull_limit)"
        # Comfort 3: Hull Angle Vel
        hull_angle_velocity = f"always(abs(phi_dot) <= angle_vel_hull_limit)"
        # Comfort Requirements
        comfort_requirement = f"({hull_angle} and {vertical_speed} and {hull_angle_velocity})"
        # combination
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec


"""
class STLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1: G(|theta|<=theta_limit)
        no_falldown = f"always(abs(theta) <= theta_limit)"
        # Safety 2: G(|x|<=x_limit)
        no_outside = f"always(abs(x) <= x_limit)"
        # Safety 3: G(not(obst_left<=pole_x<=obst_right and obst_bottom<=pole_y<=obst_top))
        obst_intersect_polex = f"(obst_left < pole_x) and (pole_x < obst_right)"
        obst_intersect_poley = f"(obst_bottom < pole_y) and (pole_y < obst_top)"
        no_collision = f"always(not(({obst_intersect_polex}) and ({obst_intersect_poley})))"
        # All Safeties: AND_{i=1,2,3} Safety_i
        safety_requirements = f"({no_falldown}) and ({no_outside}) and ({no_collision})"
        # Target 1: F(G(|x-x_target|<=x_target_tolerance))
        target_requirement = f"eventually(always(dist_target_x <= x_target_tol))"
        # Target 2: F(G(|theta-theta_target|<=theta_target_tolerance))
        balance_requirement = f"eventually(always(dist_target_theta <= theta_target_tol))"
        # All toghether: Safeties and BalanceReq and (env_feasible->TargetReq)
        spec = f"({safety_requirements}) and ((is_feasible>0)->{target_requirement}) and ({balance_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'dist_target_x', 'dist_target_theta',
                'x', 'x_limit', 'x_target', 'x_target_tol',
                'theta', 'theta_limit', 'theta_target', 'theta_target_tol',
                'pole_x', 'pole_y', 'obst_left', 'obst_right', 'obst_bottom', 'obst_top', 'is_feasible']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'x': state['x'],
            'x_limit': info['x_limit'],
            'theta_limit': info['x_limit'],
            'x_target': info['x_target'],
            'x_target_tol': info['x_target_tol'],
            'theta': state['theta'],
            'theta_limit': info['theta_limit'],
            'theta_target': info['theta_target'],
            'theta_target_tol': info['theta_target_tol'],
            'pole_x': state['x'] + info['pole_length'] * np.sin(state['theta']),
            'pole_y': info['axle_y'] + info['pole_length'] * np.cos(state['theta']),
            'obst_left': state['obstacle_left'],
            'obst_right': state['obstacle_right'],
            'obst_bottom': state['obstacle_bottom'],
            'obst_top': state['obstacle_top'],
            'dist_target_x': abs(state['x'] - info['x_target']),
            'dist_target_theta': abs(state['theta'] - info['theta_target']),
            'is_feasible': float(info['is_feasible'])
        }
        return monitored_state


class BoolSTLReward(STLRewardConfig):

    @property
    def spec(self) -> str:
        # safety specs bool
        no_falldown_bool = f"always(falldown >= 0.0)"
        no_outside_bool = f"always(outside >= 0)"
        no_collision_bool = f"always(collision >= 0)"
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
            'collision': -3.0 if info['collision'] else 3.0,
            'outside': -3.0 if info['outside'] else 3.0,
            'falldown': -3.0 if info['falldown'] else 3.0,
            'dist_target_x': abs(state['x'] - info['x_target']),
            'dist_target_theta': abs(state['theta'] - info['theta_target']),
            'is_feasible': float(info['is_feasible'])
        }
        return monitored_state
"""