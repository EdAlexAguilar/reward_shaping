from typing import List, Dict, Any
import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class STLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1 : Head away from floor and obstacles as measured by lidar
        no_falldown = f"always(collision==0)"
        safety_requirement = f"({no_falldown})"
        # Target 1: Keep moving in x
        target_requirement = f"always(v_x >= speed_x_target)"
        # Comfort 1: Small Hull Angle
        hull_angle = f"always(abs(phi) <= angle_hull_limit)"
        # Comfort 2: Vertical Speed
        vertical_speed = f"always(abs(v_y) <= speed_y_limit)"
        # Comfort 3: Hull Angle Vel
        hull_angle_velocity = f"always(abs(phi_dot) <= angle_vel_limit)"
        # Comfort Requirements
        comfort_requirement = f"({hull_angle} and {vertical_speed} and {hull_angle_velocity})"
        # combination
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'collision',
                'v_x', 'v_y',
                'phi', 'phi_dot',
                'angle_hull_limit', 'speed_y_limit',
                'angle_vel_limit', 'speed_x_target']

    @property
    def monitoring_types(self):
        return ['int', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'collision': info['collision'],
            'v_x': state[2],
            'v_y': state[3],
            'phi': state[0],
            'phi_dot': state[1],
            'angle_hull_limit': info['angle_hull_limit'],
            'speed_y_limit': info['speed_y_limit'],
            'angle_vel_limit': info['angle_vel_limit'],
            'speed_x_target': info['speed_x_target']
        }
        return monitored_state



class BoolSTLReward(STLRewardConfig):

    @property
    def spec(self) -> str:
        # Safety 1 : Head away from floor and obstacles as measured by lidar
        no_falldown = f"always(collision==0)"
        safety_requirement = f"({no_falldown})"
        # Target 1: Keep moving in x
        target_requirement = f"always(progress>= 0)"
        # Comfort 1: Small Hull Angle
        hull_angle = f"always(abs(phi) <= angle_hull_limit)"
        # Comfort 2: Vertical Speed
        vertical_speed = f"always(abs(v_y) <= speed_y_limit)"
        # Comfort 3: Hull Angle Vel
        hull_angle_velocity = f"always(abs(phi_dot) <= angle_vel_limit)"
        # Comfort Requirements
        comfort_requirement = f"({hull_angle} and {vertical_speed} and {hull_angle_velocity})"
        # combination
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec


    @property
    def monitoring_variables(self):
        return ['time', 'collision',
                'progress', 'v_y',
                'phi', 'phi_dot',
                'angle_hull_limit', 'speed_y_limit',
                'angle_vel_limit', 'speed_x_target']

    @property
    def monitoring_types(self):
        return ['int', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'collision': -3.0 if info['collision'] else 3.0,
            'progress': -3.0 if (state[2]<info['speed_x_target']) else 3.0,
            'v_y': state[3],
            'phi': state[0],
            'phi_dot': state[1],
            'angle_hull_limit': info['angle_hull_limit'],
            'speed_y_limit': info['speed_y_limit'],
            'angle_vel_limit': info['angle_vel_limit'],
            'speed_x_target': info['speed_x_target']
        }
        return monitored_state