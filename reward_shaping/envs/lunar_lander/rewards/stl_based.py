rom typing import List, Dict, Any
import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class STLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1 : the y velocity should never be such that it crashes
        no_y_crash = f"always((y+delta*y_dot)>= 0)"
        # Safety 2 : Theta angle should be bounded
        spacecraft_angle = f"always(abs(theta) < theta_limit)"
        fuel_usage = f"always(fuel >= 0)"
        safety_requirement = f"(({no_y_crash}) and ({spacecraft_angle}) and ({fuel_usage}))"

        # Target : reach origin
        target_requirement = f"eventually(always(dist_origin <= dist_origin_tol))"

        # Comfort 1: Small Horizontal Speed (same as for no_y_crash)
        horizontal_speed = f"always(sign_x*(x+delta*x_dot)>= 0)"
        # Comfort 3: Small Angle Velocity
        angular_velocity = f"always(abs(theta_dot) <= theta_dot_limit)"
        # Comfort Requirements
        comfort_requirement = f"({horizontal_speed} and {angular_velocity})"

        # combination
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'delta',
                'x', 'y',
                'x_dot', 'y_dot',
                'theta', 'theta_dot',
                'theta_limit', 'theta_dot_limit',
                'dist_origin_tol', 'sign_x',
                'fuel']

    @property
    def monitoring_types(self):
        return ['int', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float',
                'float', 'float',
                'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'delta': 1/info['FPS'],
            'x': state[0],
            'y': state[1],
            'x_dot': state[2],
            'y_dot': state[3],
            'theta': state[4],
            'theta_dot': state[5],
            'theta_limit': info['theta_limit'],
            'theta_dot_limit': info['theta_dot_limit'],
            'dist_origin_tol': info['dist_origin_tol'],
            'sign_x': np.sign(state[0]),
            'fuel': info['fuel']
        }
        return monitored_state