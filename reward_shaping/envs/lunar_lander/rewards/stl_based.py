from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class STLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1 : y-velocity should never be such that it crashes (ie. y+delta*y_dot >= 0)
        no_y_crash = "always(y_pred >= 0)"
        # Safety 2 : fuel must be always be positive
        fuel_usage = "always(fuel >= 0)"
        # Safety 3: no collision with obstacle
        no_collision = "always(collision <= 0.0)"
        # Safety 4: craft always within the x limits
        no_outside = "always(abs(x) <= x_limit)"
        # all safeties
        safety_requirement = f"(({no_y_crash}) and ({fuel_usage}) and ({no_collision}) and ({no_outside}))"

        # Target : reach origin
        target_requirement = "eventually(always(dist_target <= halfwidth_landing_area))"

        # Comfort 1: Small Horizontal Speed (same as for no_y_crash)
        limit_theta = "always(abs(theta) <= theta_limit)"
        # Comfort 2: Small Angle Velocity
        limit_theta_dot = "always(abs(theta_dot) <= theta_dot_limit)"
        # Comfort Requirements
        comfort_requirement = f"({limit_theta} and {limit_theta_dot})"

        # all together
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'y_pred',
                'x', 'x_limit',
                'theta', 'theta_dot', 'theta_limit', 'theta_dot_limit',
                'fuel', 'collision', 'dist_target', 'halfwidth_landing_area']

    @property
    def monitoring_types(self):
        return ['int', 'float',
                'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        delta_t = 1.0 / info['FPS']
        norm_target_x = (info['x_target'] - info['x_low_limit']) / (info['x_high_limit'] - info['x_low_limit'])
        norm_target_y = 0.0
        norm_land_area = info['halfwidth_landing_area'] / info['x_high_limit']
        max_dist = 2.0  # assuming norm x, y approx ranging in +- 1, the max dist is less than 2.0
        target_dist = np.linalg.norm([state[0]-norm_target_x, state[1]-norm_target_y])
        norm_target_dist = np.clip(target_dist, 0.0, max_dist) / max_dist
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'y_pred': state[1] + delta_t * state[3],  # y_pred = y + delta_t * y_dot
            'x': state[0],  # already normalized in +-1
            'x_limit': 1.0,
            'theta': state[4],  # already normalized in +-1
            'theta_dot': state[5],  # already normalized in +-1
            'theta_limit': info['theta_limit'] / np.pi,
            'theta_dot_limit': info['theta_dot_limit'] / np.pi,
            'fuel': info['fuel'],   # in [0,1]
            'collision': info['collision'],
            'dist_target': norm_target_dist,
            'halfwidth_landing_area': norm_land_area
        }
        return monitored_state