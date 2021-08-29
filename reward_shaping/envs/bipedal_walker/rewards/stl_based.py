from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class BWSTLReward(STLRewardConfig):
    @property
    def spec(self) -> str:
        # Safety 1 : Head away from floor and obstacles as measured by lidar
        safety_requirement = "always(collision<=0)"
        # Target 1: reach the end of the terrain (progress==1 <-> pos_x==target_x)
        target_requirement = "eventually(progress_x >= 1.0)"
        # Comfort 1, Small Hull Angle (|phinorm|==1 <-> |phi|==angle_limit)
        hull_angle = "always(abs(phi_norm) <= 1.0)"
        # Comfort 2: Vertical Speed
        vertical_speed = "(abs(vy_norm) <= 1.0)"
        # Comfort 3: Hull Angle Vel
        hull_angle_velocity = "(abs(phidot_norm) <= 1.0)"
        # Comfort Requirements
        comfort_requirement = f"({hull_angle} and {vertical_speed} and {hull_angle_velocity})"
        # all together
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'progress_x', 'phi_norm', 'vy_norm', 'phidot_norm']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['time'],
            'collision': info['collision'],  # already 0 or 1
            'progress_x': np.clip(info['position_x'], 0.0, info['target_x']) / info['target_x'],
            'phi_norm': np.clip(state[0], -info['angle_hull_limit'], info['angle_hull_limit']) / info[
                'angle_hull_limit'],
            'vy_norm': np.clip(state[3], -info['speed_y_limit'], info['speed_y_limit']) / info['speed_y_limit'],
            'phidot_norm': np.clip(state[1], -info['angle_vel_limit'], info['angle_vel_limit']) / info[
                'angle_vel_limit']
        }
        return monitored_state
