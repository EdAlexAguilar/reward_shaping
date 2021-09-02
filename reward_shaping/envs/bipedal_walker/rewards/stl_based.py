from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import STLRewardConfig


class BWSTLReward(STLRewardConfig):
    _no_collision = "always(collision<=0)"
    _continuous_progress = "always(eventually(vx>=0.0))"
    _comfort_angle = "always(abs(phi_norm) <= 1.0)"
    _comfort_y_speed = "always(abs(vy_norm) <= 1.0)"
    _comfort_angle_speed = "always(abs(phidot_norm) <= 1.0)"

    @property
    def spec(self) -> str:
        # Safety
        safety_requirement = self._no_collision
        # Target
        target_requirement = self._continuous_progress
        # Comfort
        comfort_requirement = f"({self._comfort_angle} and {self._comfort_y_speed} and {self._comfort_angle_speed})"
        # all together
        spec = f"({safety_requirement} and {target_requirement} and {comfort_requirement})"
        return spec

    @property
    def requirements_dict(self):
        return {'no_collision': self._no_collision,
                'cont_progress': self._continuous_progress,
                'comfort_angle': self._comfort_angle,
                'comfort_y_speed': self._comfort_y_speed,
                'comfort_angle_speed': self._comfort_angle_speed}

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'vx', 'phi_norm', 'vy_norm', 'phidot_norm']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['time'],
            'collision': info['collision'],  # already 0 or 1
            'vx': state['horizontal_speed'],
            'phi_norm': np.clip(state['hull_angle'], -info['angle_hull_limit'], info['angle_hull_limit']) / info[
                'angle_hull_limit'],
            'vy_norm': np.clip(state['vertical_speed'], -info['speed_y_limit'], info['speed_y_limit']) / info[
                'speed_y_limit'],
            'phidot_norm': np.clip(state['hull_angle_speed'], -info['angle_vel_limit'], info['angle_vel_limit']) / info[
                'angle_vel_limit']
        }
        return monitored_state
