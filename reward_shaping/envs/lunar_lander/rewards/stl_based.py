from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


class LLSTLReward(TLRewardConfig):
    _no_collision = "always(collision <= 0.0)"  # Safety 1: no collision with obstacle
    _no_outside = "always(abs(x) <= x_limit)"  # Safety 2: craft always within the x limits
    _reach_origin = "eventually(always(dist_target <= halfwidth_landing_area))"  # Target: reach origin

    @property
    def spec(self) -> str:
        safety_requirement = f"(({self._no_collision}) and ({self._no_outside}))"
        target_requirement = self._reach_origin
        # all together
        spec = f"({safety_requirement} and {target_requirement})"
        return spec

    @property
    def monitoring_variables(self):
        return ['time', 'x', 'x_limit', 'fuel', 'collision', 'dist_target', 'halfwidth_landing_area']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'x': state['x'],  # already normalized in +-1
            'x_limit': info['x_limit'],
            'fuel': state['fuel'],  # in [0,1]
            'collision': state['collision'],
            'dist_target': np.linalg.norm([state['x'], state['y']]),
            'halfwidth_landing_area': info['halfwidth_landing_area']
        }
        return monitored_state
