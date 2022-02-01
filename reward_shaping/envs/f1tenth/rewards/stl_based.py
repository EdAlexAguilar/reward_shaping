from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


class F110STLReward(TLRewardConfig):
    _no_collision = "always(collision <= 0)"
    _complete_lap = "eventually(progress >= 1.0)"

    @property
    def spec(self) -> str:
        # Safety
        safety_requirement = self._no_collision
        # Target
        target_requirement = self._complete_lap
        # all together
        spec = f"({safety_requirement} and {target_requirement})"
        return spec

    @property
    def requirements_dict(self):
        return {'no_collision': self._no_collision,
                'complete_lap': self._complete_lap}

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'progress']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['time'],
            'collision': state['collision'],  # already 0 or 1
            'progress': info['progress']
        }
        return monitored_state
