from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


class RC2STLReward(TLRewardConfig):
    _no_collision = "always((collision<=0) and (dist_ego2npc <= safety_distance))"
    _lap_completion = "eventually(progress >= target_progress)"

    @property
    def spec(self) -> str:
        # Safety
        safety_requirement = self._no_collision
        # Target
        target_requirement = self._lap_completion
        # all together
        spec = f"({safety_requirement} and {target_requirement})"
        return spec

    @property
    def requirements_dict(self):
        return {'no_collision': self._no_collision,
                'lap_completion': self._lap_completion}

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'progress', 'target_progress',
                'dist2obst', 'target_dist2obst', 'dist_ego2npc', 'safety_distance']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['steps'],
            'collision': 1.0 if state['collision'] > 0 else -1.0,
            'progress': state['progress'],  # already in 0 or 1
            'target_progress': info['target_progress'],
            'dist2obst': state['dist2obst'],  # already in 0 or 1
            'target_dist2obst': info['target_dist2obst'],
            'safety_distance': info['safety_distance'],
            'dist_ego2npc': state['dist_ego2npc'],
        }
        self._max_episode_len = info['max_steps']
        return monitored_state
