import math
from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_stl_episode


class RC2EvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'progress', 'target_progress', 'dist2obst', 'target_dist2obst',
                'velocity_x', 'min_velx', 'max_velx', 'last_steering', 'comfort_max_steering',
                'action_norm', 'comfort_max_norm', 'dist_ego2npc', 'safety_distance',
                'comfort_min_dist', 'comfort_max_dist']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['steps'],
            'collision': 1.0 if state['collision'] > 0 else -1.0,
            'progress': state['progress'],       # already in 0 or 1
            'target_progress': info['target_progress'],
            'dist2obst': state['dist2obst'],     # already in 0 or 1
            'target_dist2obst': info['target_dist2obst'],
            'velocity_x': state['velocity_x'],
            'min_velx': info['min_velx'],
            'max_velx': info['max_velx'],
            'last_steering': state['last_actions'][-1][0],
            'comfort_max_steering': info['comfort_max_steering'],
            'action_norm': np.linalg.norm([state['last_actions'][-1] - state['last_actions'][-2]]),
            'comfort_max_norm': info['comfort_max_norm'],
            'dist_ego2npc': state['dist_ego2npc'],
            'safety_distance': info['safety_distance'],
            'comfort_min_dist': info['min_comfort_distance'],
            'comfort_max_dist': info['max_comfort_distance'],
        }
        self._max_episode_len = math.ceil(info['max_steps'] / info["frame_skip"])
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix (for robustness)  # TODO maybe deprecated, check it
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        assert i_init >= 0, "issue with episode prefix"     # let's see if it is fine
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always((collision<=0) and (dist_ego2npc <= safety_distance))"
        safety_rho = monitor_stl_episode(stl_spec=safety_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        #
        target_spec = "eventually(progress>=target_progress)"
        target_rho = monitor_stl_episode(stl_spec=target_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        #
        comfort_steering = "(abs(last_steering)<=comfort_max_steering)"
        comfort_actnorm = "(action_norm<=comfort_max_norm)"
        comfort_mindist = "(dist_ego2npc >= comfort_min_dist)"
        comfort_maxdist = "(dist_ego2npc <= comfort_max_dist)"
        comfort_metrics = []
        for comfort_spec in [comfort_steering, comfort_actnorm, comfort_mindist, comfort_maxdist]:
            comfort_trace = monitor_stl_episode(stl_spec=comfort_spec,
                                                vars=self.monitoring_variables, types=self.monitoring_types,
                                                episode=episode)
            comfort_trace = comfort_trace + [[-1, -1] for _ in
                                             range((self._max_episode_len - len(comfort_trace)))]
            comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
            comfort_metrics.append(comfort_mean)
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * np.mean(comfort_metrics)
        return tot_score
