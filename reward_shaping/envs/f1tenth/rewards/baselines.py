from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_episode


class F110EvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_len = 0

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
            'collision': info['collision'],  # already 0 or 1
            'progress': info['progress']
        }
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix (for robustness)
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always(collision<=0)"
        safety_rho = monitor_episode(stl_spec=safety_spec,
                                     vars=self.monitoring_variables, types=self.monitoring_types,
                                     episode=episode)[0][1]
        #
        target_spec = "eventually(progress >= 1.0)"
        target_rho = monitor_episode(stl_spec=target_spec,
                                     vars=self.monitoring_variables, types=self.monitoring_types,
                                     episode=episode)[0][1]
        #
        comfort_metrics = [0]
        """
        comfort_spec1 = ""
        comfort_spec2 = ""        
        for comfort_spec in [comfort_spec1, comfort_spec2]:
            comfort_trace = monitor_episode(stl_spec=comfort_spec,
                                            vars=self.monitoring_variables, types=self.monitoring_types,
                                            episode=episode)
            comfort_trace = comfort_trace + [[-1, -1] for _ in
                                             range((self._max_episode_len - len(comfort_trace)))]
            comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
            comfort_metrics.append(comfort_mean)
        """
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * np.mean(comfort_metrics)
        return tot_score