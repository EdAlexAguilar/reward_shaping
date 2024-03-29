from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_stl_episode


class LLEvalConfig(EvalConfig):
    def __init__(self, **kwargs):
        super(LLEvalConfig, self).__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self):
        return ['time', 'x', 'y', 'x_limit',
                'angle', 'angle_speed', 'angle_limit', 'angle_speed_limit',
                'fuel', 'collision', 'halfwidth_landing_area', 'landing_height']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'x': state['x'],  # already normalized in +-1
            'y': state['y'],
            'x_limit': info['x_limit'],
            'angle': state['angle'],  # already normalized in +-1
            'angle_speed': state['angle_speed'],  # already normalized in +-1
            'angle_limit': info['angle_limit'],
            'angle_speed_limit': info['angle_speed_limit'],
            'fuel': info['fuel'],  # in [0,1]
            'collision': 1.0 if info['collision'] > 0 else -1.0,
            'halfwidth_landing_area': info['halfwidth_landing_area'],
            'landing_height': info['landing_height']
        }
        self._max_episode_len = info["max_steps"]
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always((collision <= 0.0) and (abs(x) <= x_limit))"
        safety_rho = monitor_stl_episode(stl_spec=safety_spec, vars=self.monitoring_variables,
                                         types=self.monitoring_types, episode=episode)[0][1]
        #
        target_spec = "eventually(always((abs(x) <= halfwidth_landing_area) and (abs(y) <= landing_height)))"
        target_rho = monitor_stl_episode(stl_spec=target_spec, vars=self.monitoring_variables,
                                         types=self.monitoring_types, episode=episode)[0][1]
        #
        comfort_ang_spec = "(abs(angle) <= angle_limit)"
        comfort_angspeed_spec = "(abs(angle_speed) <= angle_speed_limit)"
        comfort_metrics = []
        for comfort_spec in [comfort_ang_spec, comfort_angspeed_spec]:
            comfort_trace = monitor_stl_episode(stl_spec=comfort_spec, vars=self.monitoring_variables,
                                                types=self.monitoring_types, episode=episode)
            comfort_trace = comfort_trace + [[-1, -1] for _ in range(self._max_episode_len - len(comfort_trace))]
            comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
            comfort_metrics.append(comfort_mean)
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * np.mean(comfort_mean)
        return tot_score
