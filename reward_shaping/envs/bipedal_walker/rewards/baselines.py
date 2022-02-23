from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_stl_episode


class BWEvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'x',
                'target_x', 'vx', 'phi', 'vy', 'phidot', 'phi_limit',
                'vy_limit', 'phidot_limit', 'vx_target']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
                'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['time'],
            'collision': 1.0 if info['collision'] > 0 else -1.0,  # already 0 or 1
            'x': state['x'],  # already in 0 or 1
            'target_x': info['norm_target_x'],
            'vx': state['horizontal_speed'],
            'vx_target': info['speed_x_target'],
            'phi': state['hull_angle'],
            'phi_limit': info['angle_hull_limit'],
            'vy': state['vertical_speed'],
            'vy_limit': info['speed_y_limit'],
            'phidot': state['hull_angle_speed'],
            'phidot_limit': info['angle_vel_limit']
        }
        self._max_episode_len = info['max_steps']
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix (for robustness)
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always(collision<=0)"
        safety_rho = monitor_stl_episode(stl_spec=safety_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        #
        target_spec = "eventually(x>=target_x)"
        target_rho = monitor_stl_episode(stl_spec=target_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        #
        comfort_vel_spec = "(vx>=vx_target)"
        comfort_ang_spec = "(abs(phi)<=phi_limit)"
        comfort_vy_spec = "(abs(vy)<=vy_limit)"
        comfort_angvel_spec = "(abs(phidot)<=phidot_limit)"
        comfort_metrics = []
        for comfort_spec in [comfort_vel_spec, comfort_ang_spec, comfort_vy_spec, comfort_angvel_spec]:
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
