from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_episode
from reward_shaping.core.reward import RewardFunction


class RacecarMinActionReward(RewardFunction):

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert all(abs(a) <= 1 for a in action)
        if info["collision"] > 0:
            reward = -1.0
        else:
            reward = 1 - (1 / len(action) * np.linalg.norm(action) ** 2)
        return reward


class RacecarEvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'reverse', 'progress', 'velocity', 'steering',
                'comfortable_steering', 'speed_limit', 'lap']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['time'],
            'collision': 1.0 if info['wall_collision'] else -1.0,
            'reverse': 1.0 if info['wrong_way'] else -1.0,
            'progress': info['progress'],
            'velocity': state['velocity'][0],
            'steering': state['steering'],
            # 'lane': state['lane'],
            'comfortable_steering': info['norm_comf_steering'],
            'speed_limit': info['norm_speed_limit'],
            # 'favourite_lane': info['favourite_lane'],
            'lap': info['lap']
        }
        self._max_episode_len = info['max_steps'] // info['frame_skip']
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix (for robustness)
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always((collision<=0) and (reverse <= 0))"
        safety_rho = monitor_episode(stl_spec=safety_spec,
                                     vars=self.monitoring_variables, types=self.monitoring_types,
                                     episode=episode)[0][1]
        #
        target_spec = "eventually(lap >= 1.0)"
        target_rho = monitor_episode(stl_spec=target_spec,
                                     vars=self.monitoring_variables, types=self.monitoring_types,
                                     episode=episode)[0][1]
        #
        comfort_metrics = []
        comfort_speed = "(velocity <= speed_limit)"
        comfort_steer = "(abs(steering) <= comfortable_steering)"
        # comfort_lane = "(lane == favourite_lane)"
        for comfort_spec in [comfort_speed, comfort_steer]:
            comfort_trace = monitor_episode(stl_spec=comfort_spec,
                                            vars=self.monitoring_variables, types=self.monitoring_types,
                                            episode=episode)
            comfort_trace = comfort_trace + [[-1, -1] for _ in
                                             range((self._max_episode_len - len(comfort_trace)))]
            comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
            comfort_metrics.append(comfort_mean)
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * np.mean(comfort_metrics)
        return tot_score