from typing import List, Dict, Any

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_episode
from reward_shaping.core.reward import WeightedReward

from reward_shaping.core.utils import get_normalized_reward
import reward_shaping.envs.bipedal_walker.rewards.subtask_rewards as fns


class BWWeightedBaselineReward(WeightedReward):
    """
    reward(s,a) := w_s * sum([score in safeties]) + w_t * sum([score in targets]) + w_c * sum([score in comforts])
    """

    def __init__(self, env_params, safety_weight=1.0, target_weight=0.5, comfort_weight=0.25):
        # parameters
        super().__init__()
        self._safety_weight = safety_weight
        self._target_weight = target_weight
        self._comfort_weight = comfort_weight
        # prepare env info for normalize the functions
        info = {
            "angle_hull_limit": env_params['angle_hull_limit'],
            "speed_y_limit": env_params['speed_y_limit'],
            "angle_vel_limit": env_params['angle_vel_limit'],
            "speed_x_target": env_params['speed_x_target']
        }
        # safety rules
        falldown_fn = fns.get_subtask_reward("binary_falldown")
        # target rules (no need indicators)
        target_fn, _ = get_normalized_reward(fns.SpeedTargetReward(),
                                             min_r_state={'horizontal_speed': info['speed_x_target']},
                                             max_r_state={'horizontal_speed': 1.0},
                                             info=info)
        # comfort rules
        angle_comfort_fn, _ = get_normalized_reward(fns.ContinuousHullAngleReward(),
                                                    min_r_state={'hull_angle': info['angle_hull_limit']},
                                                    max_r_state={'hull_angle': 0.0},
                                                    info=info)
        vert_speed_comfort_fn, _ = get_normalized_reward(fns.ContinuousVerticalSpeedReward(),
                                                         min_r_state={'vertical_speed': info['speed_y_limit']},
                                                         max_r_state={'vertical_speed': 0.0},
                                                         info=info)
        angle_vel_comfort_fn, _ = get_normalized_reward(fns.ContinuousHullAngleVelocityReward(),
                                                        min_r_state={'hull_angle_speed': info['angle_vel_limit']},
                                                        max_r_state={'hull_angle_speed': 0.0},
                                                        info=info)

        self._safety_rules = [falldown_fn]
        self._target_rules = [target_fn]
        self._comfort_rules = [angle_comfort_fn, vert_speed_comfort_fn, angle_vel_comfort_fn]


class BWEvalConfig(EvalConfig):
    """
        Eval(phi1,w) = sign(rho(phi1, w)) safety
        Eval(phi2,w) = 0.5*sign(rho(phi2, w)) reach target

        phi3 = G phi' Eval(phi3, w) = 0.25*(\Sum_i sign(rho(phi',w,i)))/n comfort
        Eval(phi2', w) = 0.5*(\Sum_i sign(rho(phi',w,i)))/n recurrence target

        Eval(Phi, w) = Eval(phi1, w) + Eval(phi2, w) + Eval(phi3, w)
        """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_episode_len = 0

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'vx', 'phi', 'vy', 'phidot', 'phi_limit', 'vy_limit', 'phidot_limit']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        # compute monitoring variables (all of them normalized in 0,1)
        monitored_state = {
            'time': info['time'],
            'collision': info['collision'],  # already 0 or 1
            'vx': state['horizontal_speed'],
            'phi': state['hull_angle'],
            'phi_limit': info['angle_hull_limit'],
            'vy': state['vertical_speed'],
            'vy_limit': info['speed_y_limit'],
            'phidot': state['hull_angle_speed'],
            'phidot_limit': info['angle_vel_limit']
        }
        self._max_episode_len = info['max_episode_len']
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: l[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always(collision<=0)"
        safety_rho = monitor_episode(stl_spec=safety_spec,
                                     vars=self.monitoring_variables, types=self.monitoring_types,
                                     episode=episode)[0][1]
        recurrence_spec = "vx>=0.0"
        recurrence_trace = monitor_episode(stl_spec=recurrence_spec,
                                           vars=self.monitoring_variables, types=self.monitoring_types,
                                           episode=episode)
        recurrence_trace = recurrence_trace + [[-1, -1]  for _ in  range((self._max_episode_len - len(recurrence_trace)))]
        recurrence_mean = np.mean([float(rob >= 0) for t, rob in recurrence_trace])

        comfort_spec = "(abs(phi) <= phi_limit) and (abs(vy) <= vy_limit) and (abs(phidot) <= phidot_limit)"
        comfort_trace = monitor_episode(stl_spec=comfort_spec,
                                        vars=self.monitoring_variables, types=self.monitoring_types,
                                        episode=episode)
        comfort_trace = comfort_trace + [[-1, -1]  for _ in  range((self._max_episode_len - len(comfort_trace)))]
        comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
        tot_score = float(safety_rho >= 0) + 0.5 * recurrence_mean + 0.25 * comfort_mean
        return tot_score
