from typing import Dict, Any

import numpy as np

import reward_shaping.envs.lunar_lander.rewards.subtask_rewards as fns
from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_episode
from reward_shaping.core.reward import WeightedReward, RewardFunction
from reward_shaping.core.utils import get_normalized_reward


class LLSparseTargetReward(RewardFunction):
    """
    reward(s,a) := bonus, if target is reached
    reward(s,a) := small time penalty
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'halfwidth_landing_area' in info
        assert 'x' in next_state and 'y' in next_state
        dist_target = np.linalg.norm([state["x"], state["y"]])
        time_cost = 1 / info["max_steps"]
        if dist_target <= info["halfwidth_landing_area"]:
            return +1.0
        return -time_cost


class LLEvalConfig(EvalConfig):
    def __init__(self, **kwargs):
        super(LLEvalConfig, self).__init__(**kwargs)
        self._max_episode_len = 0

    @property
    def monitoring_variables(self):
        return ['time', 'x', 'x_limit',
                'angle', 'angle_speed', 'angle_limit', 'angle_speed_limit',
                'fuel', 'collision', 'dist_target', 'halfwidth_landing_area']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        target_dist = np.linalg.norm([state["x"], state["y"]])
        # compute monitoring variables
        monitored_state = {
            'time': info['time'],
            'x': state['x'],  # already normalized in +-1
            'x_limit': info['x_limit'],
            'angle': state['angle'],  # already normalized in +-1
            'angle_speed': state['angle_speed'],  # already normalized in +-1
            'angle_limit': info['angle_limit'],
            'angle_speed_limit': info['angle_speed_limit'],
            'fuel': info['fuel'],  # in [0,1]
            'collision': info['collision'],
            'dist_target': target_dist,
            'halfwidth_landing_area': info['halfwidth_landing_area']
        }
        self._max_episode_len = info["max_steps"]
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: l[i_init:] for k, l in episode.items()}
        # safety
        safety_spec = "always((collision <= 0.0) and (abs(x) <= x_limit))"
        safety_rho = monitor_episode(stl_spec=safety_spec, vars=self.monitoring_variables,
                                     types=self.monitoring_types, episode=episode)[0][1]
        # persistence
        target_spec = "eventually(always(dist_target <= halfwidth_landing_area))"
        target_rho = monitor_episode(stl_spec=target_spec, vars=self.monitoring_variables,
                                     types=self.monitoring_types, episode=episode)[0][1]
        # comfort
        comfort_spec = "(abs(angle) <= angle_limit) and (abs(angle_speed) <= angle_speed_limit)"
        comfort_trace = monitor_episode(stl_spec=comfort_spec, vars=self.monitoring_variables,
                                        types=self.monitoring_types, episode=episode)
        comfort_trace = comfort_trace + [[-1, -1] for _ in range(self._max_episode_len - len(comfort_trace))]
        comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
        # total score
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * comfort_mean
        return tot_score


class LLWeightedBaselineReward(WeightedReward):
    """
    reward(s,a) := w_s * sum([score in safeties]) + w_t * sum([score in targets]) + w_c * sum([score in comforts])
    """

    def __init__(self, env_params, safety_weight=1.0, target_weight=0.5, comfort_weight=0.25):
        # parameters
        super().__init__()
        self._env_params = env_params
        self._safety_weight = safety_weight
        self._target_weight = target_weight
        self._comfort_weight = comfort_weight
        # prepare env info for normalize the functions
        info = {'FPS': self._env_params['FPS'],
                'angle_limit': self._env_params['angle_limit'],
                'angle_speed_limit': self._env_params['angle_speed_limit'],
                "x_target": self._env_params['x_target'],
                "y_target": self._env_params['y_target'],
                "halfwidth_landing_area": self._env_params['halfwidth_landing_area'],
                }

        # safety rules (no need returned indicators)
        binary_collision = fns.get_subtask_reward("binary_collision")
        binary_exit = fns.get_subtask_reward("binary_exit")

        # target rules
        progress_fn = fns.get_subtask_reward("continuous_progress")

        # comfort rules
        angle_limit = self._env_params['angle_limit']
        angle_fn, _ = get_normalized_reward(fns.MinimizeCraftAngle(),
                                            min_r_state={'angle': angle_limit},
                                            max_r_state={'angle': 0.0},
                                            info=info)
        angle_speed_limit = self._env_params['angle_speed_limit']
        angle_speed_fn, _ = get_normalized_reward(fns.MinimizeAngleVelocity(),
                                                  min_r_state={'angle_speed': angle_speed_limit},
                                                  max_r_state={'angle_speed': 0.0},
                                                  info=info)
        # comfort rules
        self._safety_rules = [binary_collision, binary_exit]
        self._target_rules = [progress_fn]
        self._comfort_rules = [angle_fn, angle_speed_fn]
