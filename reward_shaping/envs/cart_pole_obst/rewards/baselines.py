from typing import List

import numpy as np

from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_stl_episode
from reward_shaping.core.reward import RewardFunction


class CPOContinuousReward(RewardFunction):
    """
    reward(s,a) := - dist_target + dist_obst
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x_target' in info
        assert 'pole_length' in info
        assert 'axle_y' in info
        x, theta = next_state['x'], next_state['theta']
        obst_left, obst_right = next_state['obstacle_left'], next_state['obstacle_right']
        obst_bottom, obst_top = next_state['obstacle_bottom'], next_state['obstacle_top']
        dist_target = abs(x - info['x_target']) / abs(info['x_limit'] - info['x_target'])
        pole_x, pole_y = x + info['pole_length'] * np.sin(theta), \
                         info['axle_y'] + info['pole_length'] * np.cos(theta)
        obst_x, obst_y = obst_left + (obst_right - obst_left) / 2.0, \
                         obst_bottom + (obst_top - obst_bottom) / 2.0
        dist_obst = 1 / 10 * np.sqrt((obst_x - pole_x) ** 2 + (obst_y - pole_y) ** 2)
        return 5.0 * (1 - dist_target) - (1 - dist_obst)


class CPOSparseReward(RewardFunction):
    """
    reward(s,a) := penalty, if collision or falldown
    reward(s,a) := bonus, if target is reached
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x_limit' in info and 'theta_limit' in info
        assert 'x_target' in info and 'x_target_tol' in info
        x, theta, collision = next_state['x'], next_state['theta'], next_state['collision']
        if abs(state['x'] - info['x_target']) <= info['x_target_tol'] and \
                abs(state['theta']) <= info["theta_target_tol"]:
            return +1.0
        if abs(theta) > info['theta_limit'] or abs(x) > info['x_limit'] or collision:
            return -1.0
        return 0.0


class CPOSparseTargetReward(RewardFunction):
    """
    reward(s,a) := bonus, if target is reached
    reward(s,a) := small time penalty
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x_limit' in info and 'theta_limit' in info
        assert 'x_target' in info and 'x_target_tol' in info
        x, theta, collision = next_state['x'], next_state['theta'], next_state['collision']
        time_cost = 1 / info["max_steps"]
        falldown = (abs(theta) > info["theta_limit"])
        outside = (abs(x) > info["x_limit"])
        collision = (collision > 0)
        if falldown or outside or collision:
            return -1
        if abs(x - info['x_target']) <= info['x_target_tol'] and abs(theta) <= info["theta_target_tol"]:
            return +1.0
        return -time_cost


class CPOProgressTargetReward(RewardFunction):
    """
    reward(s, a, s') := target(s') - target(s), if target is reached
    reward(s, a, s') := small time penalty
    """

    def target_potential(self, state, info):
        x, theta, collision = state['x'], state['theta'], state['collision']
        pole_x, pole_y = x + info['pole_length'] * np.sin(theta), \
                         info['axle_y'] + info['pole_length'] * np.cos(theta)
        goal_x, goal_y = info['x_target'], info['axle_y'] + info['pole_length']
        dist_goal = np.linalg.norm([goal_x - pole_x, goal_y - pole_y])
        target_reward = 1 - np.clip(dist_goal, 0, 2.5) / 2.5
        return target_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x_limit' in info and 'theta_limit' in info
        assert 'x_target' in info and 'x_target_tol' in info
        progress = self.target_potential(next_state, info) - self.target_potential(state, info)
        return progress


class CPOEvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super(CPOEvalConfig, self).__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self) -> List[str]:
        return ['time',
                'x', 'x_limit', 'x_target', 'x_target_tol',
                'theta', 'theta_limit', 'theta_target', 'theta_target_tol',
                'collision', 'dist_target_x', 'dist_target_theta']

    @property
    def monitoring_types(self) -> List[str]:
        return ['int',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float']

    def get_monitored_state(self, state, done, info):
        monitored_state = {
            'time': info['time'],
            'x': state['x'],
            'x_limit': info['x_limit'],
            'x_target': info['x_target'],
            'x_target_tol': info['x_target_tol'],
            'theta': state['theta'],
            'theta_limit': info['theta_limit'],
            'theta_target': info['theta_target'],
            'theta_target_tol': info['theta_target_tol'],
            'collision': 1.0 if info['collision'] > 0 else -1.0,
            'dist_target_x': abs(state['x'] - info['x_target']),
            'dist_target_theta': abs(state['theta'] - info['theta_target']),
        }
        self._max_episode_len = info['max_steps']
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always((abs(theta) <= theta_limit) and (abs(x) <= x_limit) and (collision <= 0.0))"
        safety_rho = monitor_stl_episode(stl_spec=safety_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        target_spec = "eventually(always(dist_target_x <= x_target_tol))"
        target_rho = monitor_stl_episode(stl_spec=target_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        comfort_spec = "dist_target_theta <= theta_target_tol"
        comfort_trace = monitor_stl_episode(stl_spec=comfort_spec,
                                            vars=self.monitoring_variables, types=self.monitoring_types,
                                            episode=episode)
        comfort_trace = comfort_trace + [[-1, -1] for _ in range((self._max_episode_len - len(comfort_trace)))]
        comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * comfort_mean
        return tot_score
