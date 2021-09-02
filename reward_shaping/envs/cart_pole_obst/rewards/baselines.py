import numpy as np

from reward_shaping.core.configs import RewardConfig
from reward_shaping.core.reward import RewardFunction, WeightedReward
from reward_shaping.core.utils import get_normalized_reward
import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns


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
        if abs(x - info['x_target']) <= info['x_target_tol']:
            return +10.0
        if abs(theta) > info['theta_limit'] or abs(x) > info['x_limit'] or collision:
            return -10.0
        return 0.0


class CPOWeightedBaselineReward(WeightedReward):
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
        info = {'x_limit': env_params['x_limit'],
                'x_target': env_params['x_target'],
                'x_target_tol': env_params['x_target_tol'],
                'theta_limit': np.deg2rad(env_params['theta_limit']),
                'theta_target': np.deg2rad(env_params['theta_target']),
                'theta_target_tol': np.deg2rad(env_params['theta_target_tol'])}

        # safety rules (no need returned indicators)
        collision_fn, _ = get_normalized_reward(fns.ContinuousCollisionReward(), min_r=-0.05, max_r=1.0, info=info)

        falldown_fn, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                               min_r_state={'theta': info['theta_limit']},
                                               max_r_state={'theta': 0.0},
                                               info=info)
        outside_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                              min_r_state={'x': info['x_limit']},
                                              max_r_state={'x': 0.0},
                                              info=info)
        # target rules
        progress_fn, _ = get_normalized_reward(fns.ProgressToTargetReward(progress_coeff=1.0),
                                               min_r=-1.0, max_r=1.0)

        # comfort rules
        balance_fn, _ = get_normalized_reward(fns.BalanceReward(),
                                              min_r_state={'theta': info['theta_target']} - info['theta_target_tol'],
                                              max_r_state={'theta': info['theta_target']},
                                              info=info)
        # comfort rules
        self._safety_rules = [collision_fn, falldown_fn, outside_fn]
        self._target_rules = [progress_fn]
        self._comfort_rules = [balance_fn]
