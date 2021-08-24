import numpy as np

from reward_shaping.core.reward import RewardFunction


class ContinuousReward(RewardFunction):
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
        dist_target = abs(x - info['x_target'])
        pole_x, pole_y = x + info['pole_length'] * np.sin(theta), \
                         info['axle_y'] + info['pole_length'] * np.cos(theta)
        obst_x, obst_y = obst_left + (obst_right - obst_left) / 2.0, \
                         obst_bottom + (obst_top - obst_bottom) / 2.0
        dist_obst = np.sqrt((obst_x - pole_x) ** 2 + (obst_y - pole_y) ** 2)
        return - dist_target + dist_obst


class SparseReward(RewardFunction):
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
