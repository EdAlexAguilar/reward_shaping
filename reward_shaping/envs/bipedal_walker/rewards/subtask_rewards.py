import numpy as np

from reward_shaping.core.reward import RewardFunction


class ContinuousFalldownReward(RewardFunction):
    """
    always(min(lidar) >= dist_hull_limit)
    last 10 points of state are lidar points
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert len(state) == 24
        assert 'dist_hull_limit' in info
        lidar = min(state[-10:])
        return lidar - info['dist_hull_limit']


class BinaryFalldownReward(RewardFunction):

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'collision' in info
        return 0.0 if info['collision'] else 1.0


class SpeedTargetReward(RewardFunction):
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        """
        always(v_x >= speed_x_target)
        state[2] = 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
        """
        return state[2] - info['speed_x_target']


class ContinuousHullAngleReward(RewardFunction):
    """
    always(abs(phi) <= angle_hull_limit)
    state[0] = self.hull.angle
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'angle_hull_limit' in info
        phi = state[0]
        return info['angle_hull_limit'] - abs(phi)


class ContinuousVerticalSpeedReward(RewardFunction):
    """
    always(abs(v_y) <= speed_y_limit
    state[3] = 0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,  #also normalized
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'speed_y_limit' in info
        return info['speed_y_limit'] - abs(state[3])


class ContinuousHullAngleVelocityReward(RewardFunction):
    """
    always(abs(phi_dot) <= angle_vel_hull_limit)
    state[1] = 2.0 * self.hull.angularVelocity / FPS,
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'angle_vel_limit' in info
        phi_dot = state[1]
        return info['angle_vel_limit'] - abs(phi_dot)
