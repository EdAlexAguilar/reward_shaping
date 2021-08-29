from typing import Dict, Tuple

from reward_shaping.core.reward import RewardFunction
import numpy as np

"""
State
s[0] x is the horizontal coordinate
s[1] y is the vertical coordinate
s[2] x_dot is the horizontal speed
s[3] y_dot is the vertical speed
s[4] theta is the angle
s[5] theta_dot is the angular speed
s[6] 1 if first leg has contact, else 0
s[7] 1 if second leg has contact, else 0
"""


class MinimizeDistanceToLandingArea(RewardFunction):
    """
    # Target : reach origin
    target_requirement = f"eventually(always(dist_origin <= dist_origin_tol))"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        x, y = state[0], state[1]
        x_target, y_target = 0, 0
        return - ((x - x_target) ** 2 + (y - y_target) ** 2)


class MinimizeYVelocity(RewardFunction):
    """
     # Safety 1 : the y velocity should never be such that it crashes
        no_y_crash = f"always((y+delta*y_dot)>= 0)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        y, y_dot = state[1], state[3]
        delta = 1 / info['FPS']
        return y + delta * y_dot


class MinimizeXVelocity(RewardFunction):
    """
    # Comfort 1: Small Horizontal Speed (same as for no_y_crash)
        horizontal_speed = f"always(sign_x*(x+delta*x_dot)>= 0)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        x, x_dot = state[0], state[2]
        delta = 1 / info['FPS']
        return np.sign(x) * (x + delta * x_dot)


class MinimizeCraftAngle(RewardFunction):
    """
     # Safety 2 : Theta angle should be bounded
        spacecraft_angle = f"always(abs(theta) < theta_limit)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        theta = state[4]
        return info['theta_limit'] - abs(theta)


class MinimizeFuelConsumption(RewardFunction):
    """
     Safety Property
     fuel_usage = f"always(fuel >= 0)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        return 1 - info['fuel']


class MinimizeAngleVelocity(RewardFunction):
    """
     # Comfort : Small Angle Velocity
        angular_velocity = f"always(abs(theta_dot) <= theta_dot_limit)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        theta_dot = state[5]
        return info['theta_dot_limit'] - abs(theta_dot)


class AvoidCrashes(RewardFunction):
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        pass
