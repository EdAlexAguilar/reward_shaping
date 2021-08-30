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
s[8], s[9] x,y coordinates of the bottom-left corner of the obstacle
s[10], s[11] x,y coordinates of the top-right corner of the obstacle
s[12] the remaining fuel
s[13] 1 if collision occurred else 0
"""


class MinimizeDistanceToLandingArea(RewardFunction):
    """
    Target : reach origin
    target_requirement = f"eventually(always(dist_origin <= dist_origin_tol))"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        x, y = state[0], state[1]
        x_target, y_target = 0, 0
        return - ((x - x_target) ** 2 + (y - y_target) ** 2)


class ProgressToTargetReward(RewardFunction):
    """
    Target: reward(s, s') = progress_coeff * |distance_{t-1} - distance_{t}|
    """

    def __init__(self, progress_coeff=1.0):
        self._progress_coeff = progress_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x_target' in info and 'y_target' in info
        if next_state is not None:
            x_pre, y_pre = state[0], state[1]
            x, y = next_state[0], next_state[1]
            dist_pre = np.linalg.norm([info['x_target'] - x_pre, info['y_target'] - y_pre])
            dist_now = np.linalg.norm([info['x_target'] - x, info['y_target'] - y])
            return self._progress_coeff * (dist_pre - dist_now)
        else:
            # it should never happen but for robustness
            return 0.0


class SlowLandingReward(RewardFunction):
    """
    Safety: the y velocity should never be such that it crashes
    this can be checked by looking at the next step (y+delta*vel) and requiring it to be positive,
    if negative means that the craft approached the ground with too much velocity.

    no_y_crash = f"always((y+delta*y_dot)>= 0)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        y, y_dot = state[1], state[3]
        delta = 1 / info['FPS']
        return y + delta * y_dot


class BinarySlowLandingReward(RewardFunction):
    """
    as above, requires always((y+delta*y_dot)>= 0)
    but return sparse reward
    """

    def __init__(self, slow_bonus=0.0, crash_penalty=0.0):
        self._slow_bonus = slow_bonus
        self._crash_penalty = crash_penalty

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        y, y_dot = state[1], state[3]
        delta = 1 / info['FPS']
        return self._slow_bonus if y + delta * y_dot >= 0.0 else self._crash_penalty


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
        # todo have they the same scale?
        return info['theta_limit'] - abs(theta)


class FuelReward(RewardFunction):
    """
     Safety Property
     fuel_usage = always(fuel >= 0)
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        return next_state[12]


class BinaryFuelReward(RewardFunction):
    """
     Safety Property
     fuel_usage = always(fuel >= 0)
    """

    def __init__(self, still_fuel_bonus=0.0, no_fuel_penalty=0.0):
        self._still_fuel_bonus = still_fuel_bonus
        self._no_fuel_penalty = no_fuel_penalty

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        fuel = next_state[12]
        return self._still_fuel_bonus if fuel >= 0.0 else self._no_fuel_penalty


class CollisionReward(RewardFunction):
    def __init__(self, collision_penalty=0.0, no_collision_bonus=0.0):
        super().__init__()
        self.collision_penalty = collision_penalty
        self.no_collision_bonus = no_collision_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        collision = next_state[13] == 1
        return self.no_collision_bonus if not collision else self.collision_penalty


class OutsideReward(RewardFunction):
    def __init__(self, exit_penalty=0.0, no_exit_bonus=0.0):
        super().__init__()
        self.exit_penalty = exit_penalty
        self.no_exit_bonus = no_exit_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in next_state and 'x_limit' in info
        x = next_state['x']
        return self.exit_penalty if (abs(x) > info['x_limit']) else self.no_exit_bonus


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
