from reward_shaping.core.helper_fns import ThresholdIndicator, NormalizedReward
from reward_shaping.core.reward import RewardFunction
import numpy as np

_registry = {}


def get_subtask_reward(name: str):
    try:
        reward = _registry[name]
    except KeyError:
        raise KeyError(f"the reward {name} is not registered")
    return reward


def register_subtask_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


class MinimizeDistanceToLandingArea(RewardFunction):
    """
    Target : reach origin
    target_requirement = f"eventually(always(dist_origin <= dist_origin_tol))"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in next_state
        assert 'y' in next_state
        assert 'x_target' in info
        assert 'y_target' in info
        assert 'halfwidth_landing_area' in info
        x, y = next_state['x'], next_state['y']
        x_target, y_target = info['x_target'], info['y_target']
        dist = np.linalg.norm([x - x_target, y - y_target])
        return info['halfwidth_landing_area'] - dist


class ProgressToOriginReward(RewardFunction):
    """
    Target: reward(s, s') = progress_coeff * |distance_{t-1} - distance_{t}|
    assume target is the origin
    """

    def __init__(self, progress_coeff=1.0):
        self._progress_coeff = progress_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'FPS' in info
        if next_state is not None:
            x_pre, y_pre = state['x'], state['y']
            x, y = next_state['x'], next_state['y']
            dist_pre = np.linalg.norm([x_pre, y_pre])
            dist_now = np.linalg.norm([x, y])
            return self._progress_coeff * (dist_pre - dist_now) * info['FPS']
        else:
            # it should never happen but for robustness
            return 0.0


class ProgressTimesDistanceToTargetReward(RewardFunction):
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in next_state and 'y' in next_state
        assert 'x_target' in info and 'y_target' in info
        assert'halfwidth_landing_area' in info and 'FPS' in info
        if next_state is not None:
            # note: the normalization factor 1.5 is to have distance in [0,1]
            dist_pre = np.linalg.norm([state['x'], state['y']]) / 1.5
            dist = np.linalg.norm([next_state['x'], next_state['y']]) / 1.5
            # note: to ensure velocity in the x scale (and not too small), rescale it with factor x_limit
            velocity = np.clip((dist_pre - dist) * info['FPS'], 0.0, 1.0)
            dist = np.clip(dist, 0.0, 1.0)
            assert 0.0 <= dist <= 1.0 and 0.0 <= velocity <= 1.0, f'dist={dist}, velocity={velocity}'
            return (1 - dist) + dist * velocity
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

    def __init__(self, slow_bonus=0.0, crash_penalty=0.0, **kwargs):
        super().__init__(**kwargs)
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
        angle = state['angle']
        angle_limit = info['angle_limit']
        return angle_limit - abs(angle)


class FuelReward(RewardFunction):
    """
     Safety Property
     fuel_usage = always(fuel >= 0)
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        return next_state['fuel']


class BinaryFuelReward(RewardFunction):
    """
     Safety Property
     fuel_usage = always(fuel >= 0)
    """

    def __init__(self, still_fuel_bonus=0.0, no_fuel_penalty=0.0, **kwargs):
        super().__init__(**kwargs)
        self._still_fuel_bonus = still_fuel_bonus
        self._no_fuel_penalty = no_fuel_penalty

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        fuel = next_state['fuel']
        return self._still_fuel_bonus if fuel > 0.0 else self._no_fuel_penalty


class CollisionReward(RewardFunction):
    def __init__(self, collision_penalty=0.0, no_collision_bonus=0.0):
        super().__init__()
        self.collision_penalty = collision_penalty
        self.no_collision_bonus = no_collision_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        collision = next_state['collision'] == 1
        return self.no_collision_bonus if not collision else self.collision_penalty


class ContinuousCollisionReward(RewardFunction):
    """
    see cartpole for explaination of the math
    """

    def __call__(self, state, action, next_state, info):
        assert 'x' in next_state and 'y' in next_state
        assert 'obstacle_left' in next_state and 'obstacle_right' in next_state
        assert 'obstacle_bottom' in next_state and 'obstacle_top' in next_state
        x, y = next_state['x'], next_state['y']
        rho = max(-(x - next_state['obstacle_left']), -(next_state['obstacle_right'] - x),
                  -(y - next_state['obstacle_bottom']), -(next_state['obstacle_top'] - y))
        return rho


class OutsideReward(RewardFunction):
    """ spec := |x| <= x_limit, score := x_limit - |x| """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x' in next_state and 'x_limit' in info
        return info['x_limit'] - abs(next_state['x'])


class BinaryOutsideReward(RewardFunction):
    def __init__(self, exit_penalty=0.0, no_exit_bonus=0.0):
        super().__init__()
        self.exit_penalty = exit_penalty
        self.no_exit_bonus = no_exit_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'x_limit' in info
        return self.exit_penalty if (abs(next_state['x']) > info['x_limit']) else self.no_exit_bonus


class MinimizeAngleVelocity(RewardFunction):
    """
     # Comfort : Small Angle Velocity
        angular_velocity = f"always(abs(theta_dot) <= theta_dot_limit)"
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        angle_speed = next_state['angle_speed']
        angle_speed_limit = info['angle_speed_limit']
        return angle_speed_limit - abs(angle_speed)


register_subtask_reward("binary_fuel", BinaryFuelReward(no_fuel_penalty=-1.0, still_fuel_bonus=0.0))
register_subtask_reward("continuous_fuel", NormalizedReward(FuelReward(), min_reward=0.0, max_reward=1.0))
register_subtask_reward("fuel_sat", ThresholdIndicator(FuelReward(), include_zero=False))

register_subtask_reward("binary_collision", CollisionReward(no_collision_bonus=0.0, collision_penalty=-1.0))
register_subtask_reward("continuous_collision", NormalizedReward(ContinuousCollisionReward(),
                                                                 min_reward=0.0, max_reward=1.0))
register_subtask_reward("collision_sat",
                        ThresholdIndicator(CollisionReward(no_collision_bonus=0.0, collision_penalty=-1.0)))

register_subtask_reward("binary_exit", BinaryOutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0))
register_subtask_reward("continuous_exit", NormalizedReward(OutsideReward(), min_reward=0.0, max_reward=1.0))
register_subtask_reward("exit_sat", ThresholdIndicator(BinaryOutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0)))

register_subtask_reward("continuous_progress", NormalizedReward(ProgressToOriginReward(progress_coeff=1.0),
                                                                min_reward=0.0, max_reward=1.0))
register_subtask_reward("progress_x_distance", ProgressTimesDistanceToTargetReward())
register_subtask_reward("target_sat", ThresholdIndicator(MinimizeDistanceToLandingArea()))
