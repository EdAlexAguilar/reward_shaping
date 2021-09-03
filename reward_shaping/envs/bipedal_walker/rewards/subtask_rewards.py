from reward_shaping.core.helper_fns import NormalizedReward
from reward_shaping.core.reward import RewardFunction


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



class ContinuousFalldownReward(RewardFunction):
    """
    always(dist_to_ground >= dist_hull_limit)
    dist_to_ground is estimated with the min lidar ray (note: norm lidar dist are -11:-1, because the last is xpos)
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'lidar' in next_state and 'dist_hull_limit' in info and 'collision' in info
        lidar = min(next_state['lidar']) if not info['collision'] else 0.0  # if no collision, approx dist to ground wt lidars
        return lidar - info['dist_hull_limit']


class BinaryFalldownReward(RewardFunction):
    def __init__(self, falldown_penalty=-1.0, no_falldown_bonus=0.0):
        self._falldown_penalty = falldown_penalty
        self._no_falldown_bonus = no_falldown_bonus

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'collision' in info
        return self._falldown_penalty if info['collision'] else self._no_falldown_bonus


class SpeedTargetReward(RewardFunction):
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        """
        always(v_x >= speed_x_target)
        state[2] = 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
        """
        assert 'horizontal_speed' in next_state
        return next_state['horizontal_speed'] - info['speed_x_target']


class ReachTargetReward(RewardFunction):
    """
    spec := F x >= x_target, score := x - x_target
    x_position in state[24], already normalized in 0..1
    """
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'hull_x' in next_state and 'norm_target_x' in info
        x = next_state['hull_x']
        return x - info['norm_target_x']


class ProgressToTargetReward(RewardFunction):
    def __init__(self, progress_coeff=1.0):
        self._progress_coeff = progress_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        if next_state is not None:
            assert 'hull_x' in state and 'x' in next_state and 'norm_target_x' in info
            x, next_x = state['hull_x'], next_state['hull_x']
            dist_pre = abs(x - info['norm_target_x'])
            dist_now = abs(next_x - info['norm_target_x'])
            tau = 1/50  # 1/fps
            return self._progress_coeff * (dist_pre - dist_now) / tau
        else:
            # it should never happen but for robustness
            return 0.0


class ContinuousHullAngleReward(RewardFunction):
    """
    always(abs(phi) <= angle_hull_limit)
    state[0] = self.hull.angle
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'hull_angle' in next_state and 'angle_hull_limit' in info
        phi = next_state['hull_angle']
        return info['angle_hull_limit'] - abs(phi)


class ContinuousVerticalSpeedReward(RewardFunction):
    """
    always(abs(v_y) <= speed_y_limit
    state[3] = 0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,  #also normalized
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'vertical_speed' in next_state and 'speed_y_limit' in info
        return info['speed_y_limit'] - abs(next_state['vertical_speed'])


class ContinuousHullAngleVelocityReward(RewardFunction):
    """
    always(abs(phi_dot) <= angle_vel_hull_limit)
    state[1] = 2.0 * self.hull.angularVelocity / FPS,
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'hull_angle_speed' in next_state and 'angle_vel_limit' in info
        phi_dot = next_state['hull_angle_speed']
        return info['angle_vel_limit'] - abs(phi_dot)



register_subtask_reward("binary_falldown", BinaryFalldownReward(falldown_penalty=-1.0, no_falldown_bonus=0.0))
register_subtask_reward("continuous_falldown", NormalizedReward(ContinuousFalldownReward(), min_r=0.0, max_r=0.5))
