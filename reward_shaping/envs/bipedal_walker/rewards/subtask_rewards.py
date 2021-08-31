from reward_shaping.core.reward import RewardFunction


class ContinuousFalldownReward(RewardFunction):
    """
    always(dist_to_ground >= dist_hull_limit)
    dist_to_ground is estimated with the min lidar ray (note: last 10 points of state are normalized lidar distances)
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert len(state) == 25 and 'dist_hull_limit' in info
        lidar = min(state[-10:])
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
        return state[2] - info['speed_x_target']


class ReachTargetReward(RewardFunction):
    """
    spec := F x >= x_target, score := x - x_target
    x_position in state[24], already normalized in 0..1
    """
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'norm_target_x' in info
        x = next_state[24]
        return x - info['norm_target_x']


class ProgressToTargetReward(RewardFunction):
    def __init__(self, progress_coeff=1.0):
        self._progress_coeff = progress_coeff

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'norm_target_x' in info
        if next_state is not None:
            x, next_x = state[24], next_state[24]
            dist_pre = abs(x - info['norm_target_x'])
            dist_now = abs(next_x - info['norm_target_x'])
            return self._progress_coeff * (dist_pre - dist_now)
        else:
            # it should never happen but for robustness
            return 0.0


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
