from typing import Dict, Tuple

from reward_shaping.envs import RewardFunction


class MinimizeDistanceToLandingArea(RewardFunction):

    def __init__(self, target_position: Tuple[float, float]):
        self._target_position = target_position

    def __call__(self, state: Dict, action=None, next_state=None) -> float:
        assert 'x' in state.keys() and 'y' in state.keys()
        x, y = state['x'], state['y']
        x_target, y_target = self._target_position
        return - ((x - x_target)**2 + (y - y_target)**2)

class AvoidCrashes(RewardFunction):
    def __call__(self, state, action=None, next_state=None) -> float:
        pass

class MinimizeFuelUsage(RewardFunction):
    pass


class MinimizeLandingVelocity(RewardFunction):
    pass

class MinimizeSwinging(RewardFunction):
    pass