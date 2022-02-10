from dataclasses import dataclass
import numpy as np


@dataclass
class ActionConfig:

    def __init__(self, min_speed: float, max_speed: float, min_steering: float, max_steering: float, wheel_base: float):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_steering = min_steering
        self.max_steering = max_steering
        self.wheel_base = wheel_base
        self.min_curv: float = np.tan(self.min_steering) / wheel_base
        self.max_curv: float = np.tan(self.max_steering) / wheel_base


@dataclass
class ObservationConfig:

    def __init__(self, l2d_max_range: float, l2d_res: float, max_halflane: float):
        self.l2d_max_range = l2d_max_range
        self.l2d_res = l2d_res
        self.max_halflane = max_halflane
        self.l2d_bins: int = int(2 * l2d_max_range / l2d_res)


@dataclass
class SpecificationsConfig:

    def __init__(self, norm_speed_limit: float, norm_comf_steering: float,
                 comf_dist_to_wall: float, tolerance_margin: float):
        self.norm_speed_limit = norm_speed_limit
        self.norm_comf_steering = norm_comf_steering
        self.comf_dist_to_wall = comf_dist_to_wall
        self.tolerance_margin = tolerance_margin