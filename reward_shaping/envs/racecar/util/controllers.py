from dataclasses import dataclass

from numba import njit
import numpy as np


@dataclass
class Gains:
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0


class PDController:
    def __init__(self, kp: float, kd: float):
        self._gains = Gains(Kp=kp, Kd=kd)
        self._err = 0.0
        self._time = 0.0

    @staticmethod
    @njit(fastmath=False, cache=True)
    def fast_ctrl(target_value, current_value, timestamp, pre_err, pre_time, kp, kd):
        err = target_value - current_value
        derr_dt = (err - pre_err) / (timestamp - pre_time)
        ctrl = kp * err + kd * derr_dt
        # sanity check
        ctrl = np.array([-1.0]) if ctrl < -1.0 else ctrl  # clip operation for numba
        ctrl = np.array([+1.0]) if ctrl > 1.0 else ctrl
        return ctrl, err

    def control(self, target_value, current_value, timestamp):
        ctrl, err = self.fast_ctrl(target_value, current_value, timestamp, self._err, self._time,
                                   self._gains.Kp, self._gains.Kd)
        # update internal variables
        self._err = err
        self._time = timestamp
        # return pid control
        return ctrl

    def reset(self):
        self._err = 0.0
        self._time = 0.0


class SteeringController:
    @staticmethod
    @njit(fastmath=False, cache=True)
    def control(wheel_base, target_curvature):
        """ this is a naive method to compute steering angle from a target curvature """
        steering = np.arctan(wheel_base * target_curvature)
        min_steering, max_steering = -0.4189, +0.4189
        steering = -1.0 + 2.0 * (steering - min_steering) / (max_steering - min_steering)  # rescale it in -1,+1
        # sanity check
        steering = np.array([-1.0]) if steering < -1.0 else steering  # clip operation for numba
        steering = np.array([1.0]) if steering > 1.0 else steering
        return steering
