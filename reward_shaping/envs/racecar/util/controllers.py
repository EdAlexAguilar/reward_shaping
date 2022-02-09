from dataclasses import dataclass

from numba import njit


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
        return kp * err + kd * derr_dt, err

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