import numpy as np
from lidarcontrol import pid


class WallFollow:
    """
    Implement Wall Following on the car
    LIDAR layout (Top is forward; +x axis)
                    0
         +90       CAR       -90
            +135        -135
    """
    def __init__(self, target_distance_left=0.4, reference_angle=55,
                 steer_kp=1.0, steer_ki=0.0, steer_kd=0.1,
                 target_velocity=1.5, throttle_kp=1.0, throttle_ki=0.0,
                 throttle_kd=0.05, base_throttle=0.0):
        # LIDAR constants
        self.angle_range = 270  # Hokuyo 10LX has 270 degrees scan
        self.num_scans = 64
        # Wall Follow Controller & Params
        self.target_distance_left = target_distance_left
        self.alpha = reference_angle  # Angle of 2nd ray # Degrees
        self.cos_alpha = np.cos(self.alpha * np.pi / 180)
        self.sin_alpha = np.sin(self.alpha * np.pi / 180)
        self.alpha_index = self.lidar_index(self.alpha)
        self.left_index = self.lidar_index(90)
        self.steer_kp = steer_kp
        self.steer_ki = steer_ki
        self.steer_kd = steer_kd
        self.steer_controller = pid.PID(self.steer_kp,
                                        self.steer_ki,
                                        self.steer_kd,
                                        init_control=0.0)
        # Velocity Controller
        self.target_velocity = target_velocity
        self.throttle_kp = throttle_kp
        self.throttle_ki = throttle_ki
        self.throttle_kd = throttle_kd
        self.base_throttle = base_throttle
        self.throttle_controller = pid.PID(self.throttle_kp,
                                           self.throttle_ki,
                                           self.throttle_kd,
                                           init_control=self.base_throttle)

    def lidar_index(self, angle):
        """
        return integer index of closest lidar angle
        """
        assert angle<=self.angle_range/2 and angle >= - self.angle_range/2
        lidar_angles = -np.arange(-self.angle_range/2, self.angle_range/2, self.angle_range/self.num_scans)
        delta = (lidar_angles[0] - np.abs(lidar_angles[-1]))/2
        lidar_angles = lidar_angles - delta
        index = np.argmin(np.abs(lidar_angles - angle))
        return index

    def clean_read(self, data, index, n=3):
        """
        Reads lidar data from data[index] and returns an average of
        n values of data, centered around index
        """
        return np.mean(data[index-n//2:index+n//2+1])

    def act(self, observation):
        velocity = observation['velocity']
        lidar = observation[f'lidar_{self.num_scans}']
        measured_vel = self.process_vel(velocity)
        throttle = self.throttle_controller.control(self.target_velocity, measured_vel)
        throttle = np.clip(throttle, 0, 1)
        measured_dist = self.process_lidar(lidar)
        steering = self.steer_controller.control(self.target_distance_left, measured_dist)
        steering = np.clip(steering, -1, 1)
        action = {'speed': throttle, 'steering': steering}
        return action

    def process_vel(self, velocity):
        """
        :param velocity: x,y,z and rotational velocities
        :return: planar speed
        """
        return np.linalg.norm([velocity[0], velocity[1]])

    def process_lidar(self, rays):
        # https://f1tenth-coursekit.readthedocs.io/en/latest/assignments/labs/lab3.html
        left_dist = self.clean_read(rays, self.left_index, n=3)
        alpha_dist = self.clean_read(rays, self.alpha_index, n=3)
        theta = np.arctan((alpha_dist * self.cos_alpha - left_dist) / (alpha_dist * self.sin_alpha))  # in radians
        dist_to_wall = left_dist * np.cos(theta)
        naive_predicted_distance = np.sin(theta)  # this might benefit from some multiplicative scalar
        return dist_to_wall + naive_predicted_distance
