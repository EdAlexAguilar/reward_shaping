import os
import pathlib
from typing import Dict

from numba import njit
from racecar_gym import SingleAgentRaceEnv, SingleAgentScenario
import gym
import numpy as np

from reward_shaping.envs.racecar_gym.configurations import ObservationConfig, ActionConfig
from reward_shaping.envs.racecar_gym.controllers import PDController
from reward_shaping.envs.racecar_gym.utils import polar2cartesian


class CustomSingleAgentRaceEnv(SingleAgentRaceEnv):
    agent_id: str = 'A'

    def __init__(self, scenario_filename: str, l2d_max_range: float, l2d_res: float,
                 min_speed: float, max_speed: float, min_steering: float, max_steering: float, wheel_base: float,
                 gui: bool = False):
        # load scenario
        scenario = SingleAgentScenario.from_spec(
            path=pathlib.Path(f"{os.path.dirname(__file__)}/config") / scenario_filename,
            rendering=gui
        )
        super().__init__(scenario)
        # modify observation and action spaces
        self.obs_conf = ObservationConfig(l2d_max_range=l2d_max_range, l2d_res=l2d_res)
        self.action_conf = ActionConfig(min_speed=min_speed, max_speed=max_speed, min_steering=min_steering,
                                        max_steering=max_steering, wheel_base=wheel_base)
        self.observation_space = self._define_new_observation_space()
        self.action_space = self._define_new_action_space()
        # define aux controllers
        # PD tuned following Zigler-Nichols method: Ku=0.80, Tu=0.40 -> Kp=0.8*Ku, Kd=0.1*Ku*Tu
        self.speed_controller = PDController(kp=0.64, kd=0.032)

    def _define_new_observation_space(self):
        """ extend the original observation space with 2d-projection of lidar scan """
        observation_dict = {}
        for k, space in self.observation_space.spaces.items():
            observation_dict[k] = space
        observation_dict["lidar_occupancy"] = gym.spaces.Box(low=0, high=255, dtype=np.uint8,
                                                             shape=(1, self.obs_conf.l2d_bins, self.obs_conf.l2d_bins))
        return gym.spaces.Dict(observation_dict)

    def _define_new_action_space(self):
        """ change the original action space to control curvature, speed """
        self.original_action_space = self.action_space
        return gym.spaces.Dict({
            "curvature": gym.spaces.Box(low=-1.0, high=+1.0, shape=(1,)),
            "speed": gym.spaces.Box(low=-1.0, high=+1.0, shape=(1,))
        })

    def step(self, action: Dict):
        # convert action in original action space
        act_conf = self.action_conf
        target_speed = act_conf.min_speed + (act_conf.max_speed - act_conf.min_speed) * (action["speed"] + 1) / 2.0
        target_curvature = act_conf.min_curv + (act_conf.max_curv - act_conf.min_curv) * (action["curvature"] + 1) / 2.0
        original_action = {
            "steering": self._control_curvature(act_conf.wheel_base, target_curvature),
            "motor": self._control_speed(target_speed)
        }
        # perform step
        obs, reward, done, info = super().step(original_action)
        # extend observation
        scan, scan_angles = obs["lidar"], np.linspace(np.deg2rad(-135), np.deg2rad(135), len(obs["lidar"]))
        obs["lidar_occupancy"] = polar2cartesian(scan, scan_angles, self.obs_conf.l2d_bins, self.obs_conf.l2d_res)
        # return
        return obs, reward, done, info

    def _control_speed(self, target_speed):
        """ this is a simple PID controller to compute the motor force to reach the target speed """
        current_speed = self.scenario.world.state()[self.agent_id]["velocity"][0]
        current_time = self.scenario.world.state()[self.agent_id]["time"]
        motor_force = self.speed_controller.control(target_speed, current_speed, current_time)
        # sanity check
        motor_force = np.clip(motor_force, -1.0, +1.0)  # scale it in the original env range
        return motor_force

    @staticmethod
    @njit(fastmath=False, cache=True)
    def _control_curvature(wheel_base, target_curvature):
        """ this is a naive method to compute steering angle from a target curvature """
        steering = np.arctan(wheel_base * target_curvature)
        min_steering, max_steering = -0.4189, +0.4189
        steering = -1.0 + 2.0 * (steering - min_steering) / (max_steering - min_steering)  # rescale it in -1,+1
        # sanity check
        steering = -1 if steering < -1 else steering  # clip operation for numba
        steering = +1 if steering > 1 else steering
        return steering

    def reset(self, mode: str = 'grid'):
        self.speed_controller.reset()
        return super(CustomSingleAgentRaceEnv, self).reset(mode)


if __name__ == "__main__":
    from reward_shaping.training.utils import load_env_params

    params = load_env_params(env="racecar_gym", task="drive")
    env = CustomSingleAgentRaceEnv(**params)
    import time

    print(env.observation_space)
    print(env.action_space)
    for ep in range(10):
        done = False
        obs = env.reset(mode='grid')

        action = {"speed": -0.5, "curvature": 0.0}
        i = 0
        t0 = time.time()
        while not done and i < 1000:
            i += 1
            obs, reward, done, state = env.step(action)
            # env.render(mode="birds_eye")
            # time.sleep(0.01)

        print(time.time() - t0)
    env.close()
