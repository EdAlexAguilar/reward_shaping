import os
import pathlib
from typing import Dict

from gym.utils import seeding
from racecar_gym import SingleAgentRaceEnv, SingleAgentScenario
import gym
import numpy as np

from reward_shaping.envs.racecar.util.configurations import ObservationConfig, ActionConfig, SpecificationsConfig
from reward_shaping.envs.racecar.util.controllers import PDController, SteeringController
from reward_shaping.envs.racecar.util.utils import polar2cartesian, dist_to_wall


class CustomSingleAgentRaceEnv(SingleAgentRaceEnv):
    agent_id: str = 'A'
    metadata = {"render.modes": ["human", "rgb_array"],
                "render.view_modes": ["follow", "birds_eye", "lidar"]}

    def __init__(self, scenario_file: str, l2d_max_range: float, l2d_res: float,
                 min_speed: float, max_speed: float, min_steering: float, max_steering: float, wheel_base: float,
                 norm_speed_limit: float, norm_comf_steering: float, max_halflane: float,
                 comf_dist_to_wall: float, tolerance_margin: float,
                 frame_skip: int, max_steps: int,
                 seed: int = None, eval: bool = False, gui: bool = False):
        # load scenario
        scenario = SingleAgentScenario.from_spec(
            path=pathlib.Path(f"{os.path.dirname(__file__)}/scenarios") / scenario_file,
            rendering=gui
        )
        super().__init__(scenario)
        # modify observation and action spaces
        self.obs_conf = ObservationConfig(l2d_max_range=l2d_max_range, l2d_res=l2d_res, max_halflane=max_halflane)
        self.action_conf = ActionConfig(min_speed=min_speed, max_speed=max_speed, min_steering=min_steering,
                                        max_steering=max_steering, wheel_base=wheel_base)
        self.specs_conf = SpecificationsConfig(norm_speed_limit=norm_speed_limit, norm_comf_steering=norm_comf_steering,
                                               comf_dist_to_wall=comf_dist_to_wall, tolerance_margin=tolerance_margin)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.time_step = 0
        #
        self.observation_space = self._define_new_observation_space()
        self.action_space = self._define_new_action_space()
        # define aux controllers
        # PD tuned following Zigler-Nichols method: Ku=0.80, Tu=0.40 -> Kp=0.8*Ku, Kd=0.1*Ku*Tu
        self.speed_controller = PDController(kp=0.64, kd=0.032)
        self.steering_controller = SteeringController()
        # seeding
        seed = np.random.randint(0, 1000000) if seed is None else seed
        self.seed(seed)

    def seed(self, seed: int):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def _define_new_observation_space(self):
        """ extend the original observation space with 2d-projection of lidar scan, steering and speed commands """
        observation_dict = {
            "velocity": self.observation_space.spaces["velocity"],
            "lidar": self.observation_space.spaces["lidar"],
            "lidar_occupancy": gym.spaces.Box(low=0, high=255, dtype=np.uint8,
                                              shape=(1, self.obs_conf.l2d_bins, self.obs_conf.l2d_bins)),
            "steering": gym.spaces.Box(low=-1, high=+1, shape=(1,)),  # note: they are already norm in +-1
            "speed": gym.spaces.Box(low=-1, high=+1, shape=(1,)),
            "dist_to_wall": gym.spaces.Box(low=-1.0, high=np.PINF, shape=(1,)),  # not norm., assume track width ~1-2 m
            "wall_collision": gym.spaces.Box(low=0.0, high=+1.0, shape=()),
            "wrong_way": gym.spaces.Box(low=0.0, high=+1.0, shape=()),
            "progress": gym.spaces.Box(low=0.0, high=1.0, shape=()),
        }
        return gym.spaces.Dict(observation_dict)

    def _define_new_action_space(self):
        """ change the original action space to control curvature, speed """
        self.original_action_space = self.action_space
        return gym.spaces.Dict({
            "curvature": gym.spaces.Box(low=-1.0, high=+1.0, shape=(1,)),
            "speed": gym.spaces.Box(low=-1.0, high=+1.0, shape=(1,))
        })

    def step(self, action: Dict):
        original_action = self._convert_to_original_action(action)
        # perform step with frame skip
        reward = 0
        for t in range(self.frame_skip):
            self.time_step += 1
            obs, r, done, info = super().step(original_action)
            done = done or self.time_step > self.max_steps
            reward += r
            if done: break
        # extend observation and infos
        steering, speed = original_action["steering"], action["speed"]
        state = self.scenario.world.state()[self.agent_id]
        obs = self._prepare_observation(state, obs, steering, speed)
        info = self._prepare_info(info, reward, done)
        return obs, reward, done, info

    def _convert_to_original_action(self, action):
        """ convert action in original action space by using aux controllers """
        act_conf = self.action_conf
        target_speed = act_conf.min_speed + (act_conf.max_speed - act_conf.min_speed) * (action["speed"] + 1) / 2.0
        target_curvature = act_conf.min_curv + (act_conf.max_curv - act_conf.min_curv) * (action["curvature"] + 1) / 2.0
        original_action = {
            "steering": self._control_curvature(target_curvature),
            "motor": self._control_speed(target_speed)
        }
        return original_action

    def _control_speed(self, target_speed):
        """ this is a simple PID controller to compute the motor force to reach the target speed """
        current_speed = self.scenario.world.state()[self.agent_id]["velocity"][0]
        current_time = self.scenario.world.state()[self.agent_id]["time"]
        motor_force = self.speed_controller.control(target_speed, current_speed, current_time)
        return motor_force

    def _control_curvature(self, target_curvature):
        return self.steering_controller.control(self.action_conf.wheel_base, target_curvature)

    def _prepare_observation(self, state: Dict, obs: Dict, steering_cmd: np.ndarray, speed_cmd: np.ndarray):
        assert "lidar" in obs
        scan, scan_angles = obs["lidar"], np.linspace(np.deg2rad(-135), np.deg2rad(135), len(obs["lidar"]))
        obs = {
            "velocity": obs["velocity"],  # ground truth velocity (unnormalized)
            "lidar": obs["lidar"],
            "lidar_occupancy": polar2cartesian(scan, scan_angles, self.obs_conf.l2d_bins, self.obs_conf.l2d_res),
            "steering": steering_cmd,  # note: they are already norm in +-1
            "speed": speed_cmd,
            "dist_to_wall": np.array([dist_to_wall(scan, self.obs_conf.max_halflane)]),
            "wall_collision": np.array(1.0 if state["wall_collision"] else 0.0),
            "wrong_way": np.array(1.0 if state["wrong_way"] else 0.0),
            "progress": np.array(state["progress"])
        }
        return obs

    def _prepare_info(self, info, default_reward, done):
        # extend original info with more task-specific parameters
        info["norm_speed_limit"] = self.specs_conf.norm_speed_limit
        info["norm_comf_steering"] = self.specs_conf.norm_comf_steering
        info["comf_dist_to_wall"] = self.specs_conf.comf_dist_to_wall
        info["tolerance_margin"] = self.specs_conf.tolerance_margin
        info["norm_max_speed"] = 1.0
        info["norm_max_steering"] = 1.0
        info["progress_target"] = 1.0
        info["default_reward"] = default_reward
        info["max_steps"] = self.max_steps
        info["frame_skip"] = self.frame_skip
        info["done"] = done
        return info

    def reset(self, mode: str = 'grid'):
        self.speed_controller.reset()
        obs = super(CustomSingleAgentRaceEnv, self).reset(mode)
        state = self.scenario.world.state()[self.agent_id]
        steering, speed = np.array([0.0]), np.array([0.0])
        obs = self._prepare_observation(state, obs, steering, speed)
        self.time_step = 0
        return obs

    def render(self, mode: str = 'human', view_mode: str = 'follow',
               width=320, height=240, info={}, **kwargs):
        rgb_array = super(CustomSingleAgentRaceEnv, self).render(mode=view_mode, width=width, height=height, **kwargs)
        return rgb_array if mode == "rgb_array" else True


if __name__ == "__main__":
    import time
    from stable_baselines3.common.env_checker import check_env

    params = {'scenario_file': 'austria.yml', 'frame_skip': 5, 'max_steps': 5000,
              'l2d_max_range': 10.0, 'l2d_res': 0.25, 'max_halflane': 2.0, 'comf_dist_to_wall': 0.40,
              'min_speed': 0.0, 'max_speed': 3.0, 'min_steering': -0.4189, 'max_steering': 0.4189,
              'wheel_base': 0.3205, 'norm_speed_limit': 0.34, 'norm_comf_steering': 0.25}
    env = CustomSingleAgentRaceEnv(**params, gui=True)
    check_env(env)

    print(env.observation_space)
    print(env.action_space)
    for ep in range(10):
        done = False
        obs = env.reset(mode='random')

        action = {"speed": np.array([-0.5]), "curvature": np.array([0.0])}
        i = 0
        t0 = time.time()
        while not done and i < 1000:
            i += params['frame_skip']
            obs, reward, done, state = env.step(action)
            print(obs["dist_to_wall"])
            env.render(mode="human", view_mode="birds_eye")
            # time.sleep(0.01)

        print(time.time() - t0)
    env.close()
