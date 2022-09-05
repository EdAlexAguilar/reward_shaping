import collections
import os
import random
from typing import List, Dict, Any

import gym
import numpy as np
import racecar_gym
from gym.spaces import Box
from racecar_gym import SingleAgentScenario
from racecar_gym.envs.gym_api import ChangingTrackSingleAgentRaceEnv
from gym.utils import seeding

from reward_shaping.envs.wrappers import ActionHistoryWrapper, DeltaSpeedWrapper, ObservationHistoryWrapper


class RacecarEnv(ChangingTrackSingleAgentRaceEnv):
    def __init__(self,
                 scenario_files: List[str],
                 order: str = 'sequential',
                 **kwargs):
        # make race environment
        params = self._get_params(**kwargs)
        scenarios = [SingleAgentScenario.from_spec(path=str(f"{os.path.dirname(__file__)}/config/{sf}"),
                                                   rendering=params["render"]) for sf in scenario_files]
        super(RacecarEnv, self).__init__(scenarios=scenarios, order=order)

        # spec params
        self._target_progress = params["reward_params"]["target_progress"]
        self._target_dist2obst = params["reward_params"]["target_dist2obst"]
        self._comfort_max_steering = params["reward_params"]["comfort_max_steering"]
        self._comfort_max_norm = params["reward_params"]["comfort_max_norm"]
        self._min_velx = params["reward_params"]["min_velx"]
        self._max_velx = params["reward_params"]["max_velx"]
        self._limit_velx = params["action_config"]["max_velx"]
        self._max_steps = params["max_steps"]
        self._frame_skip = params["frame_skip"]
        self._steps = 0
        self._initial_progress = None

        # extend observation space (we need them to compute the potential, the agent do not directly observe them)
        self.observation_space["progress"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["dist2obst"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["collision"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        minvel, maxvel = self.observation_space["velocity"].low[0], self.observation_space["velocity"].high[0]
        self.observation_space["velocity_x"] = Box(low=minvel, high=maxvel, shape=(1,))

        self._eval = params["eval"]
        self._seed = params["seed"]
        self.seed(self._seed)

    @staticmethod
    def _get_params(**kwargs):
        params = {
            "max_steps": 2500,
            "n_last_actions": 3,
            "reward_params": {
                "target_progress": 0.99,
                "target_dist2obst": 0.5,
                "min_velx": 2.0,
                "max_velx": 3.0,
                "comfort_max_steering": 0.1,
                "comfort_max_norm": 1.0,
            },
            "action_config": {
                "delta_speed": False,
                "min_velx": 0.0,
                "max_velx": +3.5,
                "cap_min_velx": 0.0,
                "cap_max_velx": +3.5,
                "max_accx": 4.0,
                "dt": 0.01,
            },
            "frame_skip": 1,
            "render": True,
            "eval": False,
            "seed": 0,
        }
        for k, v in kwargs.items():
            params[k] = v
        return params

    def reset(self):
        obs = super(RacecarEnv, self).reset(mode='grid' if self._eval else 'random')
        self._initial_progress = None
        dummy_info = {"wall_collision": False, "progress": None, "obstacle": 1.0, "lap": 1}
        obs = self._extend_obs(obs, dummy_info)
        self._steps = 0
        return obs

    def step(self, action: Dict):
        obs, reward, done, info = super(RacecarEnv, self).step(action)
        self._steps += 1
        obs = self._extend_obs(obs, info)
        done = self._check_termination(obs, done, info)
        info = self._extend_info(reward, done, info)
        return obs, reward, done, info

    def _extend_obs(self, obs, info):
        if self._initial_progress is None and info["progress"] is not None:
            # update the initial-progress on the first available progress after reset
            self._initial_progress = info["progress"]
        progress = 0.0 if self._initial_progress is None else (info["lap"] - 1) + (info["progress"] - self._initial_progress)
        obs["collision"] = float(info["wall_collision"])
        obs["progress"] = progress
        obs["dist2obst"] = info["obstacle"]
        obs["velocity_x"] = np.array([obs["velocity"][0]], dtype=np.float32)
        return obs

    def _extend_info(self, reward, done, info):
        info["default_reward"] = reward
        info["target_progress"] = self._target_progress
        info["target_dist2obst"] = self._target_dist2obst
        info["comfort_max_steering"] = self._comfort_max_steering
        info["comfort_max_norm"] = self._comfort_max_norm
        info["min_velx"] = self._min_velx
        info["max_velx"] = self._max_velx
        info["limit_velx"] = self._limit_velx
        info["steps"] = self._steps
        info["max_steps"] = self._max_steps
        info["frame_skip"] = self._frame_skip
        info["done"] = done
        return info

    def _check_termination(self, obs, done, info):
        collision = info["wall_collision"]
        lap_completion = obs["progress"] >= self._target_progress
        timeout = self._steps >= self._max_steps
        return bool(done or collision or lap_completion or timeout)

    def render(self, mode):
        view_mode = "birds_eye"
        screen = super(RacecarEnv, self).render(mode=view_mode)
        if mode == "rgb_array":
            return screen


if __name__ == "__main__":

    scenario_files = ["treitlstrasse_single_agent.yml", "treitlstrasse_single_agent.yml"]
    scenario_files = ["treitlstrasse_single_agent.yml"]
    action_config = {"min_velx": 0.0, "max_velx": 3.5,
                     "cap_min_speed": 1.0, "cap_max_speed": 3.5,
                     "max_accx": 4.0, "dt": 0.01}

    env = RacecarEnv(scenario_files, render=True, eval=True)
    env = ActionHistoryWrapper(env, n_last_actions=3)
    env = DeltaSpeedWrapper(env, action_config=action_config, frame_skip=1)
    env = ObservationHistoryWrapper(env, obs_name="lidar_64", n_last_observations=3)

    for _ in range(1):

        speeds = []
        cmds = []
        norms = []

        env.reset()
        done = False
        tot_reward = 0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            speeds.append(obs["velocity_x"])
            cmds.append(env.speed_ms)
            steps += 1
            norms.append(np.linalg.norm(obs["last_actions"][-1] - obs["last_actions"][-2]))
        print(tot_reward)
    env.close()

    import matplotlib.pyplot as plt
    plt.plot(speeds)
    plt.plot(cmds)
    plt.plot(norms)
    plt.show()

    print("done")
