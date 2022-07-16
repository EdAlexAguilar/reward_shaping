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
        self._min_speed_cmd = params["reward_params"]["min_speed_cmd"]
        self._max_speed_cmd = params["reward_params"]["max_speed_cmd"]
        self._max_steps = params["max_steps"]
        self._steps = 0
        self._initial_progress = None
        self._n_last_actions = params["n_last_actions"]
        self._last_actions = collections.deque([[0.0] * len(self.action_space)] * self._n_last_actions,
                                               maxlen=self._n_last_actions)

        # extend observation space (we need them to compute the potential, the agent do not directly observe them)
        self.observation_space["progress"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["dist2obst"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["collision"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        minvel, maxvel = self.observation_space["velocity"].low[0], self.observation_space["velocity"].high[0]
        self.observation_space["velocity_x"] = Box(low=minvel, high=maxvel, shape=(1,))
        self.observation_space["last_actions"] = Box(low=-1, high=+1,
                                                     shape=(self._n_last_actions, len(self.action_space)))

        self._eval = params["eval"]
        self._seed = params["seed"]
        self.seed(self._seed)

    @staticmethod
    def _get_params(**kwargs):
        params = {
            "max_steps": 600,
            "n_last_actions": 3,
            "reward_params": {
                "target_progress": 0.99,
                "target_dist2obst": 0.5,
                "min_speed_cmd": -1.0,
                "max_speed_cmd": -1.0,
            },
            "render": False,
            "eval": False,
            "seed": 0,
        }
        for k, v in kwargs.items():
            params[k] = v
        return params

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        obs = super(RacecarEnv, self).reset(mode='grid' if self._eval else 'random')
        # extend observations
        self._last_actions = collections.deque([[0.0] * len(self.action_space)] * self._n_last_actions,
                                               maxlen=self._n_last_actions)
        self._initial_progress = None
        dummy_info = {"wall_collision": False, "progress": None, "obstacle": 1.0}
        obs = self._extend_obs(obs, dummy_info)
        self._steps = 0
        return obs

    def step(self, action: Dict):
        obs, reward, done, info = super(RacecarEnv, self).step(action)
        flat_action = np.array([action["steering"][0], action["speed"][0]])
        self._last_actions.append(flat_action)
        self._steps += 1
        obs = self._extend_obs(obs, info)
        info = self._extend_info(reward, done, info)
        done = self._check_termination(obs, done, info)
        return obs, reward, done, info

    def _extend_obs(self, obs, info):
        if self._initial_progress is None and info["progress"] is not None:
            # update the initial-progress on the first available progress after reset
            self._initial_progress = info["progress"]
        progress = 0.0 if self._initial_progress is None else info["progress"] - self._initial_progress
        obs["collision"] = float(info["wall_collision"])
        obs["progress"] = progress
        obs["dist2obst"] = info["obstacle"]
        obs["velocity_x"] = np.array([obs["velocity"][0]], dtype=np.float32)
        obs["last_actions"] = np.array(self._last_actions, dtype=np.float32)
        return obs

    def _extend_info(self, reward, done, info):
        info["default_reward"] = reward
        info["target_progress"] = self._target_progress
        info["target_dist2obst"] = self._target_dist2obst
        info["min_speed_cmd"] = self._min_speed_cmd
        info["max_speed_cmd"] = self._max_speed_cmd
        info["steps"] = self._steps
        info["max_steps"] = self._max_steps
        info["done"] = done
        return info

    def _check_termination(self, obs, done, info):
        collision = info["wall_collision"]
        lap_completion = info["progress"] >= self._target_progress
        timeout = self._steps >= self._max_steps
        return bool(done or collision or lap_completion or timeout)

    def render(self, mode):
        view_mode = "follow"
        screen = super(RacecarEnv, self).render(mode=view_mode)
        if mode == "rgb_array":
            return screen


if __name__ == "__main__":
    scenario_files = ["treitlstrasse_single_agent.yml", "treitlstrasse_single_agent.yml"]
    env = RacecarEnv(scenario_files, render=True)
    from reward_shaping.envs.wrappers import FilterObservationWrapper, NormalizeObservationWithMinMax
    from gym.wrappers import FlattenObservation

    fields = ["lidar_64", "velocity_x", "last_actions"]
    env = FilterObservationWrapper(env, fields)
    env = NormalizeObservationWithMinMax(env, {"lidar_64": (0.0, 15.0),  # norm lidar rays from 0, 15 meters
                                               "velocity_x": (0.0, 3.5),  # norm valocity from 0, 3.5 m/s
                                               "last_actions": (-1.0, 1.0)  # norm actions in +-1
                                               })
    env = FlattenObservation(env)

    for _ in range(5):
        env.reset()
        done = False
        tot_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_reward += reward
        print(tot_reward)

    env.close()
    print("done")