import os
import random
from typing import List, Dict

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
                 target_progress: float = 1.0,
                 target_dist2obst: float = 0.5,
                 max_steps: int = 1200,
                 render: bool = False,
                 eval: bool = False,
                 seed: int = 0):
        # make race environment
        scenarios = [SingleAgentScenario.from_spec(path=str(f"{os.path.dirname(__file__)}/config/{sf}"),
                                                   rendering=render) for sf in scenario_files]
        super(RacecarEnv, self).__init__(scenarios=scenarios, order=order)

        # extend observation space (we need them to compute the potential, the agent do not directly observe them)
        self.observation_space["progress"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["dist2obst"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["collision"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        minvel, maxvel = self.observation_space["velocity"].low[0], self.observation_space["velocity"].high[0]
        self.observation_space["velocity_x"] = Box(low=minvel, high=maxvel, shape=(1,))

        # spec params
        self._target_progress = target_progress
        self._target_dist2obst = target_dist2obst
        self._max_steps = max_steps
        self._steps = 0
        self._initial_progress = None

        self._eval = eval
        self._seed = seed
        self.seed(seed)

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        obs = super(RacecarEnv, self).reset(mode='grid' if self._eval else 'random')
        # extend observations
        self._initial_progress = None
        dummy_info = {"wall_collision": False, "progress": None, "obstacle": 1.0}
        obs = self._extend_obs(obs, dummy_info)
        self._steps = 0
        return obs

    def step(self, action: Dict):
        obs, reward, done, info = super(RacecarEnv, self).step(action)
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
        obs["velocity_x"] = obs["velocity"][0]
        return obs

    def _extend_info(self, reward, done, info):
        info["default_reward"] = reward
        info["target_progress"] = self._target_progress
        info["target_dist2obst"] = self._target_dist2obst
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
