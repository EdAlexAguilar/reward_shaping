import os
import random
from typing import List, Dict

import numpy as np
import racecar_gym
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
        rendering = render or eval
        scenarios = [SingleAgentScenario.from_spec(path=str(f"{os.path.dirname(__file__)}/config/{sf}"),
                                                   rendering=rendering) for sf in scenario_files]
        super(RacecarEnv, self).__init__(scenarios=scenarios, order=order)

        # spec params
        self._target_progress = target_progress
        self._target_dist2obst = target_dist2obst
        self._max_steps = max_steps
        self._steps = 0

        self._eval = eval
        self._seed = seed
        self.seed(seed)

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action: Dict):
        obs, reward, done, info = super(RacecarEnv, self).step(action)
        self._steps += 1
        info = self._extend_info(info, reward)
        done = self._check_termination(obs, info, done)
        return obs, reward, done, info

    def _extend_info(self, info, reward):
        info["default_reward"] = reward
        info["target_progress"] = self._target_progress
        info["target_dist2obst"] = self._target_dist2obst
        info["steps"] = self._steps
        info["max_steps"] = self._max_steps
        return info

    def _check_termination(self, obs, info, done):
        collision = info["wall_collision"]
        lap_completion = info["progress"] >= self._target_progress
        timeout = self._steps >= self._max_steps
        return bool(done or collision or lap_completion or timeout)

    def reset(self):
        obs = super(RacecarEnv, self).reset(mode='grid' if self._eval else 'random')
        self._steps = 0
        return obs


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
