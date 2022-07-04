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

        self._eval = eval
        self._seed = seed
        self.seed(seed)

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def step(self, action: Dict):
        obs, reward, done, info = super(RacecarEnv, self).step(action)
        info["default_reward"] = reward
        done = done or self._check_termination(obs, info)
        return obs, reward, done, info

    def _check_termination(self, obs, info):
        collision = info["wall_collision"]
        lap_completion = info["progress"] >= self._target_progress
        return bool(collision or lap_completion)

    def reset(self):
        return super(RacecarEnv, self).reset(mode='grid' if self._eval else 'random')


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
