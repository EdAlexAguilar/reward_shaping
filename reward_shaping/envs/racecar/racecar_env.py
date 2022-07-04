import os
import random
from typing import List

import numpy as np
import racecar_gym
from racecar_gym import SingleAgentScenario
from racecar_gym.envs.gym_api import ChangingTrackSingleAgentRaceEnv
from gym.utils import seeding


class RacecarEnv(ChangingTrackSingleAgentRaceEnv):
    def __init__(self, scenario_files: List[str], order: str = 'sequential', render: bool = False, seed: int = 0):
        scenarios = [SingleAgentScenario.from_spec(path=str(f"{os.path.dirname(__file__)}/config/{sf}"),
                                                           rendering=render) for sf in scenario_files]
        super(RacecarEnv, self).__init__(scenarios=scenarios, order=order)
        self.seed(seed)
        print(seed)

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)


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
