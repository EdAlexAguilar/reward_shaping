import collections
import os
from typing import List, Dict

import gym
import numpy as np
import racecar_gym
from gym.spaces import Box
from racecar_gym import SingleAgentScenario, MultiAgentScenario
from racecar_gym.envs.gym_api import ChangingTrackMultiAgentRaceEnv


class MultiAgentRacecarEnv(ChangingTrackMultiAgentRaceEnv):
    def __init__(self,
                 scenario_files: List[str],
                 order: str = 'sequential',
                 **kwargs):
        # make race environment
        params = self._get_params(**kwargs)
        scenarios = [MultiAgentScenario.from_spec(path=str(f"{os.path.dirname(__file__)}/config/{sf}"),
                                                  rendering=params["render"]) for sf in scenario_files]
        super(MultiAgentRacecarEnv, self).__init__(scenarios=scenarios, order=order)

        # spec params
        self._max_steps = params["max_steps"]
        self._steps = 0
        self._initial_progress = None

        self._eval = params["eval"]
        self._seed = params["seed"]
        self.seed(self._seed)

    @staticmethod
    def _get_params(**kwargs):
        params = {
            "max_steps": 600,
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
        obs = super(MultiAgentRacecarEnv, self).reset(mode='grid' if self._eval else 'random_ball')
        self._initial_progress = None
        self._steps = 0
        return obs

    def step(self, action: Dict):
        obs, reward, done, info = super(MultiAgentRacecarEnv, self).step(action)
        self._steps += 1
        return obs, reward, done, info

    def render(self, mode):
        view_mode = "follow"
        screen = super(MultiAgentRacecarEnv, self).render(mode=view_mode)
        if mode == "rgb_array":
            return screen


if __name__ == "__main__":
    scenario_files = ["treitlstrasse_multi_agent.yml", "treitlstrasse_multi_agent.yml"]
    env = MultiAgentRacecarEnv(scenario_files, render=True)

    for _ in range(5):
        env.reset()
        done = False
        while not done:
            actions = env.action_space.sample()
            obss, rewards, dones, infos = env.step(actions)
            done = any(dones.values())   # since many agents, we need to aggregate the termination condition

    env.close()
    print("done")
