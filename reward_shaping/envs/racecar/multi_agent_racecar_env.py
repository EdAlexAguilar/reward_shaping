import collections
import os
from typing import List, Dict
from lidarcontrol.follow_wall import WallFollow

import gym
import numpy as np
import racecar_gym
from gym.spaces import Box
from racecar_gym import SingleAgentScenario, MultiAgentScenario
from racecar_gym.envs.gym_api import ChangingTrackMultiAgentRaceEnv
from wrappers.racecar_wrappers import Multi2SingleEnv


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

def play_random_agent(env, num_trials=30):
    progress = []
    for _ in range(num_trials):
        env.reset()
        # action = env.action_space.sample()
        action = {'speed': -1.0, 'steering': 0.0}
        obs, reward, done, info = env.step(action)
        init_progress= info['progress'] + float(info['lap'])
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                fin_progress = info['progress'] + float(info['lap'])
                progress.append(fin_progress - init_progress)
    return np.array(progress)

if __name__ == "__main__":
    wall_follow_params = {'target_distance_left': 0.4,
                          'reference_angle': 55,
                          'steer_kp': 0.9,
                          'steer_ki': 0.0,
                          'steer_kd': 0.1,
                          'target_velocity': 1,
                          'throttle_kp': 1.1,
                          'throttle_ki': 0.0,
                          'throttle_kd': 0.1,
                          'base_throttle': -0.5}
    npc_controller = WallFollow(**wall_follow_params)
    scenario_files = ["treitlstrasse_multi_agent.yml", "treitlstrasse_multi_agent.yml"]
    env = MultiAgentRacecarEnv(scenario_files, render=True)
    env = Multi2SingleEnv(env, npc_controller=npc_controller)
    progress = play_random_agent(env, num_trials=10)
    print(f"Environment ran with random agent. Avg Track Progress: {np.mean(progress):.3f}")
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)  >> complains about actions being a dict
    env.close()
    print("done")
