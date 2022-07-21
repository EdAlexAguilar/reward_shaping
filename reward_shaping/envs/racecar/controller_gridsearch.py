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
from sklearn.model_selection import ParameterGrid

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

def evaluate_agent(env, wf_params, num_trials=30):
    agent_a = WallFollow(**wf_params)
    progress = []
    for _ in range(num_trials):
        env.reset()
        done = False
        action_a = {'speed': 0.0, 'steering': 0.0}
        action_b = {'speed': -1.0, 'steering': 0.0}
        actions = {'A': action_a, 'B': action_b}
        obss, rewards, dones, infos = env.step(actions)
        init_progress_a = infos['A']['progress'] + float(infos['A']['lap'])
        while not done:
            action_a = agent_a.act(obss['A'])
            actions = {'A': action_a, 'B': action_b}
            obss, rewards, dones, infos = env.step(actions)
            done = any(dones.values())
            if done:
                fin_progress_a = infos['A']['progress'] + float(infos['A']['lap'])
                progress.append(fin_progress_a - init_progress_a)
    return np.array(progress)

if __name__ == "__main__":
    scenario_files = ["treitlstrasse_multi_agent.yml", "treitlstrasse_multi_agent.yml"]
    env = MultiAgentRacecarEnv(scenario_files, render=True)
    best_progress = 0.0
    wall_follow_params = {'target_distance_left': [0.4],
                          'reference_angle': [50, 55, 60],
                          'steer_kp': [0.7, 0.9, 1.0],
                          'steer_ki': [0.0],
                          'steer_kd': [0.05, 0.1, 0.15],
                          'target_velocity': [1],
                          'throttle_kp': [0.8, 1.0, 1.1],
                          'throttle_ki': [0.0],
                          'throttle_kd': [0.05, 0.1],
                          'base_throttle': [0.0]}
    for ii, controller_params in enumerate(list(ParameterGrid(wall_follow_params))):
        progress = evaluate_agent(env, controller_params)
        if np.mean(progress) >= best_progress:
            best_progress = np.mean(progress)
            print(f'New Best Controller: {ii} :  Mean Progress: {best_progress:.3f}')
            print(controller_params)
            print('\n')
    env.close()
    print("done")
'''
    wall_follow_params = {'target_distance_left':0.4,
                        'reference_angle':55,
                        'steer_kp':1.0,
                        'steer_ki':0.0,
                        'steer_kd':0.1,
                        'target_velocity':1,
                        'throttle_kp':1.0,
                        'throttle_ki':0.0,
                        'throttle_kd':0.05,
                        'base_throttle':0.0}
'''
    # progress = evaluate_agent(env, wall_follow_params)
