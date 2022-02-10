from typing import List, Tuple, Dict

import gym
import numpy as np
from racecar_gym.envs.util.vectorized_race import VectorizedRaceEnv

from reward_shaping.envs.racecar.single_agent_env import CustomSingleAgentRaceEnv
from reward_shaping.training.utils import load_env_params


class VectorizedCustomSingleAgentRaceEnv(gym.Env):
    metadata = {'render.modes': ['follow', 'birds_eye', 'lidar']}

    def __init__(self, scenario_files: List[str], **kwargs):
        self._env = VectorizedRaceEnv(factories=[lambda: CustomSingleAgentRaceEnv(s, **kwargs) for s in scenario_files])
        self.action_space, self.observation_space = self._env.action_space, self._env.observation_space

    def step(self, actions: Tuple[Dict]):
        return self._env.step(actions=actions)

    def reset(self, mode: str = 'grid'):
        return self._env.reset(mode=mode)

    def close(self):
        self._env.close()

    def render(self, mode='follow', **kwargs):
        return self._env.render(mode=mode, **kwargs)


if __name__ == "__main__":
    params = load_env_params(env="racecar", task="drive")
    env = VectorizedCustomSingleAgentRaceEnv(**params, gui=False)
    import time

    print(env.observation_space)
    print(env.action_space)
    for ep in range(10):
        done = False
        obs = env.reset(mode='grid')

        action = {"speed": np.array([-0.5]), "curvature": np.array([0.0])}
        i = 0
        t0 = time.time()
        while not done and i < 1000:
            i += params['frame_skip']
            obs, reward, done, state = env.step(action)
            # env.render(mode="birds_eye")
            # time.sleep(0.01)

        print(time.time() - t0)
    env.close()
