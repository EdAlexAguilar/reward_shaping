import random
from typing import List, Callable, Dict

import gym
import numpy as np
from racecar_gym.envs.util.subprocess_env import SubprocessEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from reward_shaping.envs.racecar.single_agent_env import CustomSingleAgentRaceEnv
from reward_shaping.envs.racecar.wrappers import FixResetWrapper
from reward_shaping.training.utils import load_env_params


class ChangingTrackRaceEnv(gym.Env):

    def __init__(self, env_factories: List[Callable[[], gym.Env]], order: str = 'sequential'):
        super().__init__()
        self._current_track_index = 0
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(env_factories)
        elif order == 'random':
            self._order_fn = lambda: random.choice(
                list(set(range(0, len(env_factories))) - {self._current_track_index}))
        elif order == 'manual':
            self._order_fn = lambda: self._current_track_index
        self._order = order

        self._envs = [
            SubprocessEnv(factory=factory, blocking=True)
            for factory
            in env_factories
        ]
        assert all(self._envs[0].action_space == env.action_space for env in self._envs)
        assert all(self._envs[0].observation_space == env.observation_space for env in self._envs)
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space

    def step(self, action):
        return self._get_env().step(action=action)

    def reset(self, mode: str = 'grid'):
        self._current_track_index = self._order_fn()
        return self._get_env().reset(mode=mode)

    def render(self, mode, **kwargs):
        return self._get_env().render(mode, **kwargs)

    def close(self):
        for env in self._envs:
            env.close()

    def _get_env(self):
        return self._envs[self._current_track_index]

    def set_next_env(self):
        assert self._order == 'manual'
        self._current_track_index = (self._current_track_index + 1) % len(self._envs)


class ChangingTrackSingleAgentRaceEnv(ChangingTrackRaceEnv):

    def __init__(self, scenario_files: List[str], order: str = 'sequential', **kwargs):
        env_factories = [self._make_factory(scenario_file=s, **kwargs) for s in scenario_files]
        super().__init__(env_factories, order)

    @property
    def scenario(self):
        return None

    def step(self, action: Dict):
        return super(ChangingTrackSingleAgentRaceEnv, self).step(action=action)

    def _make_factory(self, scenario_file: str, **kwargs) -> Callable[[], CustomSingleAgentRaceEnv]:
        def factory():
            return CustomSingleAgentRaceEnv(scenario_file=scenario_file, **kwargs)

        return factory


if __name__ == "__main__":
    params = load_env_params(env="racecar", task="drive_vectorized")
    env = ChangingTrackSingleAgentRaceEnv(**params, gui=True)
    env = FixResetWrapper(env, mode="random")
    print("checking")
    check_env(env)
    print("checked")

    import time

    print(env.observation_space)
    print(env.action_space)
    for ep in range(10):
        done = False
        obs = env.reset()

        action = {"speed": np.array([-0.5]), "curvature": np.array([0.0])}
        i = 0
        t0 = time.time()
        while not done and i < 1000:
            i += params['frame_skip']
            obs, reward, done, state = env.step(action)
            #env.render(mode="human", view_mode="birds_eye")
            # time.sleep(0.01)

        print(time.time() - t0)
    env.close()
