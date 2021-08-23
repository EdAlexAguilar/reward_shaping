from unittest import TestCase

from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env, make_reward_wrap
from reward_shaping.training.wrappers import FlattenFloatObservation


class TestCartPoleObstacle(TestCase):

    def _generic_test(self, task, reward_name):
        env_name = "cart_pole_obst"
        env, env_params = make_env(env_name, task, logdir=None, seed=0, prob_sampling_feasible=1.0)
        env = make_reward_wrap(env_name, env, reward_name)
        env = FlattenObservation(env)
        # check
        check_env(env)
        # evaluation
        obs = env.reset()
        env.render()
        rewards = []
        tot_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            env.render()
        print(f"[{reward_name}] tot reward: {tot_reward:.3f}")
        env.close()

    def test_fixedheight_sparse(self):
        task = "fixed_height"
        reward = "sparse"
        self._generic_test(task, reward)

    def test_fixedheight_continuous(self):
        task = "fixed_height"
        reward = "continuous"
        self._generic_test(task, reward)

    def test_fixedheight_stl(self):
        task = "fixed_height"
        reward = "stl"
        self._generic_test(task, reward)

    def test_fixedheight_boolstl(self):
        task = "fixed_height"
        reward = "bool_stl"
        self._generic_test(task, reward)

