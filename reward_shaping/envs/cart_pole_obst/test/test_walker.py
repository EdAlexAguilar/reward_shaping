from unittest import TestCase
from reward_shaping.envs.cart_pole_obst.test.test import generic_env_test

env_name = "bipedal_walker"


class TestBipedalWalker(TestCase):

    def test_default_reward(self):
        task = "forward"
        reward = "default"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_stl(self):
        task = "forward"
        reward = "stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_bool_stl(self):
        task = "forward"
        reward = "bool_stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_cr_bi(self):
        task = "forward"
        reward = "gb_cr_bi"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_weighted(self):
        task = "forward"
        reward = "weighted"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)


