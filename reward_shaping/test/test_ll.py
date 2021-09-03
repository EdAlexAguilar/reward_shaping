from unittest import TestCase

from reward_shaping.test.test import generic_env_test

env_name = "lunar_lander"


class TestDiscreteLunarLander(TestCase):

    def test_eval_reward(self):
        task = "land"
        reward = "eval"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_stl_reward(self):
        task = "land"
        reward = "stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_chain_reward(self):
        task = "land"
        reward = "gb_chain"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_weighted_reward(self):
        task = "land"
        reward = "weighted"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_bpr_bi(self):
        task = "land"
        reward = "gb_bpr_bi"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_bpr_ci(self):
        task = "land"
        reward = "gb_bpr_ci"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_bpdr_ci(self):
        task = "land"
        reward = "gb_bpdr_ci"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)
