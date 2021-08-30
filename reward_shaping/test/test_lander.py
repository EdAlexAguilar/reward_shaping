from unittest import TestCase

from reward_shaping.test.test import generic_env_test

env_name = "lunar_lander"

class TestLunarLanderObstacle(TestCase):

    def test_landing_default(self):
        task = "land"
        reward = "default"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_landing_stl(self):
        task = "land"
        reward = "stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_landing_gb_progress(self):
        task = "land"
        reward = "gb_cr_bi"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)