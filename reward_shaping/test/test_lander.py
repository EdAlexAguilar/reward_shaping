from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training

env_name = "lunar_lander"
task = "land"


class TestContinuousLunarLander(TestCase):

    def test_eval_reward(self):
        reward = "eval"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_stl_reward(self):
        reward = "stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_chain_reward(self):
        reward = "gb_chain"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_weighted_reward(self):
        reward = "weighted"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_bpdr_ci(self):
        reward = "gb_bpdr_ci"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)


class TestTrainingLoop(TestCase):

    def test_train_sparse(self):
        generic_training(env_name, task, 'default')

    def test_train_stl(self):
        generic_training(env_name, task, 'stl')

    def test_train_chain(self):
        generic_training(env_name, task, 'gb_chain')

    def test_train_weighted(self):
        generic_training(env_name, task, 'weighted')

    def test_train_hierarchical(self):
        generic_training(env_name, task, 'gb_bpdr_ci')
