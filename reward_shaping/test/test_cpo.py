import time
from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training, plot_cpole_reward, plot_cpole_progreward

env_name = "cart_pole_obst"


class TestCartPoleObstacle(TestCase):

    def test_fixedheight_sparse(self):
        task = "fixed_height"
        reward = "sparse"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_fixedheight_stl(self):
        task = "fixed_height"
        reward = "stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_chain(self):
        task = "fixed_height"
        reward = "gb_chain"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_weighted_reward(self):
        task = "fixed_height"
        reward = "weighted"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_eval_reward(self):
        task = "fixed_height"
        reward = "eval"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_dejan_reward(self):
        task = "fixed_height"
        reward = "gb_bpdr_ci"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)


class TestTrainingLoop(TestCase):

    def test_train_sparse(self):
        generic_training(env_name, 'fixed_height', 'sparse')

    def test_train_stl(self):
        generic_training(env_name, 'fixed_height', 'stl')

    def test_train_chain(self):
        generic_training(env_name, 'fixed_height', 'chain')

    def test_train_weighted(self):
        generic_training(env_name, 'fixed_height', 'weighted')

    def test_train_hierarchical(self):
        generic_training(env_name, 'fixed_height', 'gb_bpdr_ci')