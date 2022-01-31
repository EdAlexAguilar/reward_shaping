import time
from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training

env_name = "cart_pole_obst"


class TestCartPoleObstacle(TestCase):

    def test_fixedheight_sparse(self):
        task = "fixed_height"
        reward = "default"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_fixedheight_tltl(self):
        task = "fixed_height"
        reward = "tltl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_morl_uni_reward(self):
        task = "fixed_height"
        reward = "morl_uni"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_morl_dec_reward(self):
        task = "fixed_height"
        reward = "morl_dec"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_eval_reward(self):
        task = "fixed_height"
        reward = "eval"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)


class TestTrainingLoop(TestCase):

    def test_train_sparse(self):
        generic_training(env_name, 'fixed_height', 'default')

    def test_train_stl(self):
        generic_training(env_name, 'fixed_height', 'tltl')

    def test_train_morl_uni(self):
        generic_training(env_name, 'fixed_height', 'morl_uni')

    def test_train_morl_dec(self):
        generic_training(env_name, 'fixed_height', 'morl_dec')

    def test_train_hrs_pot(self):
        generic_training(env_name, 'fixed_height', 'hrs_pot')
