from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training

env_name = "bipedal_walker"
task = "forward"


class TestBipedalWalker(TestCase):

    def test_default_reward(self):
        reward = "default"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_tltl(self):
        reward = "tltl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_morl_uni(self):
        reward = "morl_uni"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_morl_dec(self):
        reward = "morl_dec"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_eval_reward(self):
        reward = "eval"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_hrs_reward(self):
        reward = "hrs_pot"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)


class TestTrainingLoop(TestCase):

    def test_train_sparse(self):
        generic_training(env_name, task, 'default')

    def test_train_tltl(self):
        generic_training(env_name, task, 'tltl')

    def test_train_hrs_pot(self):
        generic_training(env_name, task, 'hrs_pot')

    def test_train_morl_uni(self):
        generic_training(env_name, task, 'morl_uni')

    def test_train_morl_dec(self):
        generic_training(env_name, task, 'morl_dec')
