from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training

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

    def test_weighted(self):
        task = "forward"
        reward = "weighted"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_chain(self):
        task = "forward"
        reward = "gb_chain"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_binary_progress(self):
        task = "forward"
        reward = "gb_bpr_ci"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_binary_progress_and_indicator(self):
        task = "forward"
        reward = "gb_bpr_bi"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_eval_reward(self):
        task = "forward"
        reward = "eval"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)


class TestTrainingLoop(TestCase):

    def test_train_default(self):
        generic_training(env_name, 'forward', 'default')

    def test_train_stl(self):
        generic_training(env_name, 'forward', 'stl')

    def test_train_gbased_binary_ind(self):
        generic_training(env_name, 'forward', 'gb_cr_bi')