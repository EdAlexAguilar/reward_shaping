from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training, generic_env_test_wt_agent

env_name = "racecar"
task = "drive"


class TestEnv(TestCase):

    def test_default_reward(self):
        result = generic_env_test(env_name, task, reward_name="default")
        self.assertTrue(result)

    def test_eval_reward(self):
        result = generic_env_test(env_name, task, reward_name="eval")
        self.assertTrue(result)

    def test_tltl_reward(self):
        result = generic_env_test(env_name, task, reward_name="tltl")
        self.assertTrue(result)

    def test_hprs_reward(self):
        result = generic_env_test(env_name, task, reward_name="hprs")
        self.assertTrue(result)

    def test_morl_uni_reward(self):
        result = generic_env_test(env_name, task, reward_name="morl_uni")
        self.assertTrue(result)

    def test_morl_dec_reward(self):
        result = generic_env_test(env_name, task, reward_name="morl_dec")
        self.assertTrue(result)

    def test_bhnr_reward(self):
        result = generic_env_test(env_name, task, reward_name="bhnr")
        self.assertTrue(result)


class TestTrainingLoop(TestCase):

    def test_train_default(self):
        generic_training(env_name, task, 'default')

    def test_train_tltl(self):
        generic_training(env_name, task, 'tltl')

    def test_train_hrs_pot(self):
        generic_training(env_name, task, 'hrs_pot')

    def test_train_morl_uni(self):
        generic_training(env_name, task, 'morl_uni')

    def test_train_morl_dec(self):
        generic_training(env_name, task, 'morl_dec')


"""
class TestWithAgent(TestCase):
    def test_progress(self):
        from stable_baselines3 import PPO
        agent = PPO.load("/home/luigi/Development/reward_shaping/logs/racecar/drive_hrs_pot_ppo_Seed474197_1644498938/checkpoint/model_50000_steps.zip")
        generic_env_test_wt_agent(env_name, agent, task, 'default')
"""
