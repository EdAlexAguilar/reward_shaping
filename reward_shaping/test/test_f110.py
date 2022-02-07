from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training, generic_env_test_wt_agent

env_name = "f1tenth"
task = "drive"


class TestF1Tenth(TestCase):

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

    def test_bhnr_reward(self):
        reward = "bhnr"
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


class TestAgent(TestCase):

    def test_agent_hrs_pot(self):
        from stable_baselines3 import PPO
        agent = PPO.load("/home/luigi/Development/reward_shaping/logs/f1tenth/try_cont_margin/drive_hrs_pot_ppo_Seed478637_1644231785/checkpoint/model_500000_steps.zip")
        generic_env_test_wt_agent(env_name, agent, task, 'eval')

    def test_agent_min_action(self):
        from stable_baselines3 import SAC
        agent = SAC.load("../../logs/model_minaction_700000_steps")
        generic_env_test_wt_agent(env_name, agent, task, 'hrs_pot')


