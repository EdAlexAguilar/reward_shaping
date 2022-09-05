import time
from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training, generic_env_test_wt_agent

env_name = "cart_pole_obst"
task = "fixed_height"


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

    def test_train_sparse(self):
        generic_training(env_name, task, 'default')

    def test_train_tltl(self):
        generic_training(env_name, task, 'tltl')

    def test_train_morl_uni(self):
        generic_training(env_name, task, 'morl_uni')

    def test_train_morl_dec(self):
        generic_training(env_name, task, 'morl_dec')

    def test_train_hrs_pot(self):
        generic_training(env_name, task, 'hrs_pot')


class TestWithAgent(TestCase):
    def test_eval(self):
        from stable_baselines3 import SAC
        task = "fixed_height"
        checkpoint_paths = {
            # "default": "/home/luigi/Desktop/logs_iros22/cart_pole_obst/fixed_height_default_sac_Seed285740_1644341620/checkpoint/model_2000000_steps.zip",
            # "tltl": "/home/luigi/Desktop/logs_iros22/cart_pole_obst/fixed_height_tltl_sac_Seed303587_1644350363/checkpoint/model_2000000_steps.zip",
            # "bhnr": "/home/luigi/Desktop/logs_iros22/cart_pole_obst/fixed_height_bhnr_sac_Seed127338_1644350363/checkpoint/model_2000000_steps.zip",
            # "morl_uni": "/home/luigi/Desktop/logs_iros22/cart_pole_obst/fixed_height_morl_uni_sac_Seed497156_1644341630/checkpoint/model_2000000_steps.zip",
            # "morl_dec": "/home/luigi/Desktop/logs_iros22/cart_pole_obst/fixed_height_morl_dec_sac_Seed351891_1644350363/checkpoint/model_2000000_steps.zip",
            "hrs_pos": "/home/luigi/Desktop/logs_iros22/cart_pole_obst/fixed_height_hrs_pot_sac_Seed235224_1644341621/checkpoint/model_2000000_steps.zip"
        }
        for reward, checkpoint_path in checkpoint_paths.items():
            print(f"training reward: {reward}")
            agent = SAC.load(checkpoint_path)
            generic_env_test_wt_agent(env_name, agent, task, 'eval')
            print()
