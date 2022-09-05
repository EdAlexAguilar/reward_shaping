from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training, generic_env_test_wt_agent

env_name = "lunar_lander"
task = "land"


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
        generic_training(env_name, task, 'hprs')


class TestWithAgent(TestCase):
    def test_forward_eval(self):
        from stable_baselines3 import SAC
        task = "land"
        checkpoint_paths = {
            "default": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_default_sac_Seed320812_1644341625/checkpoint/model_1500000_steps.zip",
            "tltl": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_tltl_sac_Seed143933_1644390812/checkpoint/model_3000000_steps.zip",
            "morl_uni": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_morl_uni_sac_Seed27201_1644350362/checkpoint/model_3000000_steps.zip",
            "morl_dec": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_morl_dec_sac_Seed496777_1644341634/checkpoint/model_3000000_steps.zip",
            "hrs_pos": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_hrs_pot_sac_Seed271147_1644341625/checkpoint/model_1500000_steps.zip"
        }
        for reward, checkpoint_path in checkpoint_paths.items():
            print(f"training reward: {reward}")
            agent = SAC.load(checkpoint_path)
            generic_env_test_wt_agent(env_name, agent, task, 'eval')
            print()
