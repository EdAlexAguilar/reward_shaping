from unittest import TestCase

from reward_shaping.test.test import generic_env_test, generic_training, generic_env_test_wt_agent

env_name = "bipedal_walker"
task = "forward"


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

    def test_train_hrs_pot(self):
        generic_training(env_name, task, 'hrs_pot')

    def test_train_morl_uni(self):
        generic_training(env_name, task, 'morl_uni')

    def test_train_morl_dec(self):
        generic_training(env_name, task, 'morl_dec')


class TestWithAgent(TestCase):
    def test_forward_eval(self):
        from stable_baselines3 import SAC
        task = "forward"
        checkpoint_paths = {
            "default": "/home/luigi/Desktop/logs_iros22/bipedal_walker/forward/forward_default_sac_Seed860530_1644341630/checkpoint/model_2000000_steps.zip",
            "tltl": "/home/luigi/Desktop/logs_iros22/bipedal_walker/forward/forward_tltl_sac_Seed430414_1644341622/checkpoint/model_2000000_steps.zip",
            "morl_uni": "/home/luigi/Desktop/logs_iros22/bipedal_walker/forward/forward_morl_uni_sac_Seed294416_1644383514/checkpoint/model_2000000_steps.zip",
            "morl_dec": "/home/luigi/Desktop/logs_iros22/bipedal_walker/forward/forward_morl_dec_sac_Seed648302_1644341623/checkpoint/model_2000000_steps.zip",
            "hrs_pos": "/home/luigi/Desktop/logs_iros22/bipedal_walker/forward/forward_hrs_pot_sac_Seed840551_1644350359/checkpoint/model_2000000_steps.zip"
        }
        for reward, checkpoint_path in checkpoint_paths.items():
            print(f"training reward: {reward}")
            agent = SAC.load(checkpoint_path)
            generic_env_test_wt_agent(env_name, agent, task, 'eval')
            print()

    def test_hardcore_eval(self):
        from stable_baselines3 import SAC
        task = "hardcore"
        checkpoint_paths = {
            # "default": "/d_drive/reward_shaping_logs/iros22/bipedal_walker_15022022/hardcore/hardcore_default_sac_Seed750538_1644956389/checkpoint/model_1500000_steps.zip",
            # "tltl": "/home/luigi/Desktop/logs_iros22/bipedal_walker/hardcore/hardcore_tltl_sac_Seed204494_1644350360/checkpoint/model_600000_steps.zip",
            # "morl_uni": "/home/luigi/Desktop/logs_iros22/bipedal_walker/hardcore/hardcore_morl_uni_sac_Seed133842_1644350360/checkpoint/model_600000_steps.zip",
            # "morl_dec": "/home/luigi/Desktop/logs_iros22/bipedal_walker/hardcore/hardcore_morl_dec_sac_Seed100327_1644341625/checkpoint/model_600000_steps.zip",
            "hrs_pos": "/d_drive/reward_shaping_logs/iros22/bipedal_walker_15022022/hardcore/hardcore_hrs_pot_sac_Seed566507_1644956388/checkpoint/model_1500000_steps.zip"
        }
        for reward, checkpoint_path in checkpoint_paths.items():
            print(f"training reward: {reward}")
            agent = SAC.load(checkpoint_path)
            generic_env_test_wt_agent(env_name, agent, task, 'eval')
            del agent
            print()
