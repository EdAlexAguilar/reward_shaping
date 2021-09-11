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

    def test_gb_boolsafety(self):
        task = "fixed_height"
        reward = "gb_bpr_bi"
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

    def test_binaryprogress_reward(self):
        task = "fixed_height"
        reward = "gb_bpr_ci"
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

    def test_train_continuous(self):
        generic_training(env_name, 'fixed_height', 'continuous')

    def test_train_stl(self):
        generic_training(env_name, 'fixed_height', 'stl')

    def test_train_bool_stl(self):
        generic_training(env_name, 'fixed_height', 'bool_stl')

    def test_train_gbased_binary_ind(self):
        generic_training(env_name, 'fixed_height', 'gb_cr_bi')

    def test_train_gbased_continuous_ind(self):
        generic_training(env_name, 'fixed_height', 'gb_cr_ci')

    def test_train_gbased_progress(self):
        generic_training(env_name, 'fixed_height', 'gb_pcr_bi')


class PlotRewardLandscape(TestCase):

    def test_plot_all(self):
        import matplotlib.pyplot as plt
        rewards = ['sparse', 'weighted', 'gb_chain',
                   'gb_cr_ci', 'gb_cr_bi', 'gb_bcr_bi']
        rewards = ['gb_cpdr_ci', 'gb_cpr_ci']
        rows, cols, i = 1, 2, 1
        plt.figure(figsize=(10, 4))
        for reward in rewards:
            plt.subplot(rows, cols, i)
            plot_cpole_reward(reward)
            i += 1
        plt.savefig(f"cp_rewards_{time.time()}.pdf")

    def test_plot_prog(self):
        import matplotlib.pyplot as plt
        rewards = ['gb_cpr_ci', 'gb_bpdr_ci']
        rows, cols, i = 1, 2, 1
        for reward in rewards:
            plt.subplot(rows, cols, i)
            plot_cpole_progreward(reward)
            i += 1
        plt.savefig(f"cp_prog_rewards_{time.time()}.pdf")
