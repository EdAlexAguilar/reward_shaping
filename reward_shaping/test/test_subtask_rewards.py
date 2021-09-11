from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from reward_shaping.core.utils import get_normalized_reward
from reward_shaping.envs.cart_pole_obst.cp_continuousobstacle_env import Obstacle


class TestSubtaskRewards(TestCase):
    @staticmethod
    def _get_default_info(env):
        if env == "cart_pole_obst":
            info = {'x_limit': 2.5,
                    'x_target': 0.0,
                    'x_target_tol': 0.25,
                    'theta_limit': np.deg2rad(90),
                    'theta_target': np.deg2rad(0),
                    'theta_target_tol': np.deg2rad(24),
                    'axle_y': None,
                    'pole_length': 1.0,
                    'tau': 0.02}
        else:
            raise NotImplementedError(env)
        return info

    @staticmethod
    def _evaluate_on_range(reward_fn, states, info, next_states=None):
        if next_states is None:
            next_states = states
        rewards = [reward_fn(state, None, next_state, info) for state, next_state in zip(states, next_states)]
        return rewards

    @staticmethod
    def _evaluate_on_grid(reward_fn, states_grid, info):
        rewards = [[reward_fn(state, None, state, info) for state in row] for row in states_grid]
        return rewards

    @staticmethod
    def _plot(xx, yy, xlabel="", ylabel="", show=True):
        import matplotlib.pyplot as plt
        plt.plot(xx, yy)
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        if show:
            plt.show()

    @staticmethod
    def _plot_grid(values, xlabel="", ylabel=""):
        import matplotlib.pyplot as plt
        plt.imshow(values)
        plt.colorbar()
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.show()

    def test_outside_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        binary_exit_fn = fns.OutsideReward(exit_penalty=-1.0, no_exit_bonus=0.0)
        cont_exit_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                min_r_state={'x': 2.5},
                                                max_r_state={'x': 0.0},
                                                info=info)
        xx = np.linspace(-3.0, 3.0, 100)
        states = [{'x': x} for x in xx]
        rewards = self._evaluate_on_range(binary_exit_fn, states, info)
        sats = self._evaluate_on_range(cont_exit_fn, states, info)
        self._plot(xx, rewards, xlabel="x", ylabel="binary reward")
        self._plot(xx, sats, xlabel="x", ylabel="continuous sat")

    def test_reachtarget_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        target_fun, target_sat = get_normalized_reward(fns.ReachTargetReward(),
                                                       min_r_state={'x': info['x_limit']},
                                                       max_r_state={'x': info['x_target']},
                                                       info=info)
        xx = np.linspace(-3.0, 3.0, 100)
        states = [{'x': x} for x in xx]
        rewards = self._evaluate_on_range(target_fun, states, info)
        sats = self._evaluate_on_range(target_sat, states, info)
        self._plot(xx, rewards)
        self._plot(xx, sats)

    def test_collision_reward(self):
        # this test is approximated, by assuming the obstacle in a fixed position and varying x, theta
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        info['axle_y'] = 1.0 + 0.25 / 4
        info['pole_length'] = 1.0
        info['dist_to_ground'] = 0.95
        obstacle = Obstacle(info['axle_y'], info['pole_length'],
                            left_x=0.5, left_y=info['axle_y'] + info['dist_to_ground'],
                            width=0.2, height=0.1)
        collision_fn = fns.ContinuousCollisionReward()
        xx = np.linspace(-2.5, 2.5, 100)
        thetas = np.linspace(-np.pi / 2, np.pi / 2, 100)
        states = [[{'x': x, 'theta': theta,
                    'obstacle_left': obstacle.left_x, 'obstacle_right': obstacle.right_x,
                    'obstacle_bottom': obstacle.bottom_y, 'obstacle_top': obstacle.top_y} for theta in thetas] for x in
                  xx]
        rewards = self._evaluate_on_grid(collision_fn, states, info)
        print(f"Min Reward: {np.min(rewards)}")
        print(f"Max Reward: {np.max(rewards)}")
        self._plot_grid(rewards, xlabel="x", ylabel="theta")

    def test_falldown_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        falldown_fn = fns.ContinuousFalldownReward()
        norm_falldown_fn, _ = get_normalized_reward(fns.ContinuousFalldownReward(),
                                                    min_r_state={'theta': info['theta_limit']},
                                                    max_r_state={'theta': 0.0}, info=info)
        thetas = np.linspace(-np.pi / 2 - 0.1, np.pi / 2 + 0.1, 100)
        states = [{'theta': theta} for theta in thetas]
        rewards = self._evaluate_on_range(falldown_fn, states, info)
        norm_rewards = self._evaluate_on_range(norm_falldown_fn, states, info)
        print(f"Min Reward: {np.min(rewards)}")
        print(f"Max Reward: {np.max(rewards)}")
        self._plot(thetas, rewards, xlabel="theta", ylabel="reward")
        print(f"Min Norm Reward: {np.min(norm_rewards)}")
        print(f"Max Norm Reward: {np.max(norm_rewards)}")
        self._plot(thetas, norm_rewards, xlabel="theta", ylabel="normalized reward")

    def test_continuous_outside_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        cont_exit_fn = fns.ContinuousOutsideReward()
        norm_cont_exit_fn, _ = get_normalized_reward(fns.ContinuousOutsideReward(),
                                                     min_r_state={'x': info['x_limit']},
                                                     max_r_state={'x': 0.0},
                                                     info=info)
        xx = np.linspace(-2.5, 2.5, 100)
        states = [{'x': x} for x in xx]
        rewards = self._evaluate_on_range(cont_exit_fn, states, info)
        norm_rewards = self._evaluate_on_range(norm_cont_exit_fn, states, info)
        self._plot(xx, rewards, xlabel="x", ylabel="continuous outside")
        print(f"Min Reward: {np.min(rewards)}")
        print(f"Max Reward: {np.max(rewards)}")
        self._plot(xx, norm_rewards, xlabel="x", ylabel="norm continuous outside")
        print(f"Min Norm Reward: {np.min(norm_rewards)}")
        print(f"Max Norm Reward: {np.max(norm_rewards)}")

    def test_continuous_reachtarget_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        target_fun = fns.ReachTargetReward()
        norm_target_fun, _ = get_normalized_reward(fns.ReachTargetReward(),
                                                   min_r_state={'x': info['x_limit']},
                                                   max_r_state={'x': info['x_target'] - info['x_target_tol']},
                                                   info=info)
        xx = np.linspace(-3.0, 3.0, 100)
        states = [{'x': x} for x in xx]
        rewards = self._evaluate_on_range(target_fun, states, info)
        norm_rewards = self._evaluate_on_range(norm_target_fun, states, info)
        plt.subplot(1, 2, 1)
        self._plot(xx, rewards, xlabel="x", ylabel="reward", show=False)
        plt.subplot(1, 2, 2)
        self._plot(xx, norm_rewards, xlabel="x", ylabel="norm reward", show=False)
        plt.show()

    def test_progresstarget_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        progress_fun = fns.ProgressToTargetReward(progress_coeff=1.0)
        x0 = 1.0
        delta_xx = np.linspace(-.5, .5, 100)
        states = [{'x': x0} for _ in delta_xx]
        next_states = [{'x': x0 + dx} for dx in delta_xx]
        rewards = self._evaluate_on_range(progress_fun, states, info, next_states=next_states)
        self._plot(delta_xx, rewards, xlabel="delta x", ylabel="reward", show=False)
        plt.show()

    def test_balance_reward(self):
        import reward_shaping.envs.cart_pole_obst.rewards.subtask_rewards as fns
        info = self._get_default_info("cart_pole_obst")
        balance_fun = fns.BalanceReward()
        norm_balance_fun, _ = get_normalized_reward(fns.BalanceReward(),
                                                    min_r_state={
                                                        'theta': info['theta_target'] - info['theta_target_tol']},
                                                    max_r_state={'theta': info['theta_target']},
                                                    info=info)
        thetas = np.linspace(-np.pi / 2 - 0.1, np.pi / 2 + 0.1, 100)
        states = [{'theta': theta} for theta in thetas]
        rewards = self._evaluate_on_range(balance_fun, states, info)
        norm_rewards = self._evaluate_on_range(norm_balance_fun, states, info)
        plt.subplot(1, 2, 1)
        self._plot(thetas, rewards, xlabel="theta", ylabel="reward", show=False)
        print(f"Min Reward: {np.min(rewards)}")
        print(f"Max Reward: {np.max(rewards)}")
        plt.subplot(1, 2, 2)
        self._plot(thetas, norm_rewards, xlabel="theta", ylabel="norm reward", show=False)
        print(f"Min Norm Reward: {np.min(norm_rewards)}")
        print(f"Max Norm Reward: {np.max(norm_rewards)}")
        plt.show()
