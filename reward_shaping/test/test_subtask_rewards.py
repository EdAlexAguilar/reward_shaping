from unittest import TestCase

from reward_shaping.core.utils import get_normalized_reward
import numpy as np


class TestSubtaskRewards(TestCase):
    @staticmethod
    def _get_default_info(env):
        if env == "cart_pole_obst":
            info = {'x_limit': 2.5,
                    'x_target': 0.0,
                    'x_target_tol': 0.25,
                    'theta_limit': np.deg2rad(90),
                    'theta_target': np.deg2rad(0),
                    'theta_target_tol': np.deg2rad(24)}
        else:
            raise NotImplementedError(env)
        return info

    @staticmethod
    def _evaluate_on_range(reward_fn, states, info):
        rewards = [reward_fn(state, None, state, info) for state in states]
        return rewards

    @staticmethod
    def _plot(xx, yy):
        import matplotlib.pyplot as plt
        plt.plot(xx, yy)
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
        self._plot(xx, rewards)
        self._plot(xx, sats)

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
