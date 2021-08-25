import pathlib
from argparse import Namespace
from unittest import TestCase

from gym.wrappers import FlattenObservation

from reward_shaping.training.utils import make_env, make_reward_wrap, make_agent


class TestTrainingLoop(TestCase):
    def _train(self, args):
        if args.task in ['balance', 'target']:
            args.env = "cart_pole"
        else:
            args.env = "cart_pole_obst"

        # create training environment
        train_env, env_params = make_env(args.env, args.task, args.reward, seed=args.seed)

        # create agent
        model = make_agent(args.env, train_env, args.algo, logdir=None)

        # prepare for training

        # train
        model.learn(total_timesteps=args.steps)
        train_env.close()

    def _test_fast_train(self, task, reward):
        # try to run a simple training loop
        args = {'task': task, 'reward': reward,
                'algo': 'sac', 'steps': 500, 'seed': 0}
        self._train(Namespace(**args))

    def test_train_sparse(self):
        self._test_fast_train('fixed_height', 'sparse')

    def test_train_continuous(self):
        self._test_fast_train('fixed_height', 'continuous')

    def test_train_stl(self):
        self._test_fast_train('fixed_height', 'stl')

    def test_train_bool_stl(self):
        self._test_fast_train('fixed_height', 'bool_stl')

    def test_train_gbased_binary_ind(self):
        self._test_fast_train('fixed_height', 'gb_cr_bi')

    def test_train_gbased_continuous_ind(self):
        self._test_fast_train('fixed_height', 'gb_cr_ci')
