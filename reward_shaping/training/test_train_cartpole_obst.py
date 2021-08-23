from argparse import Namespace
from unittest import TestCase

from reward_shaping.training.train_cartpole import train


class TestTrainingLoop(TestCase):
    def _test_fast_train(self, task, reward):
        args = {'task': task, 'reward': reward,
                'algo': 'sac', 'steps': 500, 'seed': 0}
        train(Namespace(**args))

    def test_train_sparse(self):
        self._test_fast_train('fixed_height', 'sparse')

    def test_train_continuous(self):
        self._test_fast_train('fixed_height', 'continuous')

    def test_train_stl(self):
        self._test_fast_train('fixed_height', 'stl')

    def test_train_bool_stl(self):
        self._test_fast_train('fixed_height', 'bool_stl')
