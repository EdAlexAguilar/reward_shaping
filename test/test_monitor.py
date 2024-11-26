from unittest import TestCase

from reward_shaping.monitor.formula import Operator
from reward_shaping.monitor.monitor import EnsureMonitor, AchieveMonitor, ConquerMonitor, EncourageMonitor, Monitor


class TestMonitor(TestCase):

    def _generic_monitor_test(self, monitor, predicate, trace):
        monitor.reset()
        for s in trace:
            state, output = monitor.step(s)
            print(f"input: {s} -> state: {state}, output: {output}")
        sat, rob = monitor.is_sat(), monitor.get_counter()
        print(f"is sat: {sat}, rob: {rob}\n")
        return sat, rob

    def test_safety_1(self):
        predicate = lambda x, y: x
        trace = [1, 2, 3, 3, 2, 1, 0, -1, -2, 1, 0, 2]
        monitor = Monitor.from_spec(Operator.ENSURE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertFalse(sat)
        self.assertTrue(k == 7)

    def test_safety_2(self):
        predicate = lambda x, y: x
        trace = [1, 2, 3, 3, 2, 1]
        monitor = Monitor.from_spec(Operator.ENSURE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 6)

    def test_safety_3(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2, 1, 0, 0, 0, 0]
        monitor = Monitor.from_spec(Operator.ENSURE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertFalse(sat)
        self.assertTrue(k == 0)

    def test_safety_4(self):
        predicate = lambda x, y: x
        trace = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        monitor = Monitor.from_spec(Operator.ENSURE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 9)

    def test_liveness_1(self):
        predicate = lambda x, y: x
        trace = [1, 2, 3, 3, 2, 1, 0, -1, -2, 1, 0, 2]
        monitor = Monitor.from_spec(Operator.ACHIEVE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 10)

    def test_liveness_2(self):
        predicate = lambda x, y: x
        trace = [1, 2, 3, 3, 2, 1]
        monitor = Monitor.from_spec(Operator.ACHIEVE, predicate)
        sat, rob = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(rob == 6)

    def test_liveness_3(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2, 0, 0, 0, 0, 0]
        monitor = Monitor.from_spec(Operator.ACHIEVE, predicate)
        sat, rob = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(rob == 5)

    def test_liveness_4(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2]
        monitor = Monitor.from_spec(Operator.ACHIEVE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertFalse(sat)
        self.assertTrue(k == 0)

    def test_persistence_1(self):
        predicate = lambda x, y: x
        trace = [1, 2, 3, 3, 2, 1, 0, -1, -2, 1, 0, 2]
        monitor = Monitor.from_spec(Operator.CONQUER, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 3)

    def test_persistence_2(self):
        predicate = lambda x, y: x
        trace = [-1, 2, 3, 3, -2, -1]
        monitor = Monitor.from_spec(Operator.CONQUER, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertFalse(sat)
        self.assertTrue(k == 0)

    def test_comfort_1(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
        monitor = Monitor.from_spec(Operator.ENCOURAGE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 10)

    def test_comfort_2(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2, 0, 0, 0, 0, 0, -2]
        monitor = Monitor.from_spec(Operator.ENCOURAGE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 5)

    def test_comfort_3(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2, 0, 0, 0, 0, 0, 1]
        monitor = Monitor.from_spec(Operator.ENCOURAGE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 6)

    def test_comfort_4(self):
        predicate = lambda x, y: x
        trace = [-1, -2, -3, -3, -2, -1]
        monitor = Monitor.from_spec(Operator.ENCOURAGE, predicate)
        sat, k = self._generic_monitor_test(monitor, predicate, trace)
        self.assertTrue(sat)
        self.assertTrue(k == 0)
