from unittest import TestCase

from reward_shaping.core.helper_fns import monitor_mtl_filtering_episode

env_name = "cart_pole_obst"
task = "fixed_height"


class TestFilteringSemanticsMonitoring(TestCase):

    def _generic_example_1(self, episode, exp_rob_trace):
        # def spec
        spec = "(always(dist_obstacle>0.1)) and (eventually(always(dist_origin)<0.1))"
        vars = ["time", "dist_obstacle", "dist_origin"]
        types = ["int", "float", "float"]
        # compute result
        robustness = monitor_mtl_filtering_episode(spec, vars, types, episode)
        self.assertTrue(all([exp_rob_trace[i] == robustness[i][1] for i in range(len(robustness))]))

    def test_episode_1(self):
        episode = {"time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   "dist_origin": [2.0, 1.5, 1.0, 0.75, 0.5, 0.5, 0.5, 0.75, 0.8, 0.75],
                   "dist_obstacle": [0.3, 0.2, 0.3, 0.2, 0.2, 0.15, 0.2, 0.25, 0.3, 0.2]
                   }
        expected_robustness = [0.0] * 10
        self._generic_example_1(episode, exp_rob_trace=expected_robustness)

    def test_episode_2(self):
        episode = {"time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   "dist_origin": [2.0, 1.5, 1.0, 0.75, 0.5, 0.0, 0.2, 0.1, 0.0, 0.0],
                   "dist_obstacle": [0.15, 0.15, 0.2, 0.25, 0.2, 0.15, 0.2, 0.15, 0.2, 0.25]
                   }
        expected_robustness = [0.2] * 9 + [0.1]
        self._generic_example_1(episode, exp_rob_trace=expected_robustness)

    def test_episode_3(self):
        episode = {"time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   "dist_origin": [2.0, 1.5, 1.0, 0.75, 0.5, 0.0, 0.2, 0.1, 0.0, 0.0],
                   "dist_obstacle": [0.15, 0.2, 0.2, 0.1, 0.05, 0.0, 0.1, 0.2, 0.25, 0.2]
                   }
        expected_robustness = [0.0] * 7 + [0.2] * 2 + [0.1]
        self._generic_example_1(episode, exp_rob_trace=expected_robustness)