from unittest import TestCase

from reward_shaping.core.configs import BuildGraphReward
from reward_shaping.core.wrappers import RewardWrapper
from reward_shaping.core.graph_based import GraphBasedReward
from reward_shaping.training.utils import make_base_env, load_env_params


class TestGraphBasedRewards(TestCase):

    def _rollout(self, env, const_rew=None, render=False):
        env.reset()
        done = False
        ret = 0.0
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            ret += reward
            if render:
                env.render()
            #print(reward)
            self.assertTrue(reward == const_rew or const_rew is None)
        print(ret)

    def test_always_ones(self):
        """ each node returns reward=1.0 and sat=1.0
        then we expect a point-wise reward of N, where N is the number of nodes in the dag"""
        one = lambda state, action, next_state, info: 1.0
        reward_fns = {
            'S_coll': (one, one),
            'S_fall': (one, one),
            'H_nfeas': (one, one),
            'H_feas': (one, one),
            'T_orig': (one, one),
            'T_bal': (one, one)
        }
        topology = {
            'S_coll': ['H_feas', 'H_nfeas'],
            'S_fall': ['H_feas', 'H_nfeas'],
            'H_feas': ['T_orig'],
            'H_nfeas': ['T_bal'],
            'T_orig': ['T_bal'],
        }
        graph_reward = GraphBasedReward.from_collections(nodes=reward_fns, topology=topology)
        env_params = load_env_params('cart_pole_obst', 'fixed_height')
        env = make_base_env('cart_pole_obst', env_params)
        env = RewardWrapper(env, reward_fn=graph_reward)
        for _ in range(5):
            self._rollout(env, const_rew=1.0 * len(reward_fns), render=True)
        env.close()

    def test_enable_feas_branch(self):
        zero = lambda state, action, next_state, info: 0.0
        one = lambda state, action, next_state, info: 1.0
        reward_fns = {
            'S_coll': (one, one),
            'S_fall': (one, one),
            'H_nfeas': (zero, zero),
            'H_feas': (zero, one),
            'T_orig': (one, one),
            'T_bal': (one, one),
            'C_bal': (one, one)
        }
        topology = {
            'S_coll': ['H_feas', 'H_nfeas'],
            'S_fall': ['H_feas', 'H_nfeas'],
            'H_feas': ['T_orig'],
            'H_nfeas': ['T_bal'],
            'T_orig': ['C_bal'],
        }
        graph_reward = GraphBasedReward.from_collections(nodes=reward_fns, topology=topology)
        env_params = load_env_params('cart_pole_obst', 'fixed_height')
        env = make_base_env('cart_pole_obst', env_params)
        env = RewardWrapper(env, reward_fn=graph_reward)
        for _ in range(5):
            self._rollout(env, const_rew=4.0, render=True)
        env.close()

    def test_only_safety(self):
        zero = lambda state, action, next_state, info: 0.0
        one = lambda state, action, next_state, info: 1.0
        reward_fns = {
            'S_coll': (one, zero),
            'S_fall': (one, zero),
            'H_nfeas': (one, one),
            'H_feas': (one, one),
            'T_orig': (one, one),
            'T_bal': (one, one)
        }
        topology = {
            'S_coll': ['H_feas', 'H_nfeas'],
            'S_fall': ['H_feas', 'H_nfeas'],
            'H_feas': ['T_orig'],
            'H_nfeas': ['T_bal'],
            'T_orig': ['T_bal'],
        }
        graph_reward = GraphBasedReward.from_collections(nodes=reward_fns, topology=topology)
        env_params = load_env_params('cart_pole_obst', 'fixed_height')
        env = make_base_env('cart_pole_obst', env_params)
        env = RewardWrapper(env, reward_fn=graph_reward)
        for _ in range(5):
            self._rollout(env, const_rew=2.0, render=True)
        env.close()

    def test_render(self):
        zero = lambda state, action, next_state, info: 0.0
        one = lambda state, action, next_state, info: 1.0
        reward_fns = {
            'S_coll': (one, zero),
            'S_fall': (one, zero),
            'H_nfeas': (one, one),
            'H_feas': (one, one),
            'T_orig': (one, one),
            'T_bal': (one, one),
            'C_bal': (one, one)
        }
        topology = {
            'S_coll': ['H_feas', 'H_nfeas'],
            'S_fall': ['H_feas', 'H_nfeas'],
            'H_feas': ['T_orig'],
            'H_nfeas': ['T_bal'],
            'T_orig': ['C_bal'],
        }
        graph_reward = GraphBasedReward.from_collections(nodes=reward_fns, topology=topology)
        graph_reward.render()

    def test_gb_continuous_score_binary_indicator(self):
        from reward_shaping.envs.cart_pole_obst.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator
        env_params = load_env_params('cart_pole_obst', 'fixed_height')
        env = make_base_env('cart_pole_obst', env_params)
        graph_conf = GraphWithContinuousScoreBinaryIndicator(env_params)
        reward_fn = BuildGraphReward.from_conf(graph_config=graph_conf)
        env = RewardWrapper(env, reward_fn=reward_fn)
        for _ in range(5):
            self._rollout(env, render=True)

    def test_gb_binary_safeties(self):
        from reward_shaping.envs.cart_pole_obst.rewards.graph_based import GraphWithBinarySafetyScoreBinaryIndicator
        env_params = load_env_params('cart_pole_obst', 'fixed_height')
        env = make_base_env('cart_pole_obst', env_params)
        graph_conf = GraphWithBinarySafetyScoreBinaryIndicator(env_params)
        reward_fn = BuildGraphReward.from_conf(graph_config=graph_conf)
        env = RewardWrapper(env, reward_fn=reward_fn)
        for _ in range(5):
            self._rollout(env, render=True)

    def test_gb_single_safety_node(self):
        from reward_shaping.envs.cart_pole_obst.rewards.graph_based import GraphWithSingleConjunctiveSafetyNode
        env_params = load_env_params('cart_pole_obst', 'fixed_height')
        env = make_base_env('cart_pole_obst', env_params)
        graph_conf = GraphWithSingleConjunctiveSafetyNode(env_params)
        reward_fn = BuildGraphReward.from_conf(graph_config=graph_conf)
        env = RewardWrapper(env, reward_fn=reward_fn)
        for _ in range(5):
            self._rollout(env, render=True)