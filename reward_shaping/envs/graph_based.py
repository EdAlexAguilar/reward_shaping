import pathlib

import numpy as np
import yaml

from reward_shaping.envs.cart_pole_obst.cp_continuousobstacle_env import CartPoleContObsEnv
from reward_shaping.envs.cart_pole_obst import reward_fns

import networkx as nx

from reward_shaping.envs.core import RewardFunction, RewardWrapper


class GraphBasedReward(RewardFunction):

    def __init__(self, reward_of_interest: str):
        self._reward_label = reward_of_interest
        self._graph = nx.DiGraph()

    def add_reward(self, label: str, reward_fn: RewardFunction):
        self._graph.add_node(label, reward_fn=reward_fn)

    def add_dependency(self, source: str, target: str):
        self._graph.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError(f'Dependency {source}->{target} introduced cycle.')

    def _evaluate_node(self, node: str, state, action, next_state) -> bool:
        if 'enabled' in self._graph.nodes[node]:
            return self._graph.nodes[node]['enabled']
        else:
            predecessors = self._graph.predecessors(node)
            is_enabled = all([self._evaluate_node(pred, state, action, next_state) for pred in predecessors])
            reward_fn = self._graph.nodes[node]['reward_fn']
            score = reward_fn(state, action, next_state)
            is_satisfied = score > 0
            is_enabled = is_enabled and is_satisfied
            nx.set_node_attributes(self._graph, {node: is_enabled}, 'enabled')
            nx.set_node_attributes(self._graph, {node: score}, 'score')
            return is_enabled

    def _evaluate_graph(self, state, action, next_state):
        self._reset_graph()
        self._evaluate_node(node=self._reward_label, state=state, action=action, next_state=next_state)

    def _reset_graph(self):
        for node in self._graph.nodes:
            attributes = self._graph.nodes[node]
            if 'score' in attributes:
                attributes.pop('score')
            if 'enabled' in attributes:
                attributes.pop('enabled')

    def __call__(self, state, action=None, next_state=None) -> float:
        self._evaluate_graph(state=state, action=action, next_state=next_state)
        nodes_in_hierarchy = nx.descendants(self._graph.reverse(), source=self._reward_label)
        nodes_in_hierarchy.add(self._reward_label)
        reward = 0
        for v in nodes_in_hierarchy:
            if self._graph.nodes[v]['enabled']:
                reward += self._graph.nodes[v]['score']
        return reward


if __name__ == '__main__':
    graph = GraphBasedReward(reward_of_interest='T_bal')
    graph.add_reward('S_coll', reward_fn=reward_fns.CollisionReward(no_collision_bonus=10))
    graph.add_reward('S_fall', reward_fn=reward_fns.ContinuousFalldownReward(theta_limit=5))
    graph.add_reward('H_nfeas', reward_fn=reward_fns.Indicator(
        reward_fns.CheckOvercomingFeasibility(obstacle_y=1, axle_y=1, feasible_height=5), negate=True))
    graph.add_reward('H_feas', reward_fn=reward_fns.Indicator(
        reward_fns.CheckOvercomingFeasibility(obstacle_y=1, axle_y=1, feasible_height=5)))
    graph.add_reward('T_orig', reward_fn=reward_fns.ReachTargetReward(x_target=0, x_target_tol=2))
    graph.add_reward('T_bal', reward_fn=reward_fns.BalanceReward(theta_target=0, theta_target_tol=0.1))

    graph.add_dependency(source='S_coll', target='H_feas')
    graph.add_dependency(source='S_coll', target='H_nfeas')
    graph.add_dependency(source='S_fall', target='H_feas')
    graph.add_dependency(source='S_fall', target='H_nfeas')
    graph.add_dependency(source='H_feas', target='T_orig')
    graph.add_dependency(source='H_nfeas', target='T_bal')
    graph.add_dependency(source='T_orig', target='T_bal')

    task = "random_height"
    env_config = pathlib.Path(f"cart_pole_obst/tasks/{task}.yml")
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    env = CartPoleContObsEnv(**env_params, eval=True, seed=0)
    env = RewardWrapper(env, reward_fn=graph)
    env.reset()
    for x in range(1000000):
        obs, reward, done, info = env.step(np.array([0]))
        env.render()
        print(reward)
    x = None