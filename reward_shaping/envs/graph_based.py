import pathlib
from typing import Dict, List

import numpy as np
import yaml

from reward_shaping.envs.cart_pole_obst.cp_continuousobstacle_env import CartPoleContObsEnv
from reward_shaping.envs.cart_pole_obst import reward_fns

import networkx as nx

from reward_shaping.envs.core import RewardFunction, RewardWrapper


class GraphBasedReward(RewardFunction):

    def __init__(self, nodes: Dict[str, RewardFunction] = None, topology: Dict[str, List[str]] = None):
        self._graph = nx.DiGraph()
        if topology and not nodes:
            raise ValueError('If a topology is provided, nodes must also be provided.')
        if nodes:
            for node, reward_fn in nodes.items():
                self.add_reward(node, reward_fn)
        if topology:
            self._graph.add_edges_from(topology)
            if not nx.is_directed_acyclic_graph(self._graph):
                raise ValueError('Graph is not a DAG.')

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
        top_level_nodes = [v for v in self._graph if self._graph.out_degree(v) == 0]
        for v in top_level_nodes:
            self._evaluate_node(node=v, state=state, action=action, next_state=next_state)

    def _compute_rewards(self) -> float:
        enabled_nodes = filter(lambda v: v['enabled'], self._graph.nodes.values())
        rewards = sum(v['score'] for v in enabled_nodes)
        return rewards

    def _reset_graph(self):
        for node in self._graph.nodes:
            attributes = self._graph.nodes[node]
            if 'score' in attributes:
                attributes.pop('score')
            if 'enabled' in attributes:
                attributes.pop('enabled')

    def __call__(self, state, action=None, next_state=None) -> float:
        self._evaluate_graph(state=state, action=action, next_state=next_state)
        reward = self._compute_rewards()
        return reward


def make_env():
    global task, env
    task = "random_height"
    env_config = pathlib.Path(f"cart_pole_obst/tasks/{task}.yml")
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    env = CartPoleContObsEnv(**env_params, eval=True, seed=0)
    return env

if __name__ == '__main__':
    rewards = {
        'S_coll': reward_fns.CollisionReward(no_collision_bonus=10),
        'S_fall': reward_fns.ContinuousFalldownReward(theta_limit=5),
        'H_nfeas': reward_fns.Indicator(reward_fns.CheckOvercomingFeasibility(obstacle_y=1, axle_y=1, feasible_height=5), negate=True),
        'H_feas': reward_fns.Indicator(reward_fns.CheckOvercomingFeasibility(obstacle_y=1, axle_y=1, feasible_height=5)),
        'T_orig': reward_fns.ReachTargetReward(x_target=0, x_target_tol=2),
        'T_bal': reward_fns.BalanceReward(theta_target=0, theta_target_tol=0.1)
    }
    topology = {
        'S_coll': ['H_feas', 'H_nfeas'],
        'S_fall': ['H_feas', 'H_nfeas'],
        'H_feas': ['T_orig'],
        'H_nfeas': ['T_bal'],
        'T_orig': ['T_bal'],
    }

    graph_reward = GraphBasedReward(nodes=rewards, topology=topology)
    env = make_env()
    env = RewardWrapper(env, reward_fn=graph_reward)
    env.reset()
    for x in range(1000000):
        obs, reward, done, info = env.step(np.array([0]))
        env.render()
        print(reward)
    x = None