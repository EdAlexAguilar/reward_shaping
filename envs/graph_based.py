import pathlib

import yaml

from envs.cart_pole_obst.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.cart_pole_obst import reward_fns
from envs.core import RewardFunction
import networkx as nx

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

    def _evaluate_graph(self, node: str, state, action, next_state) -> bool:
        predecessors = self._graph.predecessors(node)
        is_enabled = all([self._evaluate_graph(pred, state, action, next_state) for pred in predecessors])
        reward_fn = self._graph.nodes[node]['reward_fn']
        score = reward_fn(state, action, next_state)
        is_satisfied = score > 0
        is_enabled = is_enabled and is_satisfied
        nx.set_node_attributes(self._graph, {node: is_enabled}, 'enabled')
        nx.set_node_attributes(self._graph, {node: score}, 'score')
        return is_enabled

    def __call__(self, state, action=None, next_state=None) -> float:
        is_enabled = self._evaluate_graph(node=self._reward_label, state=state, action=action, next_state=next_state)
        if is_enabled:
            nodes_in_hierarchy = nx.descendants(self._graph.reverse(), source=self._reward_label)
            nodes_in_hierarchy.add(self._reward_label)
            return sum(self._graph.nodes[v]['score'] for v in nodes_in_hierarchy)
        else:
            return 0

if __name__ == '__main__':
    graph = GraphBasedReward(reward_of_interest='T_orig')
    graph.add_reward('S_coll', reward_fn=reward_fns.CollisionReward())
    graph.add_reward('S_fall', reward_fn=reward_fns.ContinuousFalldownReward(theta_limit=0.3))
    graph.add_reward('H_nfeas', reward_fn=reward_fns.Indicator(reward_fns.CheckOvercomingFeasibility(obstacle_y=1, axle_y=1, feasible_height=5), negate=True))
    graph.add_reward('H_feas', reward_fn=reward_fns.Indicator(reward_fns.CheckOvercomingFeasibility(obstacle_y=1, axle_y=1, feasible_height=5)))
    graph.add_reward('T_orig', reward_fn=reward_fns.ReachTargetReward(x_target=0, x_target_tol=0.3))
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
    for x in range(1000000):
        reward = graph(state=env.observation_space.sample())
    x = None