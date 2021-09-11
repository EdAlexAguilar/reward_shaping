from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from reward_shaping.core.reward import RewardFunction


class GraphBasedReward(RewardFunction):

    def __init__(self):
        self._graph = nx.DiGraph()
        self._topologic_sorting = None
        self._compiled = False

    @staticmethod
    def from_collections(nodes: Dict[str, Tuple[RewardFunction, RewardFunction]],
                         topology: Dict[str, List[str]] = None):
        graph = GraphBasedReward()
        for node, (reward_fn, sat_fn) in nodes.items():
            graph.add_node(node, reward_fn, sat_fn)
        if topology:
            for source, targets in topology.items():
                for target in targets:
                    graph.add_dependency(source, target)
        return graph

    def add_node(self, label: str, reward_fn: RewardFunction, sat_fn: RewardFunction):
        self._graph.add_node(label, reward_fn=reward_fn, sat_fn=sat_fn)
        self._compiled = False

    def add_dependency(self, source: str, target: str):
        self._graph.add_edge(source, target)
        self._compiled = False
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError(f'Dependency {source}->{target} introduced cycle.')

    def _compile_graph(self):
        # compile graph: compute top sorting, ancestors, and node layers (for rendering)
        self._topologic_sorting = list(nx.topological_sort(self._graph))
        layers, ancestors = {}, {}
        for node in self._topologic_sorting:
            preds = list(self._graph.predecessors(node))
            ancestors[node] = nx.dag.ancestors(self._graph, node)
            layers[node] = 0 if len(preds) == 0 else max([layers[pred] for pred in preds]) + 1
        nx.set_node_attributes(self._graph, layers, "layer")  # used for rendering as multi-layer graph
        nx.set_node_attributes(self._graph, ancestors, "ancestors")  # used for computing reward
        self._compiled = True

    def _evaluate_node(self, node: str, state, action, next_state, info):
        # evaluate reward, satisfaction and final score of the node
        assert self._compiled == True
        nodes = self._graph.nodes
        reward = nodes[node]['reward_fn'](state, action, next_state, info)
        sat = nodes[node]['sat_fn'](state, action, next_state, info)
        ancestor_sats = [nodes[ancestor]['sat'] for ancestor in nodes[node]['ancestors']]
        attrs = {'reward': reward, 'sat': sat, 'score': np.prod(ancestor_sats) * reward}
        for attr, value in attrs.items():
            nx.set_node_attributes(self._graph, {node: value}, attr)

    def _evaluate_graph(self, state, action, next_state, info):
        # evaluate the score (ie, reward, sat, score) of each node in the dag
        if not self._compiled:
            self._compile_graph()
        self._reset_graph()  # reset scores
        for node in self._topologic_sorting:
            self._evaluate_node(node, state, action, next_state, info)

    def _compute_rewards(self) -> float:
        # sum up the final scores of all the nodes
        rewards = sum(self._graph.nodes[v]['score'] for v in self._graph.nodes)
        return rewards

    def _reset_graph(self):
        for node in self._graph.nodes:
            attributes = self._graph.nodes[node]
            for attr in ['reward', 'sat', 'score']:
                if attr in attributes:
                    attributes.pop(attr)

    def render(self):
        if not self._compiled:
            self._compile_graph()
        positioning = nx.multipartite_layout(self._graph, subset_key="layer")
        pos_labels = {node: (pos[0], pos[1] - .05) for node, pos in positioning.items()}
        nx.draw(self._graph, pos=positioning)
        nx.draw_networkx_edges(self._graph, positioning)
        nx.draw_networkx_labels(self._graph, pos_labels)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        self._evaluate_graph(state=state, action=action, next_state=next_state, info=info)
        reward = self._compute_rewards()
        return reward
