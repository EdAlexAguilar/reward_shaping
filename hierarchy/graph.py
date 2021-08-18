from collections import defaultdict
from time import sleep
from typing import List, Callable, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



class HierarchicalGraph(nx.DiGraph):
    """
        It is a DAG where each node has associated:
            - a score function (s: S->R),
            - an valuation function (v: S->B).
        The reward is then defined as the sum of scores of enabled nodes.
        A node is enabled iff FORALL parents p . p.v(state) == True.
    """

    def __init__(self, labels: List[str], score_functions: List[Callable[..., float]],
                 val_functions: List[Callable[..., float]], edge_list: List[Tuple[str, str]]):
        super(HierarchicalGraph, self).__init__()
        assert len(set(labels)) == len(labels), \
            f"labels must be unique identifiers, {labels}"
        assert all([e[0] in labels and e[1] in labels for e in edge_list]), \
            f"edge list must only use existing labels, {edge_list}"
        # define nodes
        self.labels = np.array(labels)
        self.score = score_functions
        self.valuation = val_functions
        self.add_nodes_from(self.labels)
        # define edges
        self.add_edges_from(edge_list)
        assert nx.is_directed_acyclic_graph(self)
        # utilities
        self.lab2id = {l: n for n, l in enumerate(labels)}
        self.id2lab = {n: l for n, l in enumerate(labels)}
        # topological sort
        self.top_sorting = list(nx.topological_sort(self))
        self.ancestors = {label: set() for label in labels}
        for node in self.top_sorting:
            for pred in self.predecessors(node):
                self.ancestors[node].update(self.ancestors[pred])
                self.ancestors[node].add(pred)
        # layer for visualization layout
        layers = {}
        for node in self.top_sorting:
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                layers[node] = 0
            else:
                layers[node] = max([layers[pred] for pred in preds]) + 1
        nx.set_node_attributes(self, layers, "layer")

    def render(self, colors=None):
        positioning = nx.multipartite_layout(self, subset_key="layer")
        pos_labels = {node: (pos[0], pos[1] - .1) for node, pos in positioning.items()}
        if colors is None:
            nx.draw(self, pos=positioning)
        else:
            plt.clf()
            color_map = {color: [] for color in colors.values()}
            for node, color in colors.items():
                color_map[color].append(node)
            for color, node_list in color_map.items():
                nx.draw_networkx_nodes(self, positioning, nodelist=node_list, node_color=color)
        nx.draw_networkx_edges(self, positioning)
        nx.draw_networkx_labels(self, pos_labels)
        plt.pause(0.001)

if __name__=="__main__":
    from hierarchy.test_graph import TestHierarchicalGraph
    test = TestHierarchicalGraph()
    test.test_topological_sorting()
    pass