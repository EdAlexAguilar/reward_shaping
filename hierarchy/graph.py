from collections import defaultdict
from time import sleep

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class HierarchicalGraph(nx.DiGraph):
    def __init__(self, vertices, edges):
        # todo: create a dictionaries F: node_id -> (lambda->real), IND: node_id -> (lambda->bool)
        super(HierarchicalGraph, self).__init__()
        self.add_nodes_from([i for i in range(vertices)])
        self.add_edges_from([(np.random.randint(0, vertices), np.random.randint(0, vertices)) for _ in range(edges)])

    def render(self):
        nx.draw_planar(self)
        plt.show()

    # todo: define a reward by using the topological sort of the functions

if __name__=="__main__":
    g = HierarchicalGraph(20, 10)
    g.render()