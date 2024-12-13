import collections

import numpy as np

class Node:
    def __init__(self, index=None):
        self.connection_labels = collections.defaultdict(set)
        self.label_to_connection = {}
        self.connection_set = set()
        self.index = index

        self.external_inputs = set()

    def add_connection(self, label, nxt):
        self.connection_labels[nxt].add(label)
        self.label_to_connection[label] = nxt
        self.connection_set.add(nxt)

class DirectedGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        node.index = len(self.nodes)
        self.nodes.append(node)

    def add_nodes(self, nodes):
        total_nodes = len(self.nodes)
        for i, node in enumerate(nodes):
            node.index = total_nodes + i
        self.nodes += nodes

    def get_parent_nodes(self):
        non_parents = set()

        for i, node in enumerate(self.nodes):
            non_parents = non_parents.union(node.connection_set)

        return [i for i, x in enumerate(self.nodes)  if x not in non_parents]

    def undirected_adjacency_matrix(self):
        n_nodes = len(self.nodes)
        adj = np.zeros((n_nodes, n_nodes))
        
        for node_idx, node in enumerate(self.nodes):
            for next_node in node.connection_set:
                adj[node_idx, next_node.index] = 1
                adj[next_node.index, node_idx] = 1
        return adj
