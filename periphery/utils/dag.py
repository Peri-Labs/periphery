import collections

class Node:
    def __init__(self, index=None):
        self.connection_labels = collections.defaultdict(set)
        self.label_to_connection = {}
        self.connection_set = set()
        self.index = index

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
