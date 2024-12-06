class Node:
    def __init__(self):
        self.connections = {}
        self.connection_set = set()

    def add_connection(self, label, nxt):
        self.connections[label] = nxt
        self.connection_set.add(nxt)

class DirectedGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_nodes(self, nodes):
        self.nodes += nodes

    def get_parent_nodes(self):
        non_parents = set()

        for i, node in enumerate(self.nodes):
            non_parents = non_parents.union(node.connection_set)

        return [i for i in range(len(self.nodes)) if i not in non_parents]
