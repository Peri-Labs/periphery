

class Orchestrator:
    def __init__(self, submodels, shard_graph, registered_nodes):
        self.submodels = submodels
        self.shard_graph = shard_graph
        self.registered_nodes = registered_nodes

    def get_assignments(self):
        pass
