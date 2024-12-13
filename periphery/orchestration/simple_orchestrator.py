from periphery.orchestration.orchestrator import Orchestrator

class SimpleOrchestrator(Orchestrator):
    def __init__(self, submodels, shard_graph, registered_nodes):
        #super().__init__(submodels, shard_graph, registered_nodes)
        self.submodels = submodels
        self.shard_graph = shard_graph
        self.registered_nodes = registered_nodes

    def get_assignments(self):
        if len(self.submodels) > len(self.registered_nodes)+1:
            raise Exception("Too many submodels for world size.")
        if len(self.submodels) == 0:
            raise Exception("No submodels included.")

        model_stack = self.shard_graph.get_parent_nodes()
        model_set = set(model_stack)
        node_stack = [x for x in self.registered_nodes]

        own_model_id = None

        assigned_models = {}
        assigned_nodes = {}

        while len(model_stack) > 0:
            model_id = model_stack.pop()
            model_set.remove(model_id)

            new_models = [x.index for x in self.shard_graph.nodes[model_id].connection_set if x not in model_set]
            model_stack += new_models
            model_set = model_set.union(set(new_models))

            if own_model_id is None:
                own_model_id = model_id
            else:
                next_node = node_stack.pop()
                assigned_models[next_node] = model_id
                assigned_nodes[model_id] =  next_node

        return own_model_id, assigned_models, assigned_nodes
