import periphery.utils.dag as dag

def infer_topology(inputs, outputs):
    """
    Return a DAG representing the connections between submodels of a model, or nodes of a model

    Parameters:
    - inputs: A list of input lists (one for each shard/node)
    - outputs: A list of output lists (one for each shard/node)
    """
    graph = dag.DirectedGraph()

    n_nodes = len(inputs)

    nodes = [dag.Node() for i in range(n_nodes)]

    graph.add_nodes(nodes)
    
    input_mapping = {}
    output_mapping = {}

    for input_no, input_names in enumerate(inputs):
        for input_name in input_names:
            input_mapping[input_name] = input_no

    for input_no, output_names in enumerate(outputs):
        for output_name in output_names:
            output_mapping[output_name] = input_no

    for input_no, _ in enumerate(inputs):
        for output in list(outputs[input_no]):
            if output in input_mapping:
                next_node = nodes[input_mapping[output]]
                nodes[input_no].add_connection(output, next_node)
    
    return graph
