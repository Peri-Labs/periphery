from onnx import helper
import onnx

from periphery.model.model import PeriModel
import periphery.utils.dag as dag

def get_partitions(model, n_shards): 
    """
    Find a partition of the model layers into n_shards

    Parameters:
    - model: The ONNX ModelProto to shard
    - n_shards: Number of shards to split the model into.
    """
    nodes = model.graph.node
    total_nodes = len(nodes)

    shard_size = total_nodes // n_shards

    partitions = []

    for shard_no in range(n_shards):
        partitions.append(nodes[(shard_no*shard_size):((shard_no+1)*shard_size)])

    if shard_size * n_shards < total_nodes:
        shortfall = total_nodes - (shard_size*n_shards)
        partitions[-1] += nodes[(total_nodes-shortfall):]

    return partitions

def infer_topology(shard_inputs, shard_outputs):
    """
    Return a DAG representing the connections between shards

    Parameters:
    - shard_inputs: A list of inputs into each submodel
    - shard_outputs: A list of outputs into each submodel
    """
    graph = dag.DirectedGraph()

    n_shards = len(shard_inputs)

    nodes = [dag.Node() for i in range(n_shards)]

    graph.add_nodes(nodes)
    
    input_mapping = {}
    output_mapping = {}

    for shard_no, input_names in enumerate(shard_inputs):
        for input_name in input_names:
            input_mapping[input_name] = shard_no

    for shard_no, output_names in enumerate(shard_outputs):
        for output_name in output_names:
            output_mapping[output_name] = shard_no

    for shard_no, shard in enumerate(shard_inputs):
        for output in list(shard_outputs[shard_no]):
            if output in input_mapping:
                next_node = nodes[input_mapping[output]]
                nodes[shard_no].add_connection(output, next_node)
    
    return graph


def shard_onnx_model(peri_model, n_shards, output_paths):
    """
    Shard an ONNX model into N smaller models.

    Parameters:
    - model_path: Path to the input ONNX model.
    - n_shards: Number of shards to split the model into.
    - output_paths: List of file paths for the output sharded models.
    """
    # Load the ONNX model
    model = peri_model.load_model()
    graph = model.graph
    nodes = graph.node

    total_nodes = len(nodes)

    if n_shards > total_nodes:
        raise ValueError("Number of shards exceeds the number of nodes in the model.")

    initializers = {init.name: init for init in graph.initializer}

    partitions = get_partitions(model, n_shards)


    all_inputs = []
    all_outputs = []

    for shard_no, partition in enumerate(partitions):
        shard_inputs = set()
        shard_outputs = set()
        shard_initializers = set()
        for node in partition:
            shard_inputs = shard_inputs.union(set(node.input))
            shard_outputs = shard_outputs.union(set(node.output))
            shard_initializers = shard_initializers.union(set([x for x in node.input if x in initializers]))
        
        intermediates = shard_inputs.intersection(shard_outputs)

        shard_inputs = shard_inputs.difference(intermediates)
        shard_outputs = shard_outputs.difference(intermediates)
        shard_inputs = shard_inputs.difference(shard_initializers)

        all_inputs.append(shard_inputs)
        all_outputs.append(shard_outputs)

        shard_inputs = [helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, None) for x in shard_inputs]
        shard_outputs = [helper.make_tensor_value_info(x, onnx.TensorProto.FLOAT, None) for x in shard_outputs]
        shard_initializers = [initializers[x] for x in shard_initializers]

        shard_graph = helper.make_graph(
            nodes=partition,
            name=f"shard_{shard_no}",
            inputs=shard_inputs,
            outputs=shard_outputs,
            initializer=shard_initializers,
        )

        # Create a new ONNX model
        shard_model = helper.make_model(shard_graph, producer_name="onnx_sharder")
        shard_model.ir_version = model.ir_version
        shard_model.opset_import.clear()
        shard_model.opset_import.extend(model.opset_import)

        # Save the sharded model
        onnx.save(shard_model, output_paths[shard_no])

    return infer_topology(all_inputs, all_outputs)
