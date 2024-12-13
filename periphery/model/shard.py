from onnx import helper
import onnx

from periphery.model.model import PeriModel
import periphery.utils.dag as dag

from periphery.utils.partition import get_partitions_spectral
from periphery.utils.topology import infer_topology

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

    partitions = get_partitions_spectral(model.graph.node, n_shards)


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
