import periphery.utils.dag as dag

from periphery.utils.topology import infer_topology

from sklearn.cluster import SpectralClustering

def get_partitions_simple(nodes, n_shards):
    # partition the model graph using a greedy algorithm
    total_nodes = len(nodes)

    shard_size = total_nodes // n_shards

    partitions = []

    for shard_no in range(n_shards):
        partitions.append(nodes[(shard_no*shard_size):((shard_no+1)*shard_size)])

    if shard_size * n_shards < total_nodes:
        shortfall = total_nodes - (shard_size*n_shards)
        partitions[-1] += nodes[(total_nodes-shortfall):]

    return partitions

def get_partitions_spectral(nodes, n_shards):
    # use spectral decomposition to partition the model graph
    node_inputs = [x.input for x in nodes]
    node_outputs = [x.output for x in nodes]

    dag = infer_topology(node_inputs, node_outputs)

    adj = dag.undirected_adjacency_matrix()

    sc = SpectralClustering(n_shards, affinity="precomputed", n_init=100)
    sc.fit(adj)

    # make sure the first node is on partition 0, swap if needed
    swap_val = int(sc.labels_[0])
    swap_if_needed = lambda x: x if not (x in [0, swap_val]) else {0: swap_val, swap_val: 0}[x]
    partition_map = [swap_if_needed(int(sc.labels_[i])) for i in range(len(nodes))]

    partitions = [list() for i in range(n_shards)]

    for node_no, partition_no in enumerate(partition_map):
        partitions[partition_no].append(nodes[node_no])

    return partitions
