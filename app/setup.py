from periphery.model.model import PeriModel
import periphery.model.shard as shard

import os

import pathlib

def wait_for_network(server, args):
    if args.master:
        server.wait_for_nodes(args.num_shards)
    else:
        master_url = f"http://{args.master_ip}:{args.master_port}"
        server.wait_for_master(master_url)
        server.register_self(master_url)

    print("Finished setup, server is live...")

def shard_and_distribute_model(server, args):
    if not args.master:
        return

    model = PeriModel(args.model_path)

    model_dir = os.path.dirname(os.path.realpath(args.model_path))
    shard_dir = os.path.join(model_dir, "shards")
    pathlib.Path(shard_dir).mkdir(parents=True, exist_ok=True)

    shard_paths = [os.path.join(shard_dir, f"shard_{i}") for i in range(args.num_shards)]

    shard_graph = shard.shard_onnx_model(model, args.num_shards, shard_paths)
    
    submodels = [PeriModel(shard_path) for shard_path in shard_paths]

    server.assign_shards(submodels, shard_graph)
