

def wait_for_network(server, args):
    if args.master:
        server.wait_for_nodes(args.num_shards)
    else:
        master_url = f"http://{args.master_ip}:{args.master_port}"
        server.wait_for_master(master_url)
        server.register_self(master_url)

    print("Finished setup, server is live...")
