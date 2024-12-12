

async def wait_for_network(server, args):
    if args.master:
        await server.wait_for_nodes(args.num_shards)
    else:
        await server.wait_for_master()
