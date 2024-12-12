import app.args as peri_args
import app.setup as peri_setup

from periphery.distributed.http_server.server import Server
from periphery.distributed.node import Node

from time import sleep

import threading

if __name__ == "__main__":
    args = peri_args.Args()

    node = Node(args.ip, args.port)
    server = Server(node)
    
    server.run(host=args.ip, port=args.port)

    peri_setup.wait_for_network(server, args)

    print("Main thread is still running...")
    try:
        server.pause_signal()
    except KeyboardInterrupt:
        server.stop()
