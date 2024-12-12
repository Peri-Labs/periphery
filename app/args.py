import argparse

class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--master_ip", type=str, default="127.0.0.1")
        parser.add_argument("--master_port", type=int, default=29500)
        parser.add_argument("--model_path", default=None, type=str, required=True, help="Directory containing .onnx model files. This may be empty if this is not the master node.")
        parser.add_argument("--ip", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=29500)
        parser.add_argument("--master", action="store_true")
        parser.add_argument("--num_shards", type=int, default=1)

        args = parser.parse_args()

        self.master_ip = args.master_ip
        self.master_port = args.master_port
        self.ip = args.ip
        self.port = args.port
        self.master = args.master or (self.master_ip == self.ip and self.master_port == self.port)
        self.model_path = args.model_path
        self.num_shards = args.num_shards
