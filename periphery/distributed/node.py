class Node:
    def __init__(self, host_ip, host_port, server, task_manager):
        self.host_ip = host_ip
        self.host_port = host_port

        self.server = server
        self.task_manager = task_manager

    def run(self):
        self.server.run()
