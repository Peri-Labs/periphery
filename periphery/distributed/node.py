class Node:
    def __init__(self, host_ip, host_port, task_manager):
        self.host_ip = host_ip
        self.host_port = host_port

        self.task_manager = task_manager
