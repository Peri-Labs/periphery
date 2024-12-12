from periphery.distributed.task_manager import TaskManager

class Node:
    def __init__(self, host_ip, host_port, task_manager=None):
        self.host_ip = host_ip
        self.host_port = host_port
        
        if not task_manager:
            self.task_manager = TaskManager()
        else:
            self.task_manager = task_manager

    def get_url(self, protocol):
        if protocol == "https":
            return f"https://{self.host_ip}:{self.host_port}"

        raise Exception(f"Protocol {protocol} not supported")
