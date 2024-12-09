from io import BytesIO
import numpy as np

import requests
import collections

class TaskManager:
    def __init__(self, model):
        self.model = model
        self.input_requests = {}
        self.outputs = {}

        self.input_names = [x.name for x in self.model.get_inputs()]

        self.children = []
        self.child_output_mappings = collections.defaultdict(list)

    def clear_model(self):
        self.model = None

    def clear_children(self):
        self.children = []
        self.child_output_mappings = collections.defaultdict(list)

    def submit_input(self, input_tensors, infer_id):
        if infer_id in input_tensors:
            self.input_requests[infer_id].update(input_tensors)
        else:
            self.input_requests[infer_id] = input_tensors

    def check_for_completion(self):
        to_remove = set()
        for infer_id in self.input_requests.keys():
            for iname in self.input_names:
                if iname not in self.input_requests[infer_id]:
                    return

            self.outputs[infer_id] = self.model.infer(self.input_requests[infer_id])
            self.send_to_children(infer_id, self.children, self.child_output_mappings)
            to_remove.add(infer_id)

        for infer_id in to_remove:
            del self.input_requests[infer_id]

    def get_buffer(self, infer_id):
        buffer = BytesIO()
        np.savez(buffer, **self.outputs[infer_id])
        buffer.seek(0)

        return buffer, f"output_{infer_id}.npz"

    def send_to_children(self, infer_id, children, child_output_mappings):
        for child, outputs in child_output_mappings.items():
            url = f"{children[child]}/submit_input/{infer_id}"
            file = self.get_selected_buffer(infer_id, outputs, child)
            requests.post(url, files={"file": file})
    
    def get_selected_buffer(self, input_id, output_names, output_id):
        buffer = BytesIO()
        selected_output = {k: v for k,v in self.outputs[infer_id].items() if k in output_names}

        np.savez(buffer, **selected_output)
        buffer.seek(0)

        return buffer, f"output_{infer_id}_{output_id}.npz"
