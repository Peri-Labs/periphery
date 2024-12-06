from io import BytesIO
import numpy as np

class TaskManager:
    def __init__(self, model):
        self.model = model
        self.input_requests = {}
        self.outputs = {}

        self.input_names = [x.name for x in self.model.get_inputs()]

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
            to_remove.add(infer_id)

        for infer_id in to_remove:
            del self.input_requests[infer_id]

    def get_buffer(self, infer_id):
        buffer = BytesIO()
        np.savez(buffer, **self.outputs[infer_id])
        buffer.seek(0)

        return buffer, f"output_{infer_id}.npz"
