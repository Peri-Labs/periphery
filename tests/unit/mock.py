import collections

import numpy as np

class MockInputs:
    def __init__(self):
        self.name = "mock"
        self.shape = [10, 10]
        self.type = "float"

class MockModel:
    def __init__(self):
        self.inputs = [MockInputs()]
        self.path = "tests/unit/mock_files/mock_model.onnx"

    def get_inputs(self):
        return self.inputs

class MockNode:
    def __init__(self):
        self.task_manager = MockTaskManager()

class MockTaskManager:
    def __init__(self):
        self.model = MockModel()
        self.input_requests = {}
        self.outputs = {}

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
        pass

    def get_buffer(self, infer_id):
        if infer_id not in self.outputs:
            raise Exception("Inference not completed")
        output = {"a": np.zeros(5)}
        buffer = io.BytesIO(output)
        return buffer, f"output_{infer_id}.bin"
