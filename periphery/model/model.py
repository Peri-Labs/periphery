import onnxruntime as ort
import onnx
from transformers import AutoTokenizer, AutoConfig

import os

class ModelSupplement:
    def __init__(self):
        pass

    def save(self):
        pass

class LLMSupplement(ModelSupplement):
    def __init__(self, tokenizer, config):
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config

    def save(self, output_folder):
        self.tokenizer.save_pretrained(output_folder)
        self.config.save_pretrained(output_folder)

    @classmethod 
    def from_pretrained(cls, model_name):
        supplement = LLMSupplement(None, None)
        supplement.tokenizer = AutoTokenizer.from_pretrained(model_name)
        supplement.config = AutoConfig.from_pretrained(model_name)

class PeriModel:
    def __init__(self, path, supplement=None):
        self.path = path
        self.supplement = supplement

        self.inputs = None
        self.outputs = None
        self.loaded_model = None
        self.session = None

    def get_inputs(self):
        if not self.inputs:
            if self.session:
                self.inputs = self.session.get_inputs()
            else:
                session = ort.InferenceSession(self.path)
                self.inputs = session.get_inputs()

        return self.inputs

    def get_outputs(self):
        if not self.outputs:
            if self.session:
                self.outputs = self.session.get_outputs()
            else:
                session = ort.InferenceSession(self.path)
                self.outputs = session.get_outputs()

        return self.outputs

    def save_supplement(self):
        self.supplement.save(os.path.dirname(self.path))

    def load_model(self):
        self.loaded_model = onnx.load(self.path)
        return self.loaded_model

    def get_onnx_model(self):
        if not self.loaded_model:
            return self.load_model()
        
        return self.loaded_model

    def infer(self, input_dict):
        if not self.session:
            self.session = ort.InferenceSession(self.path)
        output_names = [x.name for x in self.get_outputs()]
        return {name: res for name, res in zip (output_names, self.session.run(output_names, input_dict))}

    def get_data_file(self):
        return self.path + ".data"
