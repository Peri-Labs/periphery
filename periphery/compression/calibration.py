# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
import torch
import numpy as np
from datasets import load_dataset
import onnxruntime as ort
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from onnxruntime.quantization.calibrate import CalibrationDataReader

class CalibDataloader(CalibrationDataReader):
    def __init__(self, dataset, model, pad_max=196, batch_size=1, sampling_size=8):
        self.model = model
        self.tokenizer = self.model.supplement.tokenizer
        tokenize_function = lambda x: self.tokenizer(x["text"])

        self.pad_max = pad_max
        self.batch_size=batch_size
        #dataset = load_dataset(dataset, split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset.select(range(sampling_size))
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

        self.key_value_input_names = [key.name for key in model.get_inputs() if (".key" in key.name) or (".value" in key.name)]
        self.use_cache = len(self.key_value_input_names) > 0
        self.use_position_ids = "position_ids" in model.get_inputs()

        self.processed_data = iter(self.process_data(self.dataloader))

    def collate_batch(self, batch):
        input_ids_padded = []
        attention_mask_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)
    
    def process_data(self, dataloader):
        processed_data = []
        for (input_ids, attention_mask) in dataloader:
            ort_input = {}
            if not self.use_cache:
                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
                ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype('int64')
            else:
                num_attention_heads = self.model.supplement.config.num_key_value_heads
                embed_size_per_head = self.model.supplement.config.hidden_size // self.model.supplement.config.num_attention_heads
                shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                key_or_value = np.zeros(shape, dtype=np.float16)

                for key_value_input_name in self.key_value_input_names:
                    ort_input[key_value_input_name] = key_or_value

                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype('int64')
                ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

            input_shape = ort_input["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.float16).unsqueeze(0).view(-1, input_shape[-1])
            if self.use_position_ids:
                ort_input["position_ids"] = position_ids.numpy()
            processed_data.append(ort_input)
        return processed_data


    def get_next(self) -> dict:
        return next(self.processed_data, None)

    def rewind(self):
        self.processed_data = iter(self.process_data(self.dataloader))
