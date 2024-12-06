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
import os
import torch
import logging
import argparse
import numpy as np
import onnxruntime as ort
from datasets import load_dataset
import onnx
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaTokenizer
from onnxruntime.quantization import QuantType
from onnx_neural_compressor.quantization import matmul_4bits_quantizer, QuantFormat, quantize
from src.compression.calibration import CalibDataloader


def quantize_model(model, output_model, qmethod):
    #data_reader = CalibDataloader("NeelNanda/pile-10k", model, pad_max=196, batch_size=1)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    data_reader = CalibDataloader(dataset, model, pad_max=196, batch_size=1)

    if qmethod == "smoothquant":
        qconfig = config.StaticQuantConfig(calibration_data_reader=data_reader,
                        quant_format=QuantFormat.QOperator,
                        activation_type=QuantType.QUInt4,
                        weight_type=QuantType.QInt4,
                        op_types_to_quantize=["MatMul"],
                        use_external_data_format=True,
                        extra_options={"SmoothQuant": True,
                                       "SmoothQuantAlpha": 0.6,
                                       "OpTypesToExcludeOutputQuantization": ["MatMul"]})
        quantize(model.path,
                        output_model.path, 
                        qconfig)
    elif qmethod == "RTN":
        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig(quant_format=QuantFormat.QDQ)
    
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model.path, algo_config=algo_config, providers=["CPUExecutionProvider"], optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL, block_size=32, is_symmetric=False)
        quant.process()
        print("processed.")
        onnx.save_model(quant.model, output_model.path, save_as_external_data=True)
    elif qmethod == "GPTQ":
        algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(calibration_data_reader=data_reader)
    
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model.path, algo_config=algo_config)
        quant.process()
        print("processed.")
        onnx.save_model(quant.model, output_model.path, save_as_external_data=True)
    elif qmethod == "AWQ":
        algo_config = matmul_4bits_quantizer.AWQWeightOnlyQuantConfig(calibration_data_reader=data_reader)
    
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model.path, algo_config=algo_config, providers=["CPUExecutionProvider"])
        quant.process()
        onnx.save_model(quant.model, output_model.path, save_as_external_data=True)
