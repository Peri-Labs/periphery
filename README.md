![Periphery Logo](/periphery_logo.png)

# Periphery

**Periphery** is an experimental framework that focuses on advanced ONNX-based model workflows:

- **Model Compression:** Integrations with Intel’s Neural Compressor and custom quantization tooling.  
- **Distributed Inference:** Scale inference across multiple hosts or devices, dynamically splitting and orchestrating model execution.  
- **ONNX Graph Processing:** Utilities for partitioning, sharding, and optimizing ONNX graphs to handle complex inference topologies.

**Status:** Active development, interfaces subject to change.

---

## Key Features

- **Model Sharding & Partitioning:**  
  Break large ONNX models into multiple shards. Distribute these shards to different nodes for efficient parallel inference and reduced memory overhead per node.

- **Quantization & Calibration:**  
  Use integrated calibration tools and quantizers to compress models. Reduce inference latency and hardware requirements while aiming to preserve acceptable accuracy.

- **Distributed Inference Orchestration:**  
  Automate node discovery, registration, and shard assignment. Scale from a single machine to a cluster of nodes running in parallel.

- **ONNX Graph Utilities:**  
  Understand and manipulate ONNX models with a rich set of graph utilities. Perform spectral partitioning, DAG construction, and node-level transformations to fine-tune model execution flow.

---

## Repository Structure

- **`app/`**  
  A user-friendly terminal application for accessing and interacting with Periphery.  
  *Status: In progress.*

- **`periphery/`**  
  Core libraries and utilities powering Periphery's functionality. The submodules include:  
  *Status: In progress.*
  - **`compression/`**  
    A compression library frontend, compatible with Intel's Neural Compressor library and a custom fork tailored for specific needs.  
  - **`distributed/`**  
    A distributed inference library enabling efficient scaling of model inference across multiple nodes.  
  - **`model/`**  
    Tools for model sharding and processing ONNX graphs to streamline model optimization and execution.  
  - **`utils/`**  
    Utility code for graph operations, file I/O, and other common tasks.

- **TODOs:**
  - **Networking:** Enhance and possibly modularize the networking backend to support alternative communication protocols.
  - **Test Cases:** Expand coverage with unit and integration tests.
  - **GenAI Tokenizer Support:** Integrate tokenizers for advanced generative AI models.

---

## Installation

**Prerequisites:**
- Python 3.8+
- ONNX Runtime
- PyTorch (for calibration and certain compression routines)
- Intel Neural Compressor (optional, for advanced quantization workflows)

**Setup:**
```bash
gh repo clone Peri-Labs/periphery
cd periphery
pip install -r requirements.txt
```

## Example Usage

Below are examples for setting up both a **master node** and a **child node** in a distributed inference setup.

Running the Master Node:
```bash
# Master Node
python main.py --master --model_path /path/to/model.onnx --num_shards 4
```

Running the Master Node:
```bash
# Child Node
python main.py --master_ip 192.168.1.100 --master_port 29500 --ip 192.168.1.101 --port 29501
```
---

## Advanced Features

### Sharding and Distribution
Use `shard_onnx_model` to split models into *N* shards. The master node orchestrates which child nodes get which shard. Inferencing requests will flow through this distributed shard graph.

### Quantization and Calibration
With `periphery/compression/calibration.py` and `periphery/compression/quantize.py`, Periphery supports various quantization strategies:
- **smoothquant**
- **RTN**
- **GPTQ**

These strategies reduce model size and speed up inference.

### Graph Utilities
Leverage DAG operations, spectral clustering, and topology inference from `periphery/utils/` for complex graph manipulations and experiments.

---

## Roadmap

### Networking Improvements
Investigate **gRPC** or **WebSockets** to improve reliability, security, and throughput.

### Test Coverage
Develop more comprehensive tests, including performance and stress tests, to ensure stability.

### GenAI Tokenizer Integration
Incorporate tokenizers and LLM-specific preprocessing for text-based generative models.

---

## Feedback & Contributions

### Beta Status
Periphery is under rapid development. Expect instability and breaking changes as the project evolves.

### Contributing
- **Report bugs**, request features, or suggest improvements via [GitHub Issues](https://github.com/Peri-Labs/periphery/issues). 
- Submit **Pull Requests** to help improve code quality, documentation, and features.

### License
This software is distributed under the BSD-3-Clause-Clear license. Read [this](LICENSE) for more details. 

---

## Notes

- **Beta Status:**  
  Periphery is an experimental project. Features are subject to change, and certain components may be incomplete or undergoing active development.  

- **Feedback and Collaboration:**  
  Your feedback is invaluable as we refine and expand the library. Please share your thoughts, suggestions, or issues to help shape its future.

---

Thank you for exploring Periphery. Stay tuned for updates!

