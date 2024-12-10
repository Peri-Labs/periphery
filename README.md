# Periphery

Welcome to **Periphery**. This project is being actively developed and is shared to provide early access for external parties.

Periphery is designed to support advanced workflows for model compression, distributed inference, and ONNX graph processing. Here's a quick overview of its structure:

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

## Notes

- **Beta Status:**  
  Periphery is an experimental project. Features are subject to change, and certain components may be incomplete or undergoing active development.  

- **Feedback and Collaboration:**  
  Your feedback is invaluable as we refine and expand the library. Please share your thoughts, suggestions, or issues to help shape its future.

---

Thank you for exploring Periphery. Stay tuned for updates!

