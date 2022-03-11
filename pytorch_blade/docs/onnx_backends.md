# TorchBlade ONNX Backends Overview

[ONNX](https://onnx.ai/) is an open format built to represent machine learning models.
It has an activate community and a thriving ecosystem due to transparency, neutrality and interoperability
to different frameworks.

Many converters, accelerators and hardware, such as ONNX-TensorRT, are provided by the ONNX community.
These tools facilitate users to optimize and deploy deep learning models on different hardwares.

And we also found that many ONNX models are exported from PyTorch, since PyTorch is one of the most popular deep
learning frameworks and natively support exporting models to ONNX.

However, there are some deficiencies to deploy models via ONNX:

- Some control flow, dynamic shapes, and mutation operations in PyTorch can't be correctly exported via tracing
- Vendors usually privide limited supports on ONNX operators. For example, ONNX-TensorRT only supports about 129 ONNX operators out of 160+
- It's not practical to provide a full conversion coverage of all operations from PyTorch
- The compatibility among tools used in the optimization and conversion pipeline is poor usually

Thereby it's not easy for users to success deploy their models on devices via ONNX.
To solve these problems mannually is not fancy and will kill a of times.

## Automate and Simplify the Process

TorchBlade mainly improve the usability and robustness of the process with the following stratagies:

- Ensure the correctness based on TorchScript representation and runtime
- Grarantee the robustness with clustering and fallbacks
- Automate the optimization process at best efforts

### Benefits of TorchScript

### Clustering and Fallbacks

### Automate the Process
