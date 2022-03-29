# TorchBlade

TorchBlade is an end-to-end performant compiler tailored to Python subset
language for deep learning. Currently, TorchBlade mainly focuses on PyTorch
optimization and compilation for inference workloads.

## Documents for BladeDISC Developers

TorchBlade is tailored for PyTorch and BladeDISC. It chooses [MHLO](https://github.com/tensorflow/mlir-hlo) as the tensor level "hub" IR with the considerations:

- Well established by the XLA community and continuous improvements from the MHLO project
- The finer granularity and the expressive ability is suitable to bridge PyTorch operators and lower level passes
- The support for dynamic shapes is well established by BladeDISC-TensorFlow

Please reference to the following documents if you are interested in details about TorchBlade and BladeDISC.

- [TorchBlade Overview](/docs/developers/bladedisc_torch_overview.md)
- [How To Add a New Torch Operator Converter](/docs/developers/torch_add_a_new_converter.md)
- [Optimize and Inference BERT With TorchBlade](/docs/tutorials/torch_bert_inference.md)
- [BladeDISC Pass Pipeline Walkthough](/docs/developers/pass_pipeline.md)
- [Runtime Abstraction Layer](/docs/developers/runtime_abstraction_layer.md)
- [Build BladeDISC for PyTorch](/docs/build_from_source.md#build-bladedisc-for-pytorch.md)

## Docuemnts for ONNX Backends

[ONNX](https://onnx.ai/) is an open format built to represent machine learning models.
As a result, vendors' accelerators and hardware widely adopted it. To make users feasible to various ONNX tools, TorchBlade also provides a compiler path targeted to ONNX besides MHLO.

Please reference to the followering documents if you are interested in TorchBlade's ONNX backends.

- [TorchBlade ONNX Backends Overview](docs/onnx_backends.md)