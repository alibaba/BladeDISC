# TorchBlade

TorchBlade is an end-to-end performant compiler tailored to Python subset
language for deep learning. Currently, TorchBlade mainly focuses on PyTorch
optimization and compilation for inference workloads.

## Tutorials and Documents for TorchBlade Developers

- [TorchBlade Overview](/docs/developers/bladedisc_torch_overview.md)
- [How To Add a New Torch Operator Converter](/docs/developers/add_new_torch_ops.md)
- [Optimize and Inference BERT With TorchBlade](/examples/pytorch/inference/torch_bert_tutorial.md)
- [BladeDISC Pass Pipeline Walkthough](/docs/developers/pass_pipeline.md)
- [Runtime Abstraction Layer](/docs/developers/runtime_abstraction_layer.md)

## Quick Installation

Please refer to
[Build BladeDISC for PyTorch](/docs/build_from_source.md#build-bladedisc-for-pytorch.md).

## Prerequisites

To build the TorchBlade, ensure you meet the following package requirements:

- A compiler with C++14 support
- [CUDA](https://developer.nvidia.com/cuda-toolkit) >= v10.0
- [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
- [CMake](https://github.com/Kitware/CMake/releases) >= v3.0
- [Python3](https://www.python.org/downloads/release/python-365/) >= v3.6.5
- [PyTorch](https://pytorch.org/) >= v1.6.0
- [networkx](https://networkx.org/)

## Installation

Just clone this repository and `pip3 install`. Note the `--recursive` option is
needed for the pybind11 submodule:

```bash
# Environment variables you should set:
#    CUDA_HOME
#      the cuda home directory, also the cudnn's
#
git clone --recursive git@github.com:alibaba/BladeDISC.git
cd ./pytorch-addons && python3 setup.py install
```

With the `setup.py` file included in this example, the `pip3 install` command
will invoke CMake and build the pybind11 module as specified in CMakeLists.txt.

## Development

During development, it's recommended to do "development install" instead of
regular installation. To do development install, use `python3 setup.py develop`
or `pip3 install -e .` to replace the `python3 setup.py install` command in the
above Installation section. In develop mode, python3 code changes are
immediately reflected in your environment (however, if c++ code has been
changed, you still need to redo the `python3 setup.py develop` or
`pip3 install -e .` command to trigger the re-compilation of the c++ part).

After installation, to run all the test cases, you can do
`python3 -m unittest discover tests/` or simply `pytest` (you can install pytest
via `pip3 install pytest`).
