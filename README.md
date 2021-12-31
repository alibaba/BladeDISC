# BladeDISC Introduction

* [Overview](#overview)
  + [Features and Roadmap](#features-and-roadmap)
    - [Frontend Framework Support Matrix](#frontend-framework-support-matrix)
    - [Backend Support Matrix](#backend-support-matrix)
    - [Deployment Solutions](#deployment-solutions)
  + [Numbers of Typical Workloads](#numbers-of-typical-workloads)
    - [Advantage in Dynamic Shape Workloads](#advantage-in-dynamic-shape-workloads)
* [API QuickView](#api-quickview)
  + [For TensorFlow Users](#for-tensorflow-users)
  + [For PyTorch Users](#for-pytorch-users)
* [Setup and Examples](#setup-and-examples)
* [Publications](#publications)
* [Tutorials and Documents for Developers](#tutorials-and-documents-for-developers)
* [How to Contribute](#how-to-contribute)
* [FAQ](#faq)
  + [Roadmap with mlir-hlo Project](#roadmap-with-mlir-hlo-project)
* [Contact Us](#contact-us)

## Overview

BladeDISC is an end-to-end **DynamIc Shape Compiler** project for machine
learning workloads, which is one of the key components of Alibaba's
[PAI-Blade](https://www.aliyun.com/activity/bigdata/blade). BladeDISC provides
general, transparent, and ease of use performance optimization for
TensorFlow/PyTorch workloads on GPGPU and CPU backends. The architecture
natively supports dynamic shape workloads, with many considerations in the
performance of both static and dynamic shape scenarios. It also supports
multiple and flexible deployment solutions, including both Plugin Mode inside
TensorFlow/PyTorch runtime, and Standalone Mode for AOT standalone execution.
The project is based on [MLIR](https://mlir.llvm.org/) and highly related with
[mlir-hlo](https://github.com/tensorflow/mlir-hlo) project.

Refer to [our website](https://alibaba.github.io/BladeDISC/) for more
information, including the setup tutorial, developer guide, demo examples and
documents for developers.

### Features and Roadmap

#### Frontend Framework Support Matrix

|           | TensorFlow [1] | PyTorch [2]  |
|---------- | -------------- | ------------ |
| Inference |    Yes         |    Yes       |
|  Training |    Yes [3]     |  Ongoing     |

[1] TensorFlow 1.12, 1.15, 2.4 & 2.5 are supported and fully verified. For other
versions some slight works on adaptation might be needed.

[2] 1.6.0 <= PyTorch version < 1.9.0 has been fully verified.

[3] Although supported, there's much room for improvement on Op coverage for
training workloads.

#### Backend Support Matrix

|    | Memory Intensive Part | Compute Intensive Part | End-to-End Usability |
|----------- | ------------- | ---------------------- | -------------------- |
| Nvidia GPU |    Yes        |    Yes                 |    Yes               |
| AMD GPU    |  Ongoing      |  Ongoing               |     No               |
| Hygon DCU  |    Yes        |    Yes                 |    Yes               |
|  X86       |    Yes        |  Not open-sourced yet [1]  |     No           |

[1] The compute-intensive part of the X86 backend is already supported on the
internal version. The code decoupling is ongoing and will be open-sourced soon,
same for the end-to-end usability.

#### Deployment Solutions

* Plugin Mode - BladeDISC works as a plugin of TensorFlow or PyTorch. Only the
  supported Ops are clustered and compiled, and the unsupported ones will be
  executed by the original TensorFlow or PyTorch runtime. We recommend this mode
  to most of the users for its transparency and ease of use. 

* Standalone Mode - In Standalone mode, the input workload will be compiled into
  a binary that can be executed by it self, aka, does not rely on a TensorFlow
  or PyTorch runtime. In this mode all ops must be supported. 

### Numbers of Typical Workloads

By evaluating BladeDISC using a set of typical machine learning workloads for
production purpose, DISC shows up to 3x speedup compared with
TensorFlow/PyTorch.

![Numbers](./docs/pics/numbers.png)

#### Advantage in Dynamic Shape Workloads

Specifically, for the BERT large inference on T4 we provide in the
[examples](./docs/tutorials/tensorflow_inference_and_training.md), static compiler
optimization (XLA) shows severe performance degradation due to its compilation
overhead, while DISC shows a 1.75x speedup.

| TensorFlow  |    XLA    |    DISC    |
|-------------|-----------|------------|
|   1.78 s    |   41.69s  |    1.02s   |
|   1X        |           |    1.75X   |

## API QuickView

### For TensorFlow Users

Only two lines of code are needed on native Tensorflow program as the following:

``` python
import numpy as np
import tensorflow as tf

## enable BladeDISC on TensorFlow program
import tensorflow_blade_disc as disc
disc.enable()

## construct TensorFlow Graph and run it
g = tf.Graph()
with g.as_default():
    ...
    with tf.session as sess:
        sess.run(...)
```

For more information, please refer to [QuickStart for TensorFlow
Users](./docs/quickstart.md#quickstart-for-tensorflow-users)

### For PyTorch Users

PyTorch users only need the following few lines of code to enable
BladeDISC:

``` python
import torch_blade
# construct PyTorch Module
class MyModule(nn.Module):
    ...

module = MyModule()

with torch.no_grad():
    # blade_module is the optimized module by BladeDISC
    blade_module = torch_blade.optimize(module, allow_tracing=True, model_inputs=(x, y))

# run the optimized module
blade_module(x, y)
```

`torch_blade.optimize` accepts an `nn.Module` object and outputs the
optimized module.  For more information, please refer to [Quickstart
for PyTorch Users](./docs/quickstart.md#quickstart-for-pytorch-users).

## Setup and Examples

* [How to Setup and Build from Source](./docs/build_from_source.md)
* [Use Case of TensorFlow Inference and Training](./docs/tutorials/tensorflow_inference_and_training.md)
* [Use Case of PyTorch Inference](./docs/tutorials/torch_bert_inference.md)

## Publications

* [DISC: A Dynamic Shape Compiler for Machine Learning
  Workloads](https://arxiv.org/pdf/2103.05288.pdf)

## Tutorials and Documents for Developers

* [Tutorial: A Walkthough of the BladeDISC Pass Pipeline](./docs/developers/pass_pipeline.md)
* [Introduction on Runtime Abstraction Layer](./docs/developers/runtime_abstraction_layer.md)
* [TorchBlade Overview](./docs/developers/bladedisc_torch_overview.md)
* [Tutorial: How to Add a New Torch Operator Converter](./docs/developers/torch_add_a_new_converter.md)

## How to Contribute

* [Contribute to BladeDISC](./docs/contribution.md)

## FAQ

### Roadmap with mlir-hlo Project

BladeDISC is in a close relationship with
[mlir-hlo](https://github.com/tensorflow/mlir-hlo) project. Part of the building
blocks, including the MHLO Op definitions, TF to MHLO conversions, and some
general purpose passes have been upstreamed to mlir-hlo repository. We'll
continue to work in a close cooperative relationship with mlir-hlo project in
the longer term.

## Contact Us

* Mailgroup: bladedisc-dev@list.alibaba-inc.com 

* DingTalk group for support and discussion:

![DingTalk](./docs/pics/dingtalk_support.png)
