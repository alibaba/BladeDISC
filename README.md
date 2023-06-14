# BladeDISC Introduction <!-- omit in toc -->

- [What's New](#whats-new)
- [Overview](#overview)
  - [Features and Roadmap](#features-and-roadmap)
    - [Frontend Framework Support Matrix](#frontend-framework-support-matrix)
    - [Backend Support Matrix](#backend-support-matrix)
    - [Deployment Solutions](#deployment-solutions)
  - [Numbers of Typical Workloads](#numbers-of-typical-workloads)
    - [Advantage in Dynamic Shape Workloads](#advantage-in-dynamic-shape-workloads)
- [API QuickView](#api-quickview)
  - [For TensorFlow Users](#for-tensorflow-users)
  - [For PyTorch Users](#for-pytorch-users)
- [Setup and Examples](#setup-and-examples)
- [Publications](#publications)
- [Tutorials and Documents for Developers](#tutorials-and-documents-for-developers)
- [Presentations and Talks](#presentations-and-talks)
- [How to Contribute](#how-to-contribute)
- [Building Status](#building-status)
- [FAQ](#faq)
  - [Roadmap with mlir-hlo Project](#roadmap-with-mlir-hlo-project)
  - [Roadmap with Torch-MLIR Project](#roadmap-with-torch-mlir-project)
- [Contact Us](#contact-us)

## What's New

+ [🔥 2023.03.17] BladeDISC v0.4.0: [Messive performance and feature updates](https://github.com/alibaba/BladeDISC/releases/tag/v0.4.0)
+ [2022.12.08] BladeDISC v0.3.0:
 [Announce PyTorch 2.0 Compilation Support](https://github.com/alibaba/BladeDISC/releases/tag/v0.3.0)

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

[2] PyTorch version >= 1.6.0 has been fully verified.

[3] Although supported, there's much room for improvement on Op coverage for
training workloads.

#### Backend Support Matrix

|            |   Status      |
|----------- | ------------- |
| Nvidia GPU |    Yes [1]    |
| AMD GPU    |    Yes        |
| Hygon DCU  |    Yes        |
|  X86       |    Yes        |
| AArch64    |    Yes        |

[1] Support for CUDA below 11.0 have been deprecated officially since Aug, 2022.

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
production purpose, BladeDISC shows up to 8.66x speedup compared with
TensorFlow/PyTorch. Moreover, compared to static optimizing compilers (i.e.,
XLA and TensorRT), DISC shows comparable or even better performance.

<figure align="center">
<img src="./docs/pics/numbers.png" style="width:60%">
<figcaption align = "center">
<b>
Fig.1 Performance speedup over framework.
<i>Framework</i> means either TensorFlow or PyTorch.
<i>FastSpeech2</i> is TensorFlow model and others are PyTorch models.
The <i>static compiler</i> for TensorFlow is XLA and that for PyTorch is TensorRT.
Note that <i>S2T</i> and <i>T5</i> have no TensorRT performance due to wrong result.
</b>
</figcaption>
</figure>

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
import blade_disc_tf as disc
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

module = MyModule().eval()

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
* [Tutorial: How to Add a New Torch Operator](./docs/developers/torch_add_a_new_operator.md)

## Presentations and Talks
* [Performance optimization practice for dynamic shape AI workloads via a compiler-based approach](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/performance-optimization-practice.pdf)
* [2022/07/31 BladeDISC: A Practice of Dynamic Shape Deep Learning Compiler(Chinese)](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/BladeDISC%EF%BC%9A%E5%8A%A8%E6%80%81Shape%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E5%AE%9E%E8%B7%B5%E7%9A%84.pdf)
* [2022/07/07 BladeDISC and Torch-MLIR Roadmap Talk on Torch-MLIR Community](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/BladeDISC-and-TorchMLIR-Roadmap-tts.pptx)
* [GTC22-S41073, Generalized and Transparent AI Optimization Solutions with AI Compilers from Cloud Service](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/GTC22%20S41073%2C%20Generalized%20and%20Transparent%20AI%20Optimization%20Solutions%20with%20AI%20Compilers%20from%20Cloud%20Service.pdf)
* [GTC22-S41395, Easier-to-use and More Robust TensorRT via PAI-Blade](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/GTC22-S41395%2C%20Easier-to-use%20and%20More%20Robust%20TensorRT%20via%20PAI-Blade.pdf)
* [2023/2/17 bladedisc intro. (cpu vendor oriented)](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/bladedisc-intro-for-intel.pdf)
* [2023/3/10 transform dialect based codegen in bladedisc](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/transform-dialect-based-codegen-in-bladedisc.pdf)

## How to Contribute

* [Contribute to BladeDISC](./docs/contribution.md)

## Building Status

| Framework | Device| Status |
| -- | -- | -- |
| PyTorch1.6.0 | CPU | [![pytorch160_cpu](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch160_cpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch160_cpu.yml) |
| PyTorch1.7.1 | GPU |  [![pytorch171_gpu](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch171_gpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch171_gpu.yml) |
| PyTorch1.8.1 | CPU | [![pytorch181_cpu](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch181_cpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch181_cpu.yml) |
| PyTorch1.9.0 | GPU | [![pytorch1.9.0_gpu](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch190_gpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch190_gpu.yml) |
| PyTorch1.12.0 | GPU | [![pytorch112_gpu](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch112_gpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch112_gpu.yml) |
| PyTorch1.10.0 | AArch64 |  [![pytorch110_aarch64](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch110_aarch64.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/pytorch110_aarch64.yml) |
| TensorFlow1.15 | CPU| [![tf115_cpu](https://github.com/alibaba/BladeDISC/actions/workflows/tf115_cpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/tf115_cpu.yml) |
| TensorFlow2.4 | GPU | [![tf24_gpu](https://github.com/alibaba/BladeDISC/actions/workflows/tf24_gpu.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/tf24_gpu.yml) |
| TensorFlow2.8 | AArch64 | [![tf280_aarch64](https://github.com/alibaba/BladeDISC/actions/workflows/tf280_aarch64.yml/badge.svg?branch=main)](https://github.com/alibaba/BladeDISC/actions/workflows/tf280_aarch64.yml) |

## FAQ

### Roadmap with mlir-hlo Project

BladeDISC is in a close relationship with
[mlir-hlo](https://github.com/tensorflow/mlir-hlo) project. Part of the building
blocks, including the MHLO Op definitions, TF to MHLO conversions, and some
general purpose passes have been upstreamed to mlir-hlo repository. We'll
continue to work in a close cooperative relationship with mlir-hlo project in
the longer term.

### Roadmap with Torch-MLIR Project

BladeDISC compiles PyTorch workloads based on [Torch-MLIR](https://github.com/llvm/torch-mlir/).
The BladeDISC Dev Team is cooperating with the community to add Torch-To-Mhlo conversion
to Torch-MLIR, especially fully dynamic shape features.
See RFC: https://github.com/llvm/torch-mlir/issues/999.
We appeal to the community developers interested in joining.

## Contact Us

* Mailgroup: bladedisc-dev@list.alibaba-inc.com

* DingTalk group for support and discussion:

![DingTalk](./docs/pics/dingtalk_support.png)
