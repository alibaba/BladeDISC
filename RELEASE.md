# Release 0.2.0

# Performance Optimization

## GPU stitch fusion

Make use of GPU shared memory to fuse reduce operator with its consumers into one kernel.
It helps to accommodate complex memory-intensive computations (e.g., LayerNorm, SoftMax) into one kernel,
reducing off-chip memory traffics and overhead of kernel scheduling and launching.
It implements partial functions described in paper [AStitch](https://dl.acm.org/doi/abs/10.1145/3503222.3507723).
It is currently under refactoring to enhance the robustness, for which it is not enabled by default.
Users of BladeDISC can enable it by setting the environment variable `DISC_ENABLE_STITCH=true`.

Note that we have already released the CPU stitch optimization when we open-source the BladeDISC project, which is enabled by default.
Refer to the [materials](https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/performance-optimization-practice.pdf) for more information about CPU stitch technique details.

## GEMM merging

Support two types of GEMM merging optimization.
One is to merge two GEMMs sharing the same operand into a single GEMM.
The other one is to merge two GEMMs with the same shape into a batched GEMM.
The GEMM merging optimization helps to increase hardware utilization and to reduce kernel launch overhead.

## CPU GEMM/Convolution weight pre-packing optimization

Support weight pre-packing optimization for convolution (calling onednn library) and GEMM (calling mkl/onednn/acl libraries) operations.

## Convolution layout optimization and transpose elimination

Support to transform the layout of convolution operator to the friendliest format on the specific device (i.e., either CPU or GPU).
Most of the introduced transpose operators can be eliminated in a following transpose-simplifier pass.

## Other optimizations
* Optimize the schedule selection strategy for reduce operator on GPU to enhance thread-level-parallelism.
* Algebraic simplification for operators like power.
* Support to fuse splat constant operator with its consumers, reducing memory access overhead.
Refer to [issue](https://github.com/alibaba/BladeDISC/issues/113).

# Function Enhancement

## CPU end-to-end optimization

Support end-to-end optimization for X86 and AArch64 CPUs.

## TorchBlade/TensorFlowBlade clustering and optimizing with TensorRT

According to the supported operators of TensorRT, cluster sub-graphs and apply TensorRT optimization for both TensorFlow and PyTorch models.

## Accelerating PyTorch Training

Release PoC version for accelerating PyTorch training via Disc + Lazy Tensor Core,
referring to the related [issue](https://github.com/alibaba/BladeDISC/issues/156) and [design doc](https://github.com/alibaba/BladeDISC/blob/main/docs/design/ltc_disc.md).

## Shape analysis and simplifier enhancement

Enhance the shape equality analysis according to the dimension values.
Add the function to analyze the collapse and expand relationship between dimensions,
which helps to identify the dimension mapping between input and output values of reshape operator.
This is the basic function to support GPU stitch fusion.

## Codegen support for int8 datatype

Support int8 datatype for the code generation of memory-intensive operators (e.g., element-wise, reduce operators).

# Toolchain Support and Process Optimization

## Replay tool
Support to dump clusters and the corresponding input data, based on which developers can replay the execution.
It is effective to help debugging and tuning.
Refer to [issue](https://github.com/alibaba/BladeDISC/issues/76).

## CI optimization
Enhance the CI process of BladeDISC repo, which helps the people from community to contribute to BladeDISC more conveniently and efficiently.

## TorchBlade bazel build
Migrate TorchBlade's compilation toolchain from the original CMake to bazel, enhancing maintainability.

# Other

## Example preparation

Prepare a set of commonly used models as the examples for BladeDISC.
Compare the performance of BladeDISC with TensorRT, XLA and ONNX Runtime (ORT) upon the examples.

## Community TF rebase

Rebase to TensorFlow codebase for BladeDISC according to the newest community code.

## Code maintenance

Continuous bug fixing and code refactoring.

