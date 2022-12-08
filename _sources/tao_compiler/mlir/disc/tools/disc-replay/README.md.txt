# BladeDISC Replay Toolkit

In some scenarios, users want to profile or debug a single cluster using
some exiting profiler toolkit such as `nvprof` or [Nvidia Nsight System](https://developer.nvidia.com/zh-cn/nsight-systems),
this toolkit can help users to replay a single cluster on a host.

## How to Get the Toolkit

An easy way to get the replay toolkit is by pulling the BladeDISC runtime Docker
image. You can find the guide from [Install with Docker](/docs/install_with_docker.md).

You can also build it from source code, just building BladeDISC as
[Build From Source](/docs/bulid_from_source.md), then you can find an executable program
`tf_community/bazel-bin/tensorflow/compiler/mlir/disc/tools/disc-replay/disc-replay-main`
after finish the building phases.

## How to Use the Toolkit

1. enable the debug mode on a TensorFlow script with the environment variable
`DISC_DEBUG=true` as the following example

    ``` bash
    DISC_DEBUG=true python run.py 2>&1 > train.log
    ```

1. Find the input program and data file path from the logs, the
following is some log snippet

    ``` text
    ...
    2022-02-09 09:22:54.244100: I /work/tao/tao_bridge/kernels/tao_compilation_cache.cc:792] tar the cluster input tensors with protobuf format to: /tmp/tempfile-4856de862901-217fa700-29301-5d79260e310f5.tar
    ...
    2022-02-09 09:22:55.011848: I /work/tao/tao_bridge/kernels/tao_compilation_cache.cc:817] tao_compiler_input: /tmp/tempfile-4856de862901-217fa700-29301-5d79260e40fdc.input
   ```

   The `tempfile-4856de862901-217fa700-29301-5d79260e310f5.tar` is a tarball
   which compresses input tensors, and `tempfile-4856de862901-217fa700-29301-5d79260e40fdc.input`
   is the compiler input program, the protobuf definition is
   [tao_compiler_input.proto](/tao/tao_bridge/tao_compiler_input.proto).

   These two files are all you need to replay a cluster.

1. Replay the cluster using the `disc-replay-main` executable program, you can
also use `nvprof` or other profiler toolkits to help the profiling.

    ``` bash
    nvprof disc-replay-main -p /tmp/tempfile-4856de862901-217fa700-29301-5d79260e40fdc.input -d /tmp/tempfile-4856de862901-217fa700-29301-5d79260e310f5.tar
    ```
