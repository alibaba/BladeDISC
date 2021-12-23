# Build BladeDISC from Source in a Docker container

This document introduced how to build BladeDISC from source,
to easy the software installation and configuration, we provides a
[Dockerfile](/docker/dev/Dockerfile) that contains requirements software
to build and test.

## Prerequisite

1. Git for checking out the source code.
1. [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
to launch Docker container on Nvidia GPU host.

## Checkout the Source

``` bash
git clone git@github.com:pai-disc/aicompiler.git
cd aicompiler && git submodule update --init --recursive
```

## Launch a Development Docker Container

Let us launch a Docker container that running a development Docker image:

``` bash
nvidia-docker run --rm -it -v $PWD:/disc yancey1989/bladedisc:latest-devel-cuda11.0 bash
```

you can also find more CUDA version development Docker images on
[the website](https://hub.docker.com/r/yancey1989/bladedisc/tags?page=1&name=devel)

## Building BladeDISC for TensorFlow Wrapper

First of all, let us run the configuration stage:

``` bash
python scripts/python/tao_build.py /opt/venv_disc -s configure --bridge-gcc default --compiler-gcc default
```

Building tao bridge library and this stage would generate
generate a dynamic link library on path: `tao/build/libtao_ops.so`

``` bash
python scripts/python/tao_build.py /opt/venv_disc -s build_tao_bridge
```

Then, build tao compiler main file, this would generate an executable
file on path: `tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main`

``` bash
python scripts/python/tao_build.py /opt/venv_disc -s build_tao_compiler
```

Finally, let us build a Python wheel package:

``` bash
cd tao && /opt/venv_disc/bin/python setup.py bdist_wheel
```

You can find the wheel file on `tao/disc/blade_disc_tf<version>.whl`, you can install
it with `pip` toolkit.

## Build Pytorch Wrapper

TBD
