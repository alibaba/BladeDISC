# Build BladeDISC from Source

This document introduces how to build BladeDISC from source,
to make the installation and configuration easier, we provide a
[Dockerfile](/docker/dev/Dockerfile) that contains required software
to build and test.

## Prerequisite

1. Git for checking out the source code.
1. (Optional)[Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
to launch Docker container on Nvidia GPU host if you want to build BladeDISC for GPU backend.

## Checkout the Source

``` bash
git clone git@github.com:alibaba/BladeDISC.git
cd BladeDISC && git submodule update --init --recursive
```

## Launch a development Docker container
``` bash
# For GPU backend
docker run --gpus all --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cuda11.0 bash

# For x86 CPU backend
docker run --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cuda11.0 bash

# For AArch64 CPU backend
docker run --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cpu-aarch64 bash
```

please goto [this website](https://hub.docker.com/r/bladedisc/bladedisc/tags?page=1&name=devel) to find more images with various CUDA versions.

**Note that we use the same development docker images for both Nvidia GPU backend and X86 CPU backend. For X86 CPU backend, both the building phase and execution phase do not require CUDA available.**

## Building BladeDISC for TensorFlow Users

build and test tensorflow_bladedisc with an all-in-on bash script.

``` bash
# For GPU backend
bash ./scripts/ci/build_and_test.sh

# For x86 or AArch64 CPU backend
bash ./scripts/ci/build_and_test.sh --cpu-only
```

the above command generates a wheel Python package on the path: `./build`,
please free feel to install it with the [pip](https://pip.pypa.io/en/stable/installation/)
installation toolkit.

## Building BladeDISC for PyTorch Users

- step1: build and test pytorch_blade with an all-in-one script:

    ``` bash
    # For GPU backend
    cd pytorch_blade && bash ./scripts/build_pytorch_blade.sh

    # For x86 CPU backend
    export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=OFF
    export TORCH_BLADE_CI_BUILD_TORCH_VERSION=1.8.1+cpu
    cd pytorch_blade && bash ./scripts/build_pytorch_blade.sh

    # For AArch64 CPU backend
    export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=OFF
    export TORCH_BLADE_CI_BUILD_TORCH_VERSION=1.10.0+aarch64
    cd pytorch_blade && bash ./scripts/build_pytorch_blade.sh
    ```

- step2: build the pytorch_blade Python wheel package.

    ``` bash
    # For GPU backend
    python setup.py bdist_wheel

    # For x86 CPU backend
    export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=OFF
    export TORCH_BLADE_CI_BUILD_TORCH_VERSION=1.8.1+cpu
    python setup.py bdist_wheel

    # For AArch64 CPU backend
    export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=OFF
    export TORCH_BLADE_CI_BUILD_TORCH_VERSION=1.10.0+aarch64
    python setup.py bdist_wheel
    ```

    the above command generates a wheel Python package on the path: `pytorch_blade/dist/`,
    please install it with the pip installation toolkit.
