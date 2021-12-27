# Build BladeDISC from Source in a Docker container

This document introduced how to build BladeDISC from source,
to easy the software installation and configuration, we provide a
[Dockerfile](/docker/dev/Dockerfile) that contains requirements software
to build and test.

## Prerequisite

1. Git for checking out the source code.
1. [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
to launch Docker container on Nvidia GPU host.

## Checkout the Source

``` bash
git clone git@github.com:alibaba/BladeDISC.git
cd BladeDISC && git submodule update --init --recursive
```

## Building BladeDISC for TensorFlow Wrapper

1. step0: launch a development Docker container that runs a
development Docker image.

    ``` bash
    nvidia-docker run --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cuda11.0 bash
    ```

    please goto [this website](https://hub.docker.com/r/bladedisc/bladedisc/tags?page=1&name=devel) to
    find more images with various CUDA versions.

1. step1: configuration stage.

    ``` bash
    python scripts/python/tao_build.py /opt/venv_disc -s configure --bridge-gcc default --compiler-gcc default
    ```

    - `/opt/venv_disc` is a pre-installed Python virtual environment with requirement Python packages.
    - `-s configure` specified the building stage.
    - `--bridge-gcc` and `--compiler-gcc` specified the GCC version building with tao bridge and disc compiler core.

1. step2: build tao bridge library

    ``` bash
    python scripts/python/tao_build.py /opt/venv_disc -s build_tao_bridge
    ```

    the above command generates a dynamic library on path `tao/build/libtao_ops.so` if works well.

1. step3: build disc compiler core

    ``` bash
    python scripts/python/tao_build.py /opt/venv_disc -s build_tao_compiler
    ```

    the above command generates an executable file on path
    `tf_community/bazel-bin/tensorflow/compiler/decoupling/tao_compiler_main` if works well.

1. [option]step4: run unit tests

    please feel free to skip this step if you just want to build a Python wheel package.

    ```bash
    python scripts/python/tao_build.py /op/env_disc -s test_tao_compiler
    ```

1. step5: build Python wheel package

    ``` bash
    cd tao && /opt/venv_disc/bin/python setup.py bdist_wheel
    ```

    the above command generates a wheel Python package on the path: `tao/disc/`,
    please free feel to install it with [pip](https://pip.pypa.io/en/stable/installation/)
    installation toolkit.

## Building BladeDISC for PyTorch Wrapper

1. step0: launch a development Docker container that runs a
development Docker image.

    ``` bash
    nvidia-docker run --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cuda11.0 bash
    ```

1. step1: build and test PytorchBlade with an all-in-one script:

    ``` bash
    cd pytorch_blade && bash ./ci_build/build_pytorch_blade.sh
    ```

1. step2: build the PyTorchBlade Python wheel package.

    ``` bash
    python setup.py bdist_wheel 
    ```

    the above command generates a wheel Python package on the path: `pytorch_blade/dist/`,
    please install it with the pip installation toolkit.
