# Build BladeDISC from Source

This document introduces how to build BladeDISC from source,
to make the installation and configuration easier, we provide a
[Dockerfile](/docker/dev/Dockerfile) that contains required software
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

## Building BladeDISC for TensorFlow Users

- step1: launch a development Docker container that runs a
development Docker image.

    ``` bash
    docker run --gpus all --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cuda11.0 bash
    ```

    please goto [this website](https://hub.docker.com/r/bladedisc/bladedisc/tags?page=1&name=devel) to
    find more images with various CUDA versions.

- step2: build and test tensorflow_bladedisc with an all-in-on bash script.

    ``` bash
    bash ./scripts/ci/build_and_test.sh
    ```

    the above command generates a wheel Python package on the path: `./build`,
    please free feel to install it with the [pip](https://pip.pypa.io/en/stable/installation/)
    installation toolkit.

## Building BladeDISC for PyTorch Users

- step1: launch a development Docker container that runs a
development Docker image.

    ``` bash
    docker run --gpus all --rm -it -v $PWD:/disc bladedisc/bladedisc:latest-devel-cuda11.0 bash
    ```

- step2: build and test pytorch_blade with an all-in-one script:

    ``` bash
    cd pytorch_blade && bash ./ci_build/build_pytorch_blade.sh
    ```

- step3: build the pytorch_blade Python wheel package.

    ``` bash
    python setup.py bdist_wheel 
    ```

    the above command generates a wheel Python package on the path: `pytorch_blade/dist/`,
    please install it with the pip installation toolkit.
