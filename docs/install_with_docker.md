# Install BladeDISC With Docker

[Docker](https://www.docker.com/) is a light container system, it helps
users to package software and isolate BladeDISC runtime environment from
the rest of the system. BladeDISC CI system released BladeDISC with different
tag on [Docker Hub](https://hub.docker.com/repository/docker/bladedisc/bladedisc/tags?page=1&ordering=last_updated)
repository.

[Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart)
is an easy way to use GPU on Linux, please make sure you have installed it
on your host.

## Download a BladeDISC Docker Image

BladeDISC released TensorFlow and PyTorch frontend packages in separate Docker images
on [bladedisc/bladedisc](https://hub.docker.com/repository/docker/bladedisc/bladedisc/tags?page=1&ordering=last_updated).
The released Image tag is as the following table:

| Docker tag | Description |
| -- | -- |
| latest-runtime-torch1.7.1-cu110 | latest release of BladeDISC, includes PyTorch 1.7.1 and CUDA 11.0 |
| latest-runtime-torch1.12.0-cu113 | latest release of BladeDISC, includes PyTorch 1.12.0 and CUDA 11.3 |
| latest-runtime-tensorflow1.15 | latest release of BladeDISC, includes TensorFlow 1.15 and CUDA 11.0 |
| latest-runtime-tensorflow2.4 | latest release of BladeDISC, includes TensorFlow 2.4 and CUDA 11.0 |
| latest-devel-cu110 | latest build of development environment, includes CUDA 11.0 and required development toolkit |
| latest-devel-cu113 | latest build of development environment, includes CUDA 11.3 and required development toolkit |
| latest-devel-cpu-aarch64 | latest build of development environment for AArch64 platform |
| latest-runtime-tensorflow2.8-cpu-aarch64 | latest release of BladeDISC for TensorFlow 2.8 on AArch64 platform |
| latest-runtime-torch1.10.0-cpu-aarch64 | latest release of BladeDISC for PyTorch 1.10.0 on AArch64 platform |

**Note**: Users located in China can use `registry.cn-shanghai.aliyuncs.com/bladedisc/bladedisc` to get
higher download speed.

## Start a Docker Container

To launch a BladeDISC Docker container with GPU support, you can use the
following command:

``` bash
docker run --rm -it --gpus all -v [host-src/container-desc] bladedisc/bladedisc:[tag] [command]
```

- `--rm` automatically remove it after the container stops.
- `-it` runs the container with interactive mode.
- `-v [host-src/container-dest]` mount a volume from host to container.

An example to execute the `entry.py` PyTorch script with BladeDISC Docker:

``` bash
nvidia-docker run --rm -it -v $PWD:/work bladedisc/bladedisc:latest-runtime-torch1.7.1-cu110  python /work/entry.py
```
