# Contribute Code to BladeDISC

You're welcome to contribute to BladeDISC!
BladeDISC is an end-to-end
You have to agree with CLA before contributing to BladeDISC. This document
introduces how to prepare your local development environment and our workflow.

## Local Development Environment

Some software are required to contribute to BladeDISC:

- Docker
- Nvidia Container Toolkit
- Git

It's recommend to use the official development Docker image to
build and test your code:

``` bash
docker run --rm -it --gpus all -v $PWD:/workspace bladedisc/bladedisc:latest-devel-cuda11.0 bash
```

you can find more images on [install with docker](./install_with_docker.md#download-a-bladedisc-docker-image)

## Submit a Pull Request to BladeDISC

BladeDISC uses [git branching model](https://nvie.com/posts/a-successful-git-branching-model/),
the following steps guide usual contribution.

1. Fork
  
1. Clone

1. Local Feature Branch

1. Git Commit

1. Keep update

1. Create a Pull Request

## 