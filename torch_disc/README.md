# Torch-DISC

## How to build

1. Development Docker image

    ```bash
    nvidia-docker run --rm -it -v $PWD:/work pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel bash
    ```

1. install requirements

    ``` bash
    conda install -y cmake git ninja
    apt-get update -y && apt-get install -y clang-8 clang++-8
    ```

1. build pytorch and torch-ltc

    ``` bash
    (cd torch_disc/pytorch && python setup.py install)
    (cd torch_disc && setup.py develop)
    ```

1. Try to load `torch_disc` module

    ``` bash
    cd torch_disc/bazel-bin/torch_disc
    python -c "import _torch_disc"
    ```
