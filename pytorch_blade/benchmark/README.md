# Benchmark TorchBlade's TensorRT Optimization

TorchBlade supports optimization via different backends. TensorRT is one of
TorchBlade's accelerators which is integrated via ONNX.

The benchmarking shows how to use in TorchBlade's TensorRT to speed up ML
models. It's consist of `[detectron2](detectron2/)` and
`[torch-tensorrt](torch-tensorrt/)`.

## Prerequisite

Benchmark scripts depend on following Python packages in addition to
requirements.txt packages:

- Torch-TensorRT
- Torch
- TensorRT
- Torch-Blade
- Detectron2

## Quickstart

It is recommended to fetch the latest runtime Docker image with PyTorch for a
smooth setup:

```shell
docker pull bladedisc/bladedisc:disc-runtime-pt-ngc
```

### Checkout the Source

```shell
git clone git@github.com:alibaba/BladeDISC.git
```

### Launch a Docker Container

```shell
docker run --gpus all --rm -it -v $PWD:/disc -w /disc/pytorch_blade bladedisc/bladedisc:disc-runtime-pt-ngc bash
```

### Run the benchmarks

```shell
 python3 -m pip install -r benchmark/requirements.txt
 bash benchmark/detectron2/test_d2_benchmark.sh 2>&1 | tee test_d2.log
 bash benchmark/torch-tensorrt/test_trt_benchmark.sh 2>&1 | tee test_trt.log

 grep "|" test_d2.log
 grep "|" test_trt.log
```
