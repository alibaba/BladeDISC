# DISC optimization example for PyTorch Transformer-XL model.

The Transformer-XL model is proposed in [this paper](https://arxiv.org/abs/1901.02860).
This repository uses the
[model mantained in Hugging Face](https://huggingface.co/transformers/v3.0.2/model_doc/transformerxl.html),
It prepares sample input data.

## Configure and run.

These python packages are required for executing this example:

- torch >= 1.6.0
- transformers
- torch_blade

To build and install `torch_blade` package, please refer to
["Installation of TorchBlade"](../build_from_source.md) and
["Install BladeDISC With Docker"](../install_with_docker.md).

You can run the example with the following command:

```bash
python main.py
```

It will run optimizations of BladeDISC and TensorRT.

## Performance results.

We evaluate this example on T4 GPU, The CUDA version is 11.0. CuDNN version is
8.0. PyTorch version is 1.7.1. The average execution time of the 100 inferences
is as following (ms).

| PyTorch | Blade TRT static | DISC |
|---------|------------------|------|
|         |                  |      |

DISC shows a ?x speedup over basic PyTorch.
