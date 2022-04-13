# DISC optimization example for PyTorch BERT-large model.

This repository provides a script showing how to optimize a BERT large inference
model for PyTorch with DISC. The BERT-large model is constructed with Hugging
Face API, with fake parameter values. It prepares fake input data for execution.


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

By configuring `optimize_config` as `'DISC'`, or `None` in the script,
you can run the model with DISC or without any optimization.


## Performance results.

We evaluate this example on T4 GPU, The CUDA version is 11.0. CuDNN version is 8.2.
PyTorch version is 1.7.1. The average execution time of the 100 inferences is as following.

| PyTorch |    DISC    |
|-------------|-----------|
|      |   |

DISC shows a TBDx speedup over basic PyTorch.
