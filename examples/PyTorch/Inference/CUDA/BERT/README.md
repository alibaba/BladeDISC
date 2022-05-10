# DISC optimization example for PyTorch BERT-large model.

This repository provides a script showing how to optimize a BERT large inference
model for PyTorch with DISC. The BERT-large model is constructed with Hugging
Face API, with fake parameter values. It prepares fake input data for execution.
It also shows a performance comparison between BladeDISC and TensorRT (TRT).


## Configure and run.

These python packages are required for executing this example:

- torch >= 1.6.0
- transformers
- torch_blade
- volksdep

To build and install `torch_blade` package, please refer to
["Build BladeDISC from source"](../../../../docs/build_from_source.md) and
["Install BladeDISC with Docker"](../../../../docs/install_with_docker.md).
The [`volksdep`](https://github.com/Media-Smart/volksdep) library is used to
execute the generated TRT engine.

You can run the example with the following command:

```bash
python main.py
```

It will run optimizations of BladeDISC and TRT.

## Performance results.

We evaluate this example on T4 GPU, for which the clock is 1590 during the test.
The CUDA version is 11.0. CuDNN version is 8.0. TRT version is 8.2. PyTorch
version is 1.7.1. The average execution time of the 100 inferences is as
following (ms).

| PyTorch | TRT static | Blade TRT static |  DISC  |
|---------|------------|------------------|--------|
|  40.77  |    4.56    |       4.30       |  4.71  |

DISC shows a 8.66x speedup over basic PyTorch, and achieves similar performance
with TRT static optimization. Note that `TRT static` is measured with `volkdeps`
upon the TRT engine generated with `trtexec`, for which we observe some
non-computation overhead.
