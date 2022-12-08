# DISC optimization example for PyTorch ALBERT model

The ALBERT model was proposed in [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942). It presents two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. This repository uses the [model mantained in Hugging Face](https://huggingface.co/docs/transformers/model_doc/albert). It prepares fake input data for execution and run one step of the forward.

## Configure and run.

These python packages are required for executing this example:

- torch >= 1.6.0
- transformers
- tabulate
- pandas
- onnxruntime
- torch_blade

To build and install `torch_blade` package, please refer to
["Build BladeDISC from source"](../../../../docs/build_from_source.md) and
["Install BladeDISC with Docker"](../../../../docs/install_with_docker.md).

You can run the example with the following command:

```bash
# set cpu affinity to get more stable results. we only test the single thread
# scenario, thus we also set `OMP_NUM_THREADS=1`.
OMP_NUM_THREADS=1 taskset -c 0 python main.py
```


## Performance results.

### X86

We evaluate this example on Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz, using
1 thread. The PyTorch version is 1.10.0 and the onnxruntime version is 1.10.0.
The average execution time of the 200 inferences is as following (ms).

| PyTorch | ONNX Runtime | DISC  |
|---------|--------------|-------|
|  56.7   |    43.1      | 32.8  |

DISC shows a 1.73x speedup over basic PyTorch and 1.31x speedup over ONNX
Runtime.

### AArch64

We evaluate this example on Neoverse-N1 CPU @ 2.90GHz, using 1 thread. The
PyTorch version is 1.10.0 and the onnxruntime version is 1.11.1. The average
execution time of the 200 inferences is as following (ms).

| PyTorch | ONNX Runtime | DISC  |
|---------|--------------|-------|
| 192.3   |   135.3      | 112.7 |

DISC shows a 1.71x speedup over basic PyTorch and 1.20x speedup over ONNX
Runtime.
