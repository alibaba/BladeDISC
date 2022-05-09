# DISC optimization example for PyTorch S2T model.

The Speech to Text Transformer (S2T) model is used for automatic speech
recognition (ASR). The S2T model was proposed in the paper
["fairseq S2T"](https://arxiv.org/abs/2010.05171). This repository uses the
[model mantained in Hugging Face](https://huggingface.co/facebook/s2t-small-librispeech-asr),
It prepares fake input data for execution and run one step of the forward.


## Configure and run.

These python packages are required for executing this example:

- torch >= 1.6.0
- transformers
- torch_blade

To build and install `torch_blade` package, please refer to
["Build BladeDISC from source"](../../../../docs/build_from_source.md) and
["Install BladeDISC with Docker"](../../../../docs/install_with_docker.md).

You can run the example with the following command:

```bash
python main.py
```


## Performance results.

We evaluate this example on T4 GPU, The CUDA version is 11.0. CuDNN version is
8.0. PyTorch version is 1.7.1. The average execution time of the 100 inferences
is as following (ms). (We do not show the performance of TensorRT because the
result of TensorRT is not correct.)

| PyTorch | DISC |
|---------|------|
|  31.25  | 6.72 |

DISC shows a 4.65x speedup over basic PyTorch.
