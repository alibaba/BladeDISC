# DISC optimization example for PyTorch S2T model.

T5 is an encoder-decoder model for NLP tasks, which is proposed in
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf).
This repository uses the
[model mantained in Hugging Face](https://huggingface.co/docs/transformers/model_doc/t5),
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


## Performance results.

We evaluate this example on T4 GPU, The CUDA version is 11.0. CuDNN version is
8.0. PyTorch version is 1.7.1. The average execution time of the 100 inferences
is as following (ms).
result of TensorRT is not correct.)

| PyTorch | Blade TRT static | DISC |
|---------|------------------|------|
|         |                  |      |

DISC shows a ?x speedup over basic PyTorch.
