# DISC optimization example for PyTorch ResNet model.

The ResNet model is based on the [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) paper.

This repository uses the
[model mantained in torchvision](https://pytorch.org/vision/stable/models/resnet.html).


## Configure and run.

These python packages are required for executing this example:

- torch >= 1.6.0
- torchvision
- torch_blade

To build and install `torch_blade` package, please refer to
["Build BladeDISC from source"](../../../../docs/build_from_source.md) and
["Install BladeDISC with Docker"](../../../../docs/install_with_docker.md).

You can run the example with the following command:

```bash
python main.py
```

It will run optimizations of BladeDISC.

## Performance results.

We evaluate this example on A10 GPU, The CUDA version is 11.3. CuDNN version is
8.3. PyTorch version is 1.12.0. The speedup reports on ResNet50 of the 200 inferences
is as following.

### `python main.py`
|    | Backend   |   batch | precision   |   Median(FPS) |   Mean(FPS) |   Median(ms) |   Mean(ms) |    99th_p |     std_dev |
|---:|:----------|--------:|:------------|--------------:|------------:|-------------:|-----------:|----------:|------------:|
|  0 | PyTorch   |      32 | fp32        |       930.02  |     930.859 |    0.0344079 |  0.0343776 | 0.0347254 | 0.000162988 |
|  1 |TorchScript|      32 | fp32        |       918.961 |     918.161 |    0.034822  |  0.0348532 | 0.0353194 | 0.000177631 |
|  2 | DISC      |      32 | fp32        |      1130.83  |    1131.88  |    0.0282977 |  0.0282732 | 0.0287687 | 0.000219865 |

### `python main.py --fp16`
|    | Backend   |   batch | precision   |   Median(FPS) |   Mean(FPS) |   Median(ms) |   Mean(ms) |    99th_p |     std_dev |
|---:|:----------|--------:|:------------|--------------:|------------:|-------------:|-----------:|----------:|------------:|
|  0 | PyTorch   |      32 | fp16        |       1870.41 |     1866.49 |    0.0171085 |  0.017145  | 0.0173835 | 9.1966e-05  |
|  1 |TorchScript|      32 | fp16        |       1803.68 |     1803.2  |    0.0177415 |  0.0177472 | 0.0180768 | 0.000129273 |
|  2 | DISC      |      32 | fp16        |       2311.53 |     2298.32 |    0.0138436 |  0.0139253 | 0.0145824 | 0.000170253 |

### `python main.py --amp`
|    | Backend   |   batch | precision   |   Median(FPS) |   Mean(FPS) |   Median(ms) |   Mean(ms) |    99th_p |     std_dev |
|---:|:----------|--------:|:------------|--------------:|------------:|-------------:|-----------:|----------:|------------:|
|  0 | PyTorch   |      32 | amp         |       1898.59 |     1895.44 |    0.0168546 |  0.016883  | 0.017094  | 8.18696e-05 |
|  1 |TorchScript|      32 | amp         |       1837.09 |     1833.55 |    0.0174189 |  0.0174531 | 0.0176906 | 0.000100789 |
|  2 | DISC      |      32 | amp         |       2279.53 |     2282.38 |    0.014038  |  0.0140209 | 0.0142996 | 7.63628e-05 |

DISC shows speedups over all configurations.
