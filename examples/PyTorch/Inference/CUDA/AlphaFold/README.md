# DISC optimization example for PyTorch AlphaFold model.

This repository provides a tool(FoldAcc) and scripts showing how to optimize a AlphaFold model for PyTorch with DISC/AMP(auto mix precision)/Tensor Parallel.
The AlphaFold model is constructed with [Uni-Fold](https://github.com/dptech-corp/Uni-Fold).

## Prerequisite

### Installing BladeDISC

To build and install `torch_blade` package, please refer to
["Build BladeDISC from source"](../../../../docs/build_from_source.md) and
["Install BladeDISC with Docker"](../../../../docs/install_with_docker.md).

We recommend installing BladeDISC with Docker `bladedisc/bladedisc:latest-runtime-torch1.12.0-cu113`.

### Installing Uni-Fold

Please refer to [Uni-Fold](https://github.com/dptech-corp/Uni-Fold) to install UniFold.

### Installing FoldAcc

You can run the following command to install foldacc tool:

```bash
cd FoldAcc && python3 setup.py install
```

## Inference

### Accelerate Uni-Fold inference

You can run the example with the following command, the detail arguments are same as Uni-Fold:

```bash
torchrun --nproc_per_node=2 unifold_inference.py ...
```

## Quickstart

We provide a FoldAcc tools to accelerate AlphaFold model.

To make users easier to use, FoldAcc provides simple Python API is as follows:

```bash
from foldacc.fold.unifold import optimize_unifold

# create AlphaFold model by Uni-Fold
from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold

config = model_config(...)
model = AlphaFold(...)

# use optimize_unifold to accelerate model
with torch.no_grad():
    blade_model = optimize_unifold(
        model,                      # AlphaFold model.
        config,                     # AlphaFold config.
        enable_disc=True,           # whether use bladedisc (maybe spend two hours if open).
        enable_low_mem=False,       # whether use low memory optimize (for long sequence).
        enable_amp=True,            # whether use auto mix precision (only use for float16).
        enable_trace=True,          # whether convert model to torchscript.
        dtype=torch.half,           # model precision (support torch.float/torch.half/torch.bfloat16).
        device=0,                   # target device.
        save_dir=None,              # save directory of optimized model.
        load_dir=None               # load directory of optimized model.
    )

# do inference
...

```

## Acknowledgement

FoldAcc refers to the following projects:

- [AlphaFold](https://github.com/deepmind/alphafold)
- [Uni-Fold](https://github.com/dptech-corp/Uni-Fold)
- [OpenFold](https://github.com/aqlaboratory/openfold)
- [FastFold](https://github.com/hpcaitech/FastFold)