# Performance Benchmarking

This is a comprehensive Python benchmark suite to accelerate detectron2 models using TorchBlade.

## Prerequisite

Benchmark scripts depends on following Python packages in addition to [requirements.txt](../requirements.txt) packages.

+ [Detectron2](https://github.com/facebookresearch/detectron2)
+ Torch
+ TorchBlade

## Usage

To run the benchmark for a given model:

```shell
python3 run_blade.py --model testRetinaNet
```

To list all models supported in the script:

```shell
$ python3 run_blade.py --help
TorchBlade Detectron2 Benchmark

optional arguments:
  -h, --help            show this help message and exit
  --model {testCascadeRCNN,testMaskRCNNC4,testMaskRCNNFPN,testMaskRCNNFPN_b2,testMaskRCNNFPN_pproc,testRetinaNet,testRetinaNet_scripted}
```

## About Detectron2 and TorchBlade

Detectron2 can be [deploy with TorchScript](https://detectron2.readthedocs.io/en/latest/tutorials/deployment.html#deployment-with-tracing-or-scripting),
by either tracing or scripting. The output model file can be loaded without detectron2 dependency in either Python or C++.

TorchBlade provides PyTorch/TorchScript module optimization. With TorchScript exported from Detectron2, a model can be accelerated
by TorchBlade with just a few lines of code:

```python3
# code snippet from run_blade.py
class TestDetectron2:
    ...

    def _benchmark(
        self,
        func_name,
        wrapper,
        traced_model,
        inputs,
        small_inputs,
        large_inputs,
        batch,
        enable_fp16,
    ):
        with torch.no_grad():
            config = torch_blade.config.Config()
            config.dynamic_tuning_shapes = {
                "min": small_inputs,
                "max": large_inputs,
                "opts": [inputs],
            }
            config.optimization_pipeline = torch_blade_trt.backend_name()
            config.enable_fp16 = enable_fp16
            config.enable_onnx_shape_white_list = False
            with config:
                blade_model = torch_blade.optimize(traced_model, False, inputs)

            with torch.cuda.amp.autocast(enable_fp16):
                benchmark(func_name, "Torch-Blade", enable_fp16, wrapper, inputs, batch)
```


## Detectron2 TorchBlade Inference Optimization

Our benchmark shows accelerating detectron2 models with TensorRT via TorchBlade would improve performance up to 2.7x.

| Model(traced)   | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |    99th_p |    std_dev |
|:------------|:-----------|:-------|-----------:|------------:|----------:|-----------:|----------:|-----------:|
| CascadeRCNN | Torch      | fp16   |    19.6151 |     19.0296 | 0.0509812 |  0.0534221 | 0.0976605 | 0.00884817 |
| CascadeRCNN | TorchBlade | fp16   |    36.5378 |     35.9904 | 0.0273689 |  0.0281801 | 0.0435887 | 0.00547062 |
| CascadeRCNN | Torch      | fp32   |    11.9749 |     11.7929 | 0.0835077 |  0.0853493 | 0.121567  | 0.00910164 |
| CascadeRCNN | TorchBlade | fp32   |    16.8001 |     16.4923 | 0.0595234 |  0.0611271 | 0.0946139 | 0.00652134 |

| Model(traced)  | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |   99th_p |    std_dev |
|:-----------|:-----------|:-------|-----------:|------------:|----------:|-----------:|---------:|-----------:|
| MaskRCNNC4 | Torch      | fp16   |    5.42709 |     5.38309 | 0.184261  |  0.18593   | 0.217443 | 0.00584642 |
| MaskRCNNC4 | TorchBlade | fp16   |   14.7845  |    14.4137  | 0.0676383 |  0.0701595 | 0.123343 | 0.00960248 |
| MaskRCNNC4 | Torch      | fp32   |    2.28852 |     2.28604 | 0.436964  |  0.437487  | 0.446839 | 0.00461236 |
| MaskRCNNC4 | TorchBlade | fp32   |    6.17004 |     6.09966 | 0.162073  |  0.164303  | 0.200569 | 0.00830803 |

| Model(traced)   | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |    99th_p |    std_dev |
|:------------|:-----------|:-------|-----------:|------------:|----------:|-----------:|----------:|-----------:|
| MaskRCNNFPN | Torch      | fp16   |    27.9353 |     27.1975 | 0.035797  |  0.0379382 | 0.100995  | 0.0111086  |
| MaskRCNNFPN | TorchBlade | fp16   |    55.3479 |     52.4638 | 0.0180675 |  0.0205101 | 0.0711264 | 0.00936073 |
| MaskRCNNFPN | Torch      | fp32   |    17.4301 |     16.9622 | 0.057372  |  0.0600043 | 0.114071  | 0.010588   |
| MaskRCNNFPN | TorchBlade | fp32   |    24.6164 |     24.0594 | 0.0406234 |  0.0422053 | 0.0903505 | 0.00732614 |

| Model(traced) | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |    99th_p |    std_dev |
|:---------------|:-----------|:-------|-----------:|------------:|----------:|-----------:|----------:|-----------:|
| MaskRCNNFPN_b2 | Torch      | fp16   |    37.1356 |     37.0281 | 0.0538567 |  0.0541153 | 0.0698256 | 0.00272032 |
| MaskRCNNFPN_b2 | TorchBlade | fp16   |    54.0766 |     52.9738 | 0.0369846 |  0.0383186 | 0.0736446 | 0.00629445 |
| MaskRCNNFPN_b2 | Torch      | fp32   |    18.381  |     18.2328 | 0.108808  |  0.109908  | 0.140845  | 0.00539663 |
| MaskRCNNFPN_b2 | TorchBlade | fp32   |    22.4722 |     22.1447 | 0.0889987 |  0.0907318 | 0.12361   | 0.00677864 |

| Model(traced) | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |    99th_p |   std_dev |
|:------------------|:-----------|:-------|-----------:|------------:|----------:|-----------:|----------:|----------:|
| MaskRCNNFPN_pproc | Torch      | fp16   |    25.0847 |     24.6139 | 0.0398649 |  0.0414263 | 0.0999447 | 0.0090731 |
| MaskRCNNFPN_pproc | TorchBlade | fp16   |    38.3059 |     37.7374 | 0.0261057 |  0.0268968 | 0.0618138 | 0.0049947 |
| MaskRCNNFPN_pproc | Torch      | fp32   |    16.3987 |     16.0337 | 0.0609803 |  0.0637757 | 0.130296  | 0.0134555 |
| MaskRCNNFPN_pproc | TorchBlade | fp32   |    22.5599 |     22.1358 | 0.0443265 |  0.0460027 | 0.0946874 | 0.0086997 |

| Model(traced) | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |    99th_p |    std_dev |
|:----------|:-----------|:-------|-----------:|------------:|----------:|-----------:|----------:|-----------:|
| RetinaNet | Torch      | fp16   |    25.5776 |     24.8496 | 0.0390967 |  0.0410799 | 0.0893885 | 0.0083293  |
| RetinaNet | TorchBlade | fp16   |    57.5659 |     55.9292 | 0.0173714 |  0.0187742 | 0.0614159 | 0.00762535 |
| RetinaNet | Torch      | fp32   |    20.2    |     19.7936 | 0.0495051 |  0.051284  | 0.0882284 | 0.00893048 |
| RetinaNet | TorchBlade | fp32   |    29.9236 |     29.5097 | 0.0334184 |  0.0341761 | 0.0490422 | 0.00389048 |

| Model(scripted) | Backend    | precision | Median(FPS) |   Mean(FPS) | Median(ms) |   Mean(ms) |    99th_p |    std_dev |
|:-------------------|:-----------|:-------|-----------:|------------:|----------:|-----------:|----------:|-----------:|
| RetinaNet | Torch      | fp16   |    28.2023 |     27.3567 | 0.0354581 |  0.0376919 | 0.0870901 | 0.00991941 |
| RetinaNet | TorchBlade | fp16   |    56.5545 |     55.3561 | 0.0176821 |  0.0184771 | 0.0417752 | 0.00427876 |
