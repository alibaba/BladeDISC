
# Performance Benchmarking

**NOTE that this package is modified from:**\
[Torch-TensorRT/exmaples/benchmark/py/README.md](https://github.com/NVIDIA/Torch-TensorRT/blob/95d2b6e003a0392fe08ce49520f015c230d4c750/examples/benchmark/py/README.md)

This is a comprehensive Python benchmark suite to run perf runs using different supported backends. Following backends are supported:

1. Torch
2. Torch-TensorRT
3. TensorRT
4. Torch-Blade

Note: Please note that for ONNX models, user can convert the ONNX model to TensorRT serialized engine and then use this package.

## Prerequisite

Benchmark scripts depends on following Python packages in addition to [requirements.txt](../requirements.txt) packages

1. Torch-TensorRT
2. Torch
3. TensorRT
4. Torch-Blade

## Structure

```text
./
├── config
│   ├── crnn.yml
│   ├── vgg16.yml
│   └── yolov5.yml
├── models
├── perf_run.py
├── test_trt_benchmark.sh
└── README.md
```

Please save your configuration files at config directory. Similarly, place your model files at models path.

## Usage

To run the benchmark for a given configuration file:

```shell
python perf_run.py --config=config/vgg16.yml
```

## Configuration

There are two sample configuration files added.

* crnn.yml, vgg16.yml and yolov5.yml demonstrates a configuration with all the supported backends (Torch, Torch-TensorRT, TensorRT, Torch-Blade)

### Supported fields

| Name | Supported Values | Description |
| --- | --- | --- |
| backend | all, torch, torch_tensorrt, tensorrt | Supported backends for inference. |
| input | - | Input binding names. Expected to list shapes of each input bindings |
| model | - | Configure the model filename and name |
| filename | - | Model file name to load from disk. |
| name | - | Model name |
| runtime | - | Runtime configurations |
| device | 0 | Target device ID to run inference. Range depends on available GPUs |
| precision | fp32, fp16 or half, int8 | Target precision to run inference. int8 cannot be used with 'all' backend |
| calibration_cache | - | Calibration cache file expected for torch_tensorrt runtime in int8 precision |

Note:

1. Please note that torch runtime perf is not supported for int8 yet.
2. Torchscript module filename should end with .jit.pt otherwise it will be treated as a TensorRT engine.

Additional sample use case:

```text
backend: 
  - torch
  - torch_tensorrt
  - tensorrt
  - torch_blade
input: 
  input0: 
    - 3
    - 224
    - 224
  num_inputs: 1
model: 
  filename: model.plan
  name: vgg16
runtime: 
  device: 0
  precision: 
    - fp32
    - fp16
```


## Compairison (@Nvidia T4)

| Model | Backend        | precision   |   Median(FPS) |   Mean(FPS) |   Median-Latency(ms) |   Mean-Latency(ms) |     99th_p |     std_dev |
|------:|:---------------|:------------|--------------:|------------:|---------------------:|-------------------:|-----------:|------------:|
| vgg16 | Torch          | fp32        |       117.516 |     116.689 |           0.00850948 |         0.00857261 | 0.00896431 | 0.000158592 |
| vgg16 | Torch-Blade    | fp32        |       137.365 |     137.068 |           0.00727989 |         0.00729606 | 0.00744398 | 5.56986e-05 |
| vgg16 | Torch-TensorRT | fp32        |       140.318 |     140.042 |           0.00712666 |         0.00716695 | 0.0073506  | 0.000575365 |
| vgg16 | TensorRT       | fp32        |       139.23  |     138.601 |           0.00718237 |         0.00721714 | 0.00762804 | 0.000127496 |
| vgg16 | Torch          | fp16        |       246.146 |     246.007 |           0.00406262 |         0.00406507 | 0.00413304 | 2.49436e-05 |
| vgg16 | Torch-Blade    | fp16        |       400.434 |     401.282 |           0.00249729 |         0.0024925  | 0.00253454 | 3.45862e-05 |
| vgg16 | Torch-TensorRT | fp16        |       381.676 |     364.289 |           0.00262002 |         0.00280076 | 0.00422715 | 0.000472002 |
| vgg16 | TensorRT       | fp16        |       384.943 |     390.188 |           0.00259779 |         0.00256511 | 0.00267093 | 7.56818e-05 |

The benchmark on vgg16 shows that TorchBlade has similar performance to Torch-TensorRT and TensorRT.

| Model   | Backend     | precision   |   Median(FPS) |   Mean(FPS) |   Median-Latency(ms) |   Mean-Latency(ms) |     99th_p |     std_dev |
|--------:|:------------|:------------|--------------:|------------:|---------------------:|-------------------:|-----------:|------------:|
| yolov5s | Torch       | fp32        |       141.408 |     140.719 |           0.00707173 |         0.00710688 | 0.00725652 | 6.02203e-05 |
| yolov5s | Torch-Blade | fp32        |       171.632 |     171.931 |           0.00582641 |         0.00581733 | 0.0059523  | 7.83001e-05 |
| yolov5s | Torch       | fp16        |       119.978 |     120.182 |           0.00833483 |         0.00832087 | 0.00837208 | 3.86582e-05 |
| yolov5s | Torch-Blade | fp16        |       414.022 |     413.138 |           0.00241533 |         0.00242073 | 0.0024913  | 2.37797e-05 |
| crnn    | Torch       | fp32        |       126.432 |     126.395 |           0.00790937 |         0.00791219 | 0.00804682 | 6.26799e-05 |
| crnn    | Torch-Blade | fp32        |       175.106 |     175.274 |           0.00571081 |         0.00570566 | 0.00579444 | 4.15103e-05 |
| crnn    | Torch-Blade | fp16        |       258.254 |     258.346 |           0.00387215 |         0.00387082 | 0.00389814 | 1.38553e-05 |

The benchmark on yolov5s and crnn shows that TorchBlade is robust, while Torch-TensorRT/TensorRT failed to optimize the model.
