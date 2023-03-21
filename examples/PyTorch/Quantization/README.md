# Performing DISC Quantization Optimization


This folder provides an end-to-end examples about how to perform DISC **quantization** optimization on PyTorch's models.

## How does this sample work
In order to deploy a quantized model, two things need to be done:
1. Frontend: Modify the inference graph and obtain the quantization information of each activation/weight in the
specific position.
2. Backend: Compile the model processed in step 1 to get the quantized model.

#### Frontend
A light-weight quantization tool named [torch-quant](https://github.com/alibaba/BladeDISC/tree/main/tools/torch_quant) 
is used as the frontend. 
A typical workflow of torch-quant is as following:
- Step 0: Get the pre-trained model. (A `nn.Module` provided by the users)
- Step 1: Pass the pre-trained model to torch-quant's quantizer. It will convert the `nn.Module`
to `fx.Module` and automatically modify the graph according to the requirements of the backend.
- Step 2: Run inference on the `fx.Module` using a set of calibration data and the quantization
information will be collected automatically during the inference. This process is often called as Post-training
quantization (PTQ).
- Step 3: [Optional] After PTQ, if you feel that the accuracy of the model does not meet the requirements,
you can finetune the weight & quantization information of the fx.Module to get better accuracy. This process
is often called as Quantization-aware training (QAT).
- Step 4: Convert the `fx.Module` to `torchscript` (which is needed by the DISC backend).

#### Backend
BladeDISC is used as a backend in this example. Compared with fp32 optimization, quantization optimization only
requires an additional config configuration, and other usage methods are exactly the same.

## How to run this example

#### Prerequisite

1. Installing the required python package
```text
PyTorch>=1.11 (tested on 1.12.0)
transformers
datasets
evaluate
tqdm
scipy
scikit-learn
```

2. Installing the torch-quant.
```shell
cd root_of_torch_quant
python setup.py install
```

3. Installing the torch_blade
To build and install `torch_blade` package, please refer to
["Build BladeDISC from source"](https://github.com/alibaba/BladeDISC/blob/main/docs/build_from_source.md) and
["Install BladeDISC with Docker"](https://github.com/alibaba/BladeDISC/blob/main/docs/install_with_docker.md).

#### Run the example
```shell
# use OMP_NUM_THREADS to control the number of cpu cores to use
# If the target device is aarch66, use --device aarch64 instead
OMP_NUM_THREADS=1 python bert_ptq.py --device x86
```

#### Performance results
TBD