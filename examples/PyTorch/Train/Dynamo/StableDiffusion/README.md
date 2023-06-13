# StableDiffusion Examples

BladeDISC acceleartes StableDiffusion dreambooth fine-tune with 1.10X,
users can reproduce the results as the following steps, for more StableDiffusion examples,
please go to the official [diffusers repo](https://github.com/huggingface/diffusers/blob/main/examples).

## Dreambooth

### Steps

- step1:

    clone diffusers repo and install requirements in the BladeDISC runtime Docker image.

    ``` bash
    docker run --rm -it -v $PWD:/workspace -w /workspace bladedisc/bladedisc:latest-runtime-torch-pre-cu117 bash
    pip install diffusers==0.17.0
    ```

- step2: 

    launch the fine-tune job with TorchBlade accelerator.

    ``` bash
    cd examples/PyTorch/Train/Dynamo/StableDiffusion
    bash launch_dreambooth_train.sh
    ```

    if you want to execute with PyTorch eager mode, please remove the `--enable_torch_compile` argument.

### Results

| Backend | Batch Size | Throughput (samples/sec) | Speedup |
| :-----: | :--------: | :----------------------: | :-----: |
| Eager | 1 | 3.02 | 1.00 |
| TorchBlade | 1 | 3.34 | 1.10 |