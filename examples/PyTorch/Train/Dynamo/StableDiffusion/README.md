# StableDiffusion Examples

BladeDISC acceleartes StableDiffusion dreambooth fine-tune with 1.09X,
users can reproduce the results as the following steps, for more StableDiffusion examples,
please go to the official [diffusers repo](https://github.com/huggingface/diffusers/blob/main/examples).

## Dreambooth

### Steps

- step1:

    clone diffusers repo and install requirements in the BladeDISC runtime Docker image.

    ``` bash
    docker run --rm -it -v $PWD:/workspace -w /workspace nvcr.io/nvidia/pytorch:21.08-py3 bash
    git clone  --depth 1  --branch v0.13.0  git@github.com:huggingface/diffusers.git
    cd diffusers/examples/dreambooth && python -m pip install -r requirements.txt
    ```

- step2: 

    launch the fine-tune job with TorchBlade accelerator.

    ``` bash
    bash launch_dreambooth_train.sh
    ```

    if you want to execute with PyTorch eager mode, please remove the `--enable_disc` argument.

### Results

| Backend | Batch Size | Throughput (samples/sec) | Speedup |
| :-----: | :--------: | :----------------------: | :-----: |
| Eager | 1 | 3.11 | 1.00 |
| TorchBlade | 1 | 3.40 | 1.09 |