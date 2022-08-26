# TorchDisc Replay Toolkit

TorchDisc replay toolkit usage:

1. dump TorchScript grpah

    ``` bash
    export TORCH_DISC_REPLAY_PATH=replay_dump
    python train.py
    ```

    this will dump the graph and input data on `replay_dump/<grpah-hash>` director.


2. load and replay the graph

    It's easy to replay the graph with Python program:

    ``` python
    import torch_blade
    import torch._lazy as ltc
    import torch_blade._torch_blade._ltc as disc_ltc
    torch_blade.init_ltc_disc_backend()

    disc_ltc.load_and_replay("replay_dump/712e64888e349b5ef2b667f78259986c", 10, 10)
    ```

    fill the dump path in `load_and_replay` function, and the program will output the performance as the following:

    ``` text
    ....
    warmup: 10 iters: 10 every iteration cost: :452.878 ms
    ```
