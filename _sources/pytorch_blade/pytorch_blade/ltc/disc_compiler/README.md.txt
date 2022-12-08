# TorchDisc Replay Toolkit

To make profiling easier and faster, TorchDISC provides a replay toolkit to replay
a TorchScript program.  With this replay toolkit, developers do not need to re-run a whole
training or inference program to checking the effect, instead of a running a disc sub-graph.

It's easy to use the replay toolkit with the 2 steps:

1. enable replay toolkit with setting the environment variable `TORCH_DISC_ENABLE_REPLAY `

    ``` bash
    export TORCH_DISC_ENABLE_REPLAY=true
    python train.py
    ```

    the replay toolkit will dump TorchScript graph and input data into a path as these log pattern:

    1. the graph after lazy TSBackend lowering:

        ``` text
        ...
        replay toolkit dump TorchScript program and data on: /tmp/xxxxx
        ```

    1. the sub-graph that Disc to be compiled:

        ``` text
        replay toolkit dump cluster on: /tmp/xxxxx
        ```

    Just pick one of these replay record path and load them with the replay program.

2. write a replay program to replay a TorchScript graph:

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
