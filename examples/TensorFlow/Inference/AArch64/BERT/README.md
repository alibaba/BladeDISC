# DISC optimization example for TensorFlow inference.

This repository provides a script showing how to optimize a BERT large inference
model with DISC on AArch64 CPU.


## Content.

This repository contains a script to download a frozen BERT large model, whose
batch-size is dynamic and can be specified at runtime. We prepare a script,
named `main.py`, to load the frozen model and prepare fake input data for a
specified batch-size. The script shows how to enable DISC with a few lines of
code.


## Configure and run.

```bash
# Download model:
sh download_model.sh

# Run:
python main.py
```

By configuring `optimize_config` as `'DISC'`, `'XLA'` or `None` in the script,
you can run the model with XLA, DISC or without any optimization.


## Performance results.

We evaluate this example on Neoverse-N1 using 16 threads, with 10
inferences of different batch-size settings. The overall execution time of the
10 inferences is as following.

| TensorFlow  |    XLA    |    DISC    |
|-------------|-----------|------------|
|   4.60 s    |   22.2s   |    2.67s   |

DISC shows a 1.72x speedup over TensorFlow. XLA shows negative optimization as
it lacks support of dynamic shape and has poor multi-threading performance.
