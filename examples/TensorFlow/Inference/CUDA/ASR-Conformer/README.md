# DISC optimization example for TensorFlow inference.

This repository provides a script showing how to optimize an ASR inference model
with DISC just-in-time.


## Content.

This repository contains a script to download a saved ASR Conformer model.
This model is saved from [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR).

We prepare a script, named `main.py`, to load the saved model and prepare fake input data.
The script shows how to enable DISC with a few lines of code.


## Configure and run.

```bash
# Download and extract saved model:
sh download_model.sh

# Run:
python main.py
```

By configuring `optimize_config` as `'DISC'`, `'XLA'` or `None` in the script,
you can run the model with XLA, DISC or without any optimization.


## Performance results.

TBD.

| TensorFlow  |    XLA    |    DISC    |
|-------------|-----------|------------|
|             |           |            |

