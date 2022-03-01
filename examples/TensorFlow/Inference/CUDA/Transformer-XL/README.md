# DISC optimization example for TensorFlow inference.

This repository provides a script showing how to optimize a Transformer-XL inference
model with DISC just-in-time.


## Content.

This repository contains a script to download a frozen [Transformer-XL model]
(https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL).
We prepare a script, named `main.py`, to load the frozen model and prepare
fake input data. The script shows how to enable DISC with a few lines of code.


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

TBD.
