# DISC optimization example for TensorFlow inference.

This repository provides a script showing how to optimize a TTS inference model
with DISC just-in-time.


## Content.

This repository contains a script to download a saved TTS FastSpeech2 model.
This model is saved from [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS):
`fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")`.

We prepare a script, named `main.py`, to load the saved model and prepare fake
input data. The script shows how to enable DISC with a few lines of code.


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

We evaluate this example on T4 GPU, The CUDA version is 11.0. CuDNN version is
8.0. TensorFlow version is 2.4. The average execution time of the 100 inferences
is as following (ms).

| TensorFlow  |    XLA    |    DISC    |
|-------------|-----------|------------|
|    24.47    |   17.97   |    15.74   |

DISC shows a 1.55x speedup over basic TensorFlow, and 1.14x speedup over XLA
static optimization.
