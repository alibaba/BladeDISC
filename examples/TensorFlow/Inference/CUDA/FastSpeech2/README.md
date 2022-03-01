# DISC optimization example for TensorFlow inference.

This repository provides a script showing how to optimize a TTS inference model
with DISC just-in-time.


## Content.

This repository contains a script to download a saved TTS FastSpeech2 model.
This model is saved from [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS):
`fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")`.

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

