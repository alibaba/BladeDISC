# TensorFlow-Blade optimization example for TensorFlow inference.

This repository provides a script showing how to optimize a BERT large inference
model with TensorFlow-Blade's ahead-of-time optimization with TensorRT.


## Just run python script

```bash
# Run:
python3 bert_inference_opt.py
```

## Performance results.

We evaluate origin model and optimized model on T4 GPU, with 1000 inferences.
The overall execution time of the 1000 inferences is as following.

| TensorFlow  |  TensorFlow-Blade  |
|-------------|--------------------|
|   12.40ms   |       3.39ms       |

TensorFlow-Blade's TensorRT optimization shows a 3.66X speedup over TensorFlow.
