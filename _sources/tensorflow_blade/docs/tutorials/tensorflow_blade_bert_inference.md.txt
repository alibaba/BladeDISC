# Tutorial: Optimize BERT Inference with TensorFlow-Blade

In this tutorial, we show how to optimize BERT model for inference with a few lines code to call TensorRT optimization pass from TensorFlow-Blade.

**BERT**, short for Bidirectional Encoder Representations from Transformers, is one of the most popular natural language processing (NLP) model in the world. BERT models and its many varieties have been widely used for language modelling tasks. **However, with great model accuracy comes with a great amount of computations.** The latency of BERT model is quite high, since it has both massive GEMM operations and element-wise operations with lots of redundancy memory accesses, making BERT model difficult to deploy for real-time applications.
In this tutotial, we use a pre-trained "**bert-base-cased**" BERT model from **Huggingface Transformers**.

**TensorRT** is an SDK for deep learning inference powered by Nvidia, which includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications.

TensorFlow-Blade can do ahead-of-time optimization with **TensorRT** backend, providing a state-of-the-art inference performance on Nvidia GPGPUs.

As for TF-TRT inside TensorFlow, since it is compiled with TensorRT 7.1.3, which does not support optimization for BERT-like models. Thus we only compare the optimized model from TensorFlow-Blade with the origin TensorFlow baseline.
```bash
python3 -c 'from tensorflow.python.compiler.tensorrt import trt_convert as trt;print(trt._pywrap_py_utils.get_linked_tensorrt_version())'

(7, 1, 3)
```

The content of this tutorial is as following.
  - [Prologue: Prepare](#prologue-prepare)
  - [Optimize with TensorFlow-Blade's TensorRT backend](#optimize-with-tensorflow-blade)
  - [Epilogue: Run Inference for origin and optimized model](#epilogue-run-inference-for-origin-and-optimized-model)

## Prologue: Prepare
### environment setup
These packages are required:
- tensorflow-gpu == 2.4.0
- transformers
- tensorflow_blade

To build and install `tensorflow_blade` package, please refer to
["Installation of TensorFlowBlade"](../build_from_source.md) and
["Install BladeDISC With Docker"](../../docs/install_with_docker.md) to get a pre-build tensorflow-runtime docker.

The system environments and packages used in this tutorial:

- Docker Image: bladedisc/bladedisc:latest-runtime-tensorflow2.4
- Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz, 96CPU
- Nvidia Driver 470.57.02
- CUDA 11.0
- CuDNN 8.2.1

### get a pre-trained BERT base-uncased from hugging face and prepare data
```python
model = TFBertModel.from_pretrained("bert-base-cased")
```
Then get the tf.GraphDef and model fetched using functions from the [tutorial scripts](../../examples/TensorFlow/Inference/CUDA/BERT/TensorRT/bert_inference_opt.py).

```python
origin_graph_def, fetches = get_tf_graph_def()
model_outputs = [fetch.split(":")[0] for fetch in fetches]

feed_dicts = list()
feed_dicts.append({
    'input_ids:0' : np.ones((1, 5), dtype=int),
})
```

## Optimize model with TensorFlow-Blade
We will use TensorFlow-Blade TensorRT optimization pass, which can be imported like this:
```python
from tf_blade.gpu.tf_to_trt import Tf2TrtOpt
```

```python
opt_pass = Tf2TrtOpt()

opt_graph_def = opt_pass.optimize_graph_def(
    origin_graph_def, model_outputs, feed_dicts, True,
)
```

## Epilogue: Run Inference for origin/optimized model and compare result

```python
def run_benchmark(graph_def, fetches, feed_dict, model_name):
    tf.compat.v1.reset_default_graph()
    session_config = tf.compat.v1.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=session_config) as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
        output = sess.run(fetches, feed_dict)
        # Warmup!
        for i in range(0, 100):
            sess.run(fetches, feed_dict)

        # Benchmark!
        num_runs = 300
        start = time.time()
        for i in range(0, num_runs):
            sess.run(fetches, feed_dict)
        elapsed = time.time() - start
        rt_ms = elapsed / num_runs * 1000.0

        # Show the result!
        print("Latency of {} model: {:.2f} ms".format(model_name, rt_ms))
    return output
```

```python
output_origin = run_benchmark(origin_graph_def, fetches, feed_dicts[0], "origin")
output_opt = run_benchmark(opt_graph_def, fetches, feed_dicts[0], "optimized")
assert(len(output_origin) == len(output_opt))
for i in range(len(output_origin)):
    assert(np.allclose(output_origin[i], output_opt[i], rtol=1e-6, atol=1e-3))
```

And we can get the following result:
```shell
Latency of origin model: 12.40ms
Latency of optimized model: 3.39ms
```
There we can see, using TensorRT's fp32 optimization, we get a **3.66X** speedup for BERT model inference with acceptable numerical errors.

The complete python scripts for this tutorial can be found at [TensorFlow BERT Inference with TensorRT](https://github.com/alibaba/BladeDISC/tree/main/examples/TensorFlow/Inference/CUDA/BERT/TensorRT).
