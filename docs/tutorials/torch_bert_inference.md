# Tutorial: Optimize and Inference BERT with TorchBlade

- [Quick Tour](#quick-tour)
  - [Load the Pre-Trained BERT Module and Tokenizer From HuggingFace](#load-the-pre-trained-bert-module-and-tokenizer-from-huggingface)
  - [Optimize the Module With TorchBlade](#optimize-the-module-with-torchblade)
  - [Benchmark the Optimized Module](#benchmark-the-optimized-module)
  - [Play With the Optimized Module](#play-with-the-optimized-module)
- [Deep Dive](#deep-dive)

TorchBlade supports to compile PyTorch models in place with only a few lines of
code while maintaining the same original PyTorch code and interface.

In this tutorial, we will introduce how to compile a pre-trained BERT-based
semantic model from HuggingFace with BladeDISC.

Nowadays, neural networks based on transformers are very popular and used in
many application domains, such as NLP and CV. The main building blocks of
Transformers are MHA(Multi-Head Attention), FFN, Add & Norm. Usually, there are
many redundancy memory accesses between those layers. Also, the MHA and FFN are
computing intensitive. One would also like to feed the neural networks with
fully dynamic lengths of sequences and sizes of images.

Intuitively, mixed-precision and kernel fusion would be helpful to speed up the
inference. TorchBlade makes your transformers available to those optimizations
while persists the ability of the module to forward dynamic inputs. Let's show
the example.

## Quick Tour

These packages are required before going through the tour:

- torch >= 1.6.0
- transformers
- torch_blade

To build and install `torch_blade` package, please refer to
["Installation of TorchBlade"](../build_from_source.md) and
["Install BladeDISC With Docker"](../install_with_docker.md).

The system environments and packages used in this tutorial:

- Docker Image: bladedisc/bladedisc:latest-runtime-torch1.7.1
- Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz, 96CPU
- Nvidia Driver 470.57.02
- CUDA 11.0
- CuDNN 8.2.1

### Load the Pre-Trained BERT Module and Tokenizer From HuggingFace

```python
import torch
from transformers import (
 pipeline,
 AutoTokenizer,
 AutoModelForSequenceClassification,
 TextClassificationPipeline,
)

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# get tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_name)

# place model to cuda and set it to evaluate mode
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda().eval()

def plain_tokenizer(inputs_str, return_tensors):
  inputs = tokenizer(inputs_str, return_tensors=return_tensors, padding=True)
  inputs = dict(map(lambda x: (x[0], x[1].cuda()), inputs.items()))

  return (inputs['input_ids'].cuda(), inputs['attention_mask'].cuda(), inputs['token_type_ids'].cuda(),)

class PlainTextClassificationPipeline(TextClassificationPipeline):
  def _forward(self, model_inputs):
    return self.model(*model_inputs)

# build a sentiment analysis classifier pipeline
classifier = pipeline('sentiment-analysis',
           model=model,
           tokenizer=plain_tokenizer,
           pipeline_class=PlainTextClassificationPipeline,
           device=0)

input_strs = [
 "We are very happy to show you the story.",
 "We hope you don't hate it."]

results = classifier(input_strs)

for inp_str, result in zip(input_strs, results):
  print(inp_str)
  print(f" label: {result['label']}, with a score: {round(result['score'], 4)}")
```

```text
We are very happy to show you the story.
 label: 5 stars, with a score: 0.6456
We hope you don't hate it.
 label: 5 stars, with a score: 0.2365
```

We have built a sentiment analysis classifier pipeline that uses a pre-trained
BERT base from [NLP Town](https://huggingface.co/nlptown). The full example can
be found at from HuggingFace's
[Transformers Quick Tour](https://huggingface.co/docs/transformers/quicktour).

### Optimize the Module With TorchBlade

Add a few lines of codes on the original scripts:

```python
import torch_blade

inputs_str = "Hey, the cat is cute."
inputs = plain_tokenizer(inputs_str, return_tensors="pt")

torch_config = torch_blade.config.Config()
torch_config.enable_mlir_amp = False # disable mix-precision
with torch.no_grad(), torch_config:
  # BladeDISC torch_blade optimize will return an optimized TorchScript
  optimized_ts = torch_blade.optimize(model, allow_tracing=True, model_inputs=tuple(inputs))

# The optimized module could be saved as a TorchScript
torch.jit.save(optimized_ts, "opt.disc.pt")
```

A tuple of inputs arguments should be provided to infer some auxiliary
information, such as data types and ranks. The context `torch.no_grad()` hints
that there could be no gradient calculations because it optimizes inference.

The optimization configurations could be passing through
`torch_blade.config.Config()`. Currently, we have turned off the mix-precision.

`torch_blade.optimize` takes an instance of `torch.nn.Module` or
`torch.jit.ScriptModule` as input model. Before compiling a `torch.nn.Module`,
BladeDISC trys to script it into `torch.jit.ScriptModule` recursively.

If set `allow_tracing=True`, BladeDISC will also try tracing after scripting
fails. Then it runs a clustering algorithm to find subgraphs and lower them to
BladeDISC backbend.

Finally, let's serialize the script module that has been compiled into a file
named with “opt.disc.pt”.

### Benchmark the Optimized Module

```python
import time

@torch.no_grad()
def benchmark(model, inputs, num_iters=1000):
  for _ in range(10):
    model(*inputs)
  torch.cuda.synchronize()

  start = time.time()
  for _ in range(num_iters):
    model(*inputs)
  torch.cuda.synchronize()
  end = time.time()
  return (end - start) / num_iters * 1000.0

def bench_and_report(input_strs):
  inputs = plain_tokenizer(input_strs, return_tensors="pt")
  avg_lantency_baseline = benchmark(model, inputs)
  avg_lantency_bladedisc = benchmark(optimized_ts, inputs)

  print(f"Seqlen: {[len(s) for s in input_strs]}")
  print(f"Baseline: {avg_lantency_baseline} ms")
  print(f"BladeDISC: {avg_lantency_bladedisc} ms")
  print(f"BladeDISC speedup: {avg_lantency_baseline/avg_lantency_bladedisc}")


input_strs = [
 "We are very happy to show you the story.",
 "We hope you don't hate it."]

bench_and_report(input_strs)
```

```text
Seqlen: [40, 26]
Baseline: 14.193938970565796 ms
BladeDISC: 3.432901382446289 ms
BladeDISC speedup: 4.134677169331087
```

The benchmark shows the speedup after compiled with BladeDISC.

### Play With the Optimized Module

```python
from transformers import TextClassificationPipeline

optimized_ts.config = model.config
# build a sentiment analysis classifier pipeline
blade_classifier = pipeline('sentiment-analysis',
              model=optimized_ts,
              tokenizer=plain_tokenizer,
              pipeline_class=PlainTextClassificationPipeline,
              device=0)

results = blade_classifier(input_strs)
for inp_str, result in zip(input_strs, results):
  print(inp_str)
  print(f" label: {result['label']}, with a score: {round(result['score'], 4)}")
```

```text
We are very happy to show you the story.
 label: 5 stars, with a score: 0.6456
We hope you don't hate it.
 label: 5 stars, with a score: 0.2365
```

There will be a few warnings that the optimized model is not supported for
sentiment analysis, which is expected since it is not registered. Now you must
be curious about the predicted result compared to the original model. To use
with a large dataset, refer to
[iterating over a pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines).

Some more examples with the following:

```python
input_strs = ["I really like the new design of your website!",
       "I'm not sure if I like the new design.",
       "The new design is awful!",
       "It will be awesome if you give us feedback!",]

print("Predict with Baseline PyTorch model:")
results = classifier(input_strs)
for inp_str, result in zip(input_strs, results):
  print(inp_str)
  print(f" label: {result['label']}, with a score: {round(result['score'], 4)}")

print("Predict with BladeDISC optimized model:")
results = blade_classifier(input_strs)
for inp_str, result in zip(input_strs, results):
  print(inp_str)
  print(f" label: {result['label']}, with a score: {round(result['score'], 4)}")
```

```text
Predict with Baseline PyTorch model:
I really like the new design of your website!
 label: 5 stars, with a score: 0.6289
I'm not sure if I like the new design.
 label: 3 stars, with a score: 0.5344
The new design is awful!
 label: 1 star, with a score: 0.8435
It will be awesome if you give us feedback!
 label: 5 stars, with a score: 0.4164
Predict with BladeDISC optimized model:
I really like the new design of your website!
 label: 5 stars, with a score: 0.6289
I'm not sure if I like the new design.
 label: 3 stars, with a score: 0.5344
The new design is awful!
 label: 1 star, with a score: 0.8435
It will be awesome if you give us feedback!
 label: 5 stars, with a score: 0.4164
```

Looks good! Next you may want to turn on mix-precision configuration
`torch_config.enable_mlir_amp = True` and try it. Wish you have a good time.

## Deep Dive

To have a glance at what TorchBlade has done during the optimization, one could
print code of the forward method of the optimized model with the following
example:

```python
# print the optimized code
print(optimized_ts.code)
```

```text
def forward(self,
  input_ids: Tensor,
  attention_mask: Tensor,
  input: Tensor) -> Dict[str, Tensor]:
 seq_length = ops.prim.NumToTensor(torch.size(input_ids, 1))
 _0 = torch.add(seq_length, CONSTANTS.c0, alpha=1)
 _1 = torch.tensor(int(_0), dtype=None, device=None, requires_grad=False)
 _2 = self.disc_grp0_len1264_0
 _3 = [input, input_ids, attention_mask, _1]
 _4, = (_2).forward(_3, )
 return {"logits": _4}
```

As the printed code shows, almost the whole forward computations were clustered
and will be lowered by TorchBlade(aka, the cluster `disc_grpx_lenxxx_x` where
the optimizations happened).

Another useful debugging tool is to set the environment variable
`TORCH_BLADE_DEBUG_LOG=on`. With the env variable set, the TorchScript
subgraphs, the associated MHLO modules, and the compilation logs would be dumped
and saved to a local directory `dump_dir`. Since the dumped files can be very
large if the Module is large enough, we would like to choose a tiny DNN to
demostrate the process:

```python
import os
# set TORCH_BLADE_DEBUG_LOG=on
os.environ["TORCH_BLADE_DEBUG_LOG"] = "on"
# do BladeDISC optimization
w = h = 10
dnn = torch.nn.Sequential(
      torch.nn.Linear(w, h),
      torch.nn.ReLU(),
      torch.nn.Linear(h, w),
      torch.nn.ReLU()).cuda().eval()
with torch.no_grad():
  # BladeDISC torch_blade optimize will return an optimized TorchScript
  opt_dnn_ts = torch_blade.optimize(
    dnn, allow_tracing=True, model_inputs=(torch.ones(w, h).cuda(),))

# print optimized code
print(opt_dnn_ts.code)

# list the debug files dumped
!ls dump_dir
```

```text
def forward(self,
  input: Tensor) -> Tensor:
 _0 = (self.disc_grp0_len10_0).forward([input], )
 _1, = _0
 return _1

dump.2021_12_22-18_24_04.226587.mlir
dump.2021_12_22-18_24_04.226587.pretty.mlir
graph.2021_12_22-18_24_04.226587.txt
mhlo_compile.2021_12_22-18_24_04.226587.log
out.2021_12_22-18_24_04.226587.so
out.2021_12_22-18_24_04.226587.so.pbtxt
```
