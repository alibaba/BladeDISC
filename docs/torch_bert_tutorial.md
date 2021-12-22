## Tutorial: Optimize and inference BERT with BladeDISC(PyTorch)

BladeDISC brings supports to PyTorch inference. It's clean and easy since one can compile PyTorch models in place
with only a few lines of code while maintaining the exact same original PyTorch code and interface.

In this tutorial, we would like to introduce how to compile a pre-trained BERT-based sematic model from
HuggingFace with BladeDISC.

Nowadays, neural networks based on transformers are very popular. They have been used in many domains such as NLP,
CV, and ASR. The main building blocks of Transformers are MHA(Multi-Head Attention), FFN, Add & Norm. Usually,
there are many redundancy memory accesses between those layers. Also, the MHA and FFN are computing intensitive.
Even more, one would like to feed the neural networks with fully dynamic lengths of sequences and sizes of images.

Intuitively, mix-precision and kernel fusion would be helpful to speedup the inference. BladeDISC makes your transformers available to those optimizations while persists the ability of the module to forward dynamic inputs. Let's show the example.


### Quick tour

The prerequisite wheels must be installed before going through the tour:

+ torch >= 1.6.0
+ transformers
+ torch_addons

To build and install our BladeDISC `torch_addons` package, please refer to
["Installation of BadeDISC TorchAddons"](../pytorch_addons/README.md).

The system environments and packages:

+ CentOS 7
+ Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz, 96CPU
+ Nvidia Driver 470.57.02
+ CUDA 11.0
+ CuDNN 8.2.1

#### 1. Load the pre-trained BERT module and tokenizer from HuggingFace


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline

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

input_strs = ["We are very happy to show you the story.", "We hope you don't hate it."]
results = classifier(input_strs)

for inp_str, result in zip(input_strs, results):
    print(inp_str)
    print(f"\tlabel: {result['label']}, with score: {round(result['score'], 4)}")
```

    We are very happy to show you the story.
    	label: 5 stars, with score: 0.6456
    We hope you don't hate it.
    	label: 5 stars, with score: 0.2365


We have build a sentiment analysis classifier pipeline that use pre-trained BERT base from
[NLP Town](https://huggingface.co/nlptown). The full example can be found from HuggingFace's
[Transformers Quick Tour](https://huggingface.co/docs/transformers/quicktour).
Please look at it if you are interested.

#### 2. Optimize the module with BladeDISC torch_addons

The optimization and compilation with BladeDISC can be simple with a couple of lines.


```python
import torch_addons

inputs_str = "Hey, the cat is cute."
inputs = plain_tokenizer(inputs_str, return_tensors="pt")

torch_config = torch_addons.config.Config()
torch_config.enable_mlir_amp = False # disable mix-precision
with torch.no_grad(), torch_config:
    # BladeDISC torch_addons optimize will return an optimized TorchScript
    optimized_ts = torch_addons.optimize.optimize(model, allow_tracing=True, model_inputs=tuple(inputs))

# The optimized module could be saved as a TorchScript
torch.jit.save(optimized_ts, "opt.disc.pt")
```

A tuple of inputs arguments should be provided so that some auxiliary information will be inferred such as
data types and ranks. The context `torch.no_grad()` hints that there could be no gradient calculations because
it's optimizing for inference. 

The optimization configurations could be passing through `torch_addons.config.Config()`.
Currently, we have turned off the mix-precision.

`torch_addons.optimize.optimize` takes an instance of `torch.nn.Module` or `torch.jit.ScriptModule` as input model.
Before compiling a `torch.nn.Module`, BladeDISC trys to script it into `torch.jit.ScriptModule` recursively.

If `allow_tracing=True` was set, tracing will also be tried after scripting failed.
Then a clustering algorithmn will be run to find subgraphs and lower them to BladeDISC backbend.

Finally, let’s serialize the script module have been compiled into a file named with “opt.disc.pt”.

#### 3. Benchmark the optimized Module


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


input_strs = ["We are very happy to show you the story.", "We hope you don't hate it."]
bench_and_report(input_strs)
```

    Seqlen: [40, 26]
    Baseline: 14.193938970565796 ms
    BladeDISC: 3.432901382446289 ms
    BladeDISC speedup: 4.134677169331087


The benchmark shows the speedup after compiled with BladeDISC.

#### 4. Play with the optimized Module



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
    print(f"\tlabel: {result['label']}, with score: {round(result['score'], 4)}")
```

    We are very happy to show you the story.
    	label: 5 stars, with score: 0.6456
    We hope you don't hate it.
    	label: 5 stars, with score: 0.2365


There will be a few warnings about the optimized model is not supported for sentiment analysis, which is expected
since that it is not registered. Now you must be curious about the predicted result compared to the original model.
To use with a large dataset, look at
[iterating over a pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines).

Some more examples with the following:


```python
input_strs = ["I really like the new design of your website!",
              "I’m not sure if I like the new design.",
              "The new design is awful!",
              "It would be awesome if you give us feedback!",]

print("Predict with Baseline PyTorch model:")
results = classifier(input_strs)
for inp_str, result in zip(input_strs, results):
    print(inp_str)
    print(f"\tlabel: {result['label']}, with score: {round(result['score'], 4)}")

print("Predict with BladeDISC optimized model:")
results = blade_classifier(input_strs)
for inp_str, result in zip(input_strs, results):
    print(inp_str)
    print(f"\tlabel: {result['label']}, with score: {round(result['score'], 4)}")
```

    Predict with Baseline PyTorch model:
    I really like the new design of your website!
    	label: 5 stars, with score: 0.6289
    I’m not sure if I like the new design.
    	label: 3 stars, with score: 0.5344
    The new design is awful!
    	label: 1 star, with score: 0.8435
    It would be awesome if you give us feedback!
    	label: 5 stars, with score: 0.4164
    Predict with BladeDISC optimized model:
    I really like the new design of your website!
    	label: 5 stars, with score: 0.6289
    I’m not sure if I like the new design.
    	label: 3 stars, with score: 0.5344
    The new design is awful!
    	label: 1 star, with score: 0.8435
    It would be awesome if you give us feedback!
    	label: 5 stars, with score: 0.4164


Looks good! Next turn on mix-precision configuration `torch_config.enable_mlir_amp = True` and try it. Wish you have a good time.

### Deep dive

To have a glance at what BladeDISC have done during the optimization, one could print code of the forward
method of the optimized module with the following example:


```python
# print the optimized code
print(optimized_ts.code)
```

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
    


As the printed code shows, almost the whole forward computations were clustered and lowered to BladeDISC
(aka, the cluster `disc_grpx_lenxxx_x` where the optimizations happened).

Another debugging tool is to set the environment variable `TORCH_ADDONS_DEBUG_LOG=on`. With the env variable set,
the TorchScript subgraphs, their MHLO module lowered, and the compilation logs would be dumped and saved to local
directory dump_dir. Since the dumped files grow very faster if the Module is large enough, we would like to choose
a tiny DNN to demostrate the process:



```python
import os
# set TORCH_ADDONS_DEBUG_LOG=on
os.environ["TORCH_ADDONS_DEBUG_LOG"] = "on"
# do BladeDISC optimization
w = h = 10
dnn = torch.nn.Sequential(
            torch.nn.Linear(w, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, w),
            torch.nn.ReLU()).cuda().eval()
with torch.no_grad():
    # BladeDISC torch_addons optimize will return an optimized TorchScript
    opt_dnn_ts = torch_addons.optimize.optimize(
        dnn, allow_tracing=True, model_inputs=(torch.ones(w, h).cuda(),))

# print optimized code
print(opt_dnn_ts.code)

# list the debug files dumped
!ls dump_dir
```

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


Those debug information would be useful when you run into problem. If the optimization isn't peforming as expected,
please see the [Troubles Shooting Guide]().
