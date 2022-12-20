# Accelerate Inference of PyTorch Models from HuggingFace Transformers using BladeDISC

*(under development)*

## 1. Prerequisites

- torch==1.12.0+cu113
- transformers>=4.24.0
- Latest BladeDISC (torch_blade), pull docker image `bladedisc/bladedisc:latest-runtime-torch1.12.0-cu113`

Please note that, you may need to build torch_blade from source if using difference torch and cuda versions.
Refer to [build BladeDISC for PyTorch](../../../../docs/build_from_source.md).

## 2. High-level Concepts

BladeDISC for PyTorch (a.k.a torch_blade) takes in a torchscript format model, performs compilation process 
(a series of lowering and transformations), and gives out the optimized model(with same format and 
input/output signature just like the original torchscript model).

Basic steps to accelerate inference of Transformers PyTorch models are as following:
- step 1: export pretrained model as torchscript format;
- step 2: call torch_blade optimization;
- step 3: map original model io to optimized model io  

Despite of straightforwardness of these steps, some trivial details need to be handled.
We provide a generic adapter to simplify the intergration of BladeDISC for basic usecases.
The adapter can also serve as a reference implementation, which can be further customized
for advanced usecases.


## 3. Adapter Usages

The adapter provides easy to use interfaces, so you can accelerate HuggingFace Transformers
models with a few modification on existing usages.

### 3.1 `pipeline`

Use `blade_adapter.pipeline` (similar as `transfromers.pipeline`) to build a `Pipeline` object containing
blade-optimized model instead of original pretrained model.

```
from blade_adapter import pipeline

classifier = pipeline(task='text-classification')
print(classifier('I like you'))
print(classifier('I hate you'))
```

Like using `transformers.pipeline`, you can specify task name (will use default model of the task) or model id (will use default task for the model). Please note, when setting model as a PreTrainedModel
object, you need create the model with `torchscript=True` option. Please find more examples of `pipeline`
in [tests/test_pipeline.py](tests/test_pipeline.py)

### 3.2 `optimize`

Use `blade_adapter.optimize` to get a optimized model object for flexible integration.

```
from blade_adapter import optimize

model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased', torchscript=True)
blade_model = optimize(task='fill-mask', model=model)

inputs = {
    'input_ids': torch.tensor(...)
    ...
}
print(blade_model(**inputs))
# {'logits': tensor(...)}
```

`optimize` can take in task name, or model id, or PreTrainedModel object, and then perform
tracing and compilation. Please note, if input PreTrainedModel object, you need create the model with
`torchscript=True` option. Please find more examples of `optimize` in
[tests/test_optimize.py](tests/test_optimize.py)


## 4. Speedup of Popular Models
*TBA*

## TODOs:
- More tests and benchmark results
- Support more advanced optimization options
- More task and model coverage