# Quantization Toolkit for PyTorch Models

*under development*

## 1. Quantization Basic Concepts

Deep learning model quantization is the process of reducing the precision of the weights
and activations of a neural network model in order to reduce the memory footprint
and computational complexity of the model. A common way to perform quantization is
to represent the weights and activations using fewer bits than the original model,
typically using 8 or fewer bits.

A typical workflow of quantization is as following:
- Step 0: Pre-train model
- Step 1: Calibrate quantization parameters(e.g. scal, zero point) by running forward on typical inputs.
- Step 2: [Optional] Fine-tune model with quantization constraints (QAT, Quantization Aware Training)
- Step 3: Convert original model to quantized model(e.g. convert conv/linear ops to quantized version)

This toolkit helps you to perform step 1~3, so you can compress and accelerate PyTorch models using
BladeDISC and quantization techniques.

## 2. Toolkit Usages

The main interface of toolkit is `Quantizer` class. Instantiate a quantizer object with some
configuration (like target backend type, excluded module types, etc.), then use it to create
proxy models from your original model for quantization workflow. There are 3 types of proxy
model, and corresponding methods:
- proxy model for calibration: created from `Quantizer.calib()`
- proxy model for quantization-aware training: created from `Quantizer.qat()`
- proxy model representing final quantized model: created from `Quantizer.quantize()`

A basic usage for post-training quantization:

```python
model = MyModel() # torch.nn.Module
typical_data = torch.tensor([1, 2, 3])
quantizer = Quantizer()

# create a proxy model and run forward to calibrate quantization params
quantizer.calib(model)(typical_data)

# [Optional] perform automatic mixed precision quantization
# create a proxy model and run forward to fallback few sensitive layers to float precision
amp_model = quantizer.amp(model)
amp_model(typical_data)
quantizer.fallback(amp_model, num=1)

# create a proxy model representing quantized model
quant_model = quantizer.quantize(model)

# run inference on quantized model
output = quant_model(typical_data)

# or save model as torchscript
traced = torch.jit.trace(quant_model, typical_data)
torch.jit.save(traced, 'quant.pt')

# or further optimized using BladeDISC
opt = torch_blade.optimize(quant_model)

```

## 3. End-to-End Examples

*TBD*

A initial example can be found at [bert_ptq_demo.py](bert_ptq_demo.py)
