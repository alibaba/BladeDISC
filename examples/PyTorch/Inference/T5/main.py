# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# Enable stitch fusion optimization.
os.environ["DISC_ENABLE_STITCH"] = "true"
os.environ["DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE"] = "true"

from transformers import T5Tokenizer, T5Model, T5Config
import torch
import time
import ctypes

import torch_blade
import torch_blade.tensorrt

# from common_utils import benchmark

_cudart = ctypes.CDLL('libcudart.so')


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)


def trace_model(model, inputs, amp: bool):
    if amp is True:
        with torch.cuda.amp.autocast(enable_amp), torch.no_grad():
            traced_model = torch.jit.trace(model, inputs, strict=False)
    else:
        traced_model = torch.jit.trace(model, inputs, strict=False)
    torch._C._jit_pass_inline(traced_model.graph)
    return traced_model


def evaluate_torch(model, inputs):
    # warmup
    for i in range(20):
        model(*tuple(inputs))

    iters = 100
    tic = time.time()
    for i in range(iters):
        model(*tuple(inputs))
    avg_time = (time.time() - tic) / iters
    print("average time in {} iterations: {} seconds".format(iters, avg_time))

    # profile start
    cu_prof_start()
    model(*tuple(inputs))
    cu_prof_stop()
    # profile end


def disc_optimize(model, inputs, out_file: str):
    torch_config = torch_blade.config.Config()
    torch_config.enable_mlir_amp = False  # disable mix-precision

    traced_model = torch.jit.trace(model.cuda(), inputs,
                                   strict=False).cuda().eval()

    torch._C._jit_pass_inline(traced_model._c.forward.graph)
    torch._C._jit_pass_remove_dropout(traced_model._c)

    with torch.no_grad(), torch_config:
        # BladeDISC torch_blade optimize will return an optimized TorchScript
        optimized_ts = torch_blade.optimize(traced_model,
                                            allow_tracing=True,
                                            model_inputs=tuple(inputs))
    torch.jit.save(optimized_ts, out_file)


def blade_trt_optimize(model, inputs, fp16: bool, out_file: str):
    cfg = torch_blade.Config.get_current_context_or_new().clone()
    cfg.optimization_pipeline = torch_blade.tensorrt.backend_name()
    cfg.customize_onnx_opset_version = 12
    cfg.enable_fp16 = fp16

    traced_model = torch.jit.trace(model.cuda().eval(), inputs,
                                   strict=False).cuda().eval()

    with cfg, torch_blade.logging.logger_level_context('INFO'):
        opt_model = torch_blade.optimize(traced_model,
                                         False,
                                         model_inputs=tuple(inputs))
    torch.jit.save(opt_model, out_file)


def run():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you",
        return_tensors="pt").input_ids  # Batch size 1
    input_ids = input_ids.cuda()
    mask = torch.ones_like(input_ids)
    decoder_input_ids = tokenizer(
        "Studies show that", return_tensors="pt").input_ids  # Batch size 1
    decoder_input_ids = decoder_input_ids.cuda()
    inputs = (input_ids, mask, decoder_input_ids)

    model = T5Model.from_pretrained("t5-small", torchscript=True).eval().cuda()
    traced_model = trace_model(model, inputs, False).eval().cuda()
    traced_model_amp = trace_model(model, inputs, True).eval().cuda()

    # Run naive torch.
    print("Naive PyTorch.")
    model = traced_model_amp
    evaluate_torch(model, inputs)

    # Run BladeDISC optimization.
    print("BladeDISC Optimization.")
    disc_optimize(traced_model_amp, inputs, 't5_amp.disc.pt')
    model = torch.jit.load('t5_amp.disc.pt').cuda().eval()
    evaluate_torch(model, inputs)

    # Run TorchBlade-TensorRT optimization.
    print("TorchBlade-TensorRT Optimization.")
    blade_trt_optimize(model, inputs, True, 't5_amp.trt.pt')
    model = torch.jit.load('t5_amp.trt.pt').cuda().eval()
    evaluate_torch(model, inputs)


if __name__ == '__main__':
    run()

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# #tokenizer = T5Tokenizer.from_pretrained("t5-base")
# #tokenizer = T5Tokenizer.from_pretrained("t5-large")

# config = T5Config(torchscript=True, )
# model = T5Model(config)
# model.eval()
# model = T5Model.from_pretrained("t5-small", torchscript=True)
# #model = T5Model.from_pretrained("t5-base", torchscript=True)
# #model = T5Model.from_pretrained("t5-large", torchscript=True)

# input_ids_a = tokenizer(
#     "Studies have been shown that owning a dog is good for you",
#     return_tensors="pt").input_ids  # Batch size 1
# input_ids_b = tokenizer(
#     "Studies have been shown that owning a black and while dog together with a cat is good for you",
#     return_tensors="pt").input_ids  # Batch size 1
# decoder_input_ids = tokenizer("Studies show that",
#                               return_tensors="pt").input_ids  # Batch size 1
# input_ids_b = input_ids_b.cuda()

# input_ids_small = input_ids_a.cuda()
# mask_small = torch.ones_like(input_ids_small)
# decoder_input_ids_small = decoder_input_ids.cuda()

# input_ids_a_padded = torch.nn.functional.pad(input_ids_small, (0, 8),
#                                              "constant", 0)
# mask_a_padded = torch.nn.functional.pad(mask_small, (0, 8), "constant", 0)
# input_ids_large = torch.cat([input_ids_a_padded, input_ids_b], 0)
# mask_large = torch.cat([mask_a_padded, torch.ones_like(input_ids_b)], 0)
# decoder_input_ids_large = torch.cat(
#     [decoder_input_ids_small, decoder_input_ids_small], 0)

# print("inputs:")
# print(input_ids_small)
# print(mask_small)
# print(decoder_input_ids_small)
# print(input_ids_large)
# print(mask_large)
# print(decoder_input_ids_large)

# model.cuda()
# # forward pass
# #outputs = model(input_ids=input_ids, attention_mask=mask, decoder_input_ids=decoder_input_ids)
# # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# #last_hidden_states = outputs.last_hidden_state

# #print(last_hidden_states)
# inputs_small = (input_ids_small, mask_small, decoder_input_ids_small)
# inputs_large = (input_ids_large, mask_large, decoder_input_ids_large)

# inputs = [inputs_small, inputs_large]
# #baseline:
# benchmark(model, inputs_small)

# enable_amp = False

# with torch.cuda.amp.autocast(enable_amp), torch.no_grad():
#     traced_model = torch.jit.trace(model, inputs_small, strict=False)
#     torch._C._jit_pass_inline(traced_model.graph)
#     with open('opt.traced.py', 'w') as f:
#         f.write(traced_model.code)
#     torch.jit.save(traced_model, "opt.traced.pt")
# traced_model = torch.jit.load("opt.traced.pt")

# torch_config = torch_blade.config.Config()
# torch_config.optimization_pipeline = torch_blade.tensorrt.backend_name()
# torch_config.enable_onnx_shape_white_list = True
# torch_config.enable_fp16 = True  # disable mix-precision
# torch_config.dynamic_tuning_shapes = {
#     "min": inputs_small,
#     "max": inputs_large,
#     "opts": [inputs_small, inputs_large],
# }
# with torch.no_grad(), torch_config, torch.cuda.amp.autocast(enable_amp):
#     # BladeDISC torch_blade optimize will return an optimized TorchScript
#     optimized_ts = torch_blade.optimize(traced_model,
#                                         allow_tracing=True,
#                                         model_inputs=inputs_small)
#     print("optimization done")
#     with open('opt.disc.py', 'w') as f:
#         f.write(optimized_ts.code)

#     # The optimized module could be saved as a TorchScript
#     torch.jit.save(optimized_ts, "opt.disc.pt")

#     optimized_ts = torch.jit.load("opt.disc.pt")

#     benchmark(optimized_ts, inputs_small)
