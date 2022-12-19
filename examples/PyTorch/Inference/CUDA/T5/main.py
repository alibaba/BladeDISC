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

from transformers import T5Tokenizer, T5Model, T5Config
import torch
import time
import ctypes

import torch_blade
import torch_blade.tensorrt

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
    with torch.cuda.amp.autocast(amp), torch.no_grad():
        traced_model = torch.jit.trace(model, inputs, strict=False, check_trace=False)
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
    torch_config.enable_force_to_cuda = True

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


def run():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you",
        return_tensors="pt").input_ids  # Batch size 1
    input_ids = input_ids.cuda()
    mask = torch.ones_like(input_ids)
    decoder_input_ids = tokenizer(
        "Studies show that", return_tensors="pt").input_ids  # Batch size 1
    decoder_input_ids = decoder_input_ids.cuda()
    inputs = (input_ids, mask, decoder_input_ids)

    model = T5Model.from_pretrained("t5-base", torchscript=True).eval().cuda()
    traced_model_amp = trace_model(model, inputs, True).eval().cuda()

    # Run naive torch.
    print("Naive PyTorch.")
    model = traced_model_amp
    evaluate_torch(model, inputs)

    # Run BladeDISC optimization.
    print("BladeDISC Optimization.")
    disc_optimize(traced_model_amp, inputs, 't5-base_amp.disc.pt')
    model = torch.jit.load('t5-base_amp.disc.pt').cuda().eval()
    evaluate_torch(model, inputs)


if __name__ == '__main__':
    run()
