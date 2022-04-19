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

from transformers import TransfoXLModel, TransfoXLTokenizer
import torch
import time
import ctypes

import torch_blade

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
        with torch.cuda.amp.autocast(amp), torch.no_grad():
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
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").cuda()
    # inputs = torch.tensor([[14049, 2, 617, 3225, 23, 16072]]).long().to(device)

    model = TransfoXLModel.from_pretrained('transfo-xl-wt103',
                                           torchscript=True).cuda().eval()
    traced_model_amp = trace_model(model, inputs, True).eval().cuda()

    # Run naive torch.
    print("Naive PyTorch.")
    evaluate_torch(traced_model_amp, inputs)

    # Run BladeDISC optimization.
    print("BladeDISC Optimization.")
    disc_optimize(traced_model_amp, inputs, 'trans-xl_amp.disc.pt')
    model = torch.jit.load('trans-xl_amp.disc.pt').cuda().eval()
    evaluate_torch(model, inputs)

    # Run TorchBlade-TensorRT optimization.
    print("TorchBlade-TensorRT Optimization.")
    blade_trt_optimize(model, inputs, True, 'trans-xl_amp.trt.pt')
    model = torch.jit.load('trans-xl_amp.trt.pt').cuda().eval()
    evaluate_torch(model, inputs)


if __name__ == '__main__':
    run()
