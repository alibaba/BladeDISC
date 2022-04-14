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

#os.environ["TORCH_BLADE_DEBUG_LOG"] = "on"
os.environ["DISC_ENABLE_ASTITCH"] = "true"

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


def run(optimize_config: str = None):
    model = TransfoXLModel.from_pretrained('transfo-xl-wt103',
                                           torchscript=True).cuda()
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").cuda()
    # inputs = torch.tensor([[14049, 2, 617, 3225, 23, 16072]]).long().to(device)

    model.eval()
    traced_model = torch.jit.trace(model, inputs, strict=False)

    model = traced_model
    if optimize_config is 'DISC':
        # Optimize with BladeDISC
        torch_config = torch_blade.config.Config()
        torch_config.enable_mlir_amp = True  # enable mix-precision
        torch_config.enable_force_to_cuda = True
        with torch.no_grad(), torch_config:
            # It will return an optimized TorchScript
            optimized_ts = torch_blade.optimize(traced_model,
                                                allow_tracing=True,
                                                model_inputs=inputs)

        output_pt_file = "trans-xl_disc_opt.pt"
        torch.jit.save(optimized_ts, output_pt_file)
        model = torch.jit.load(output_pt_file).eval()
    elif optimize_config is 'TRT':
        cfg = torch_blade.Config.get_current_context_or_new().clone()
        cfg.optimization_pipeline = torch_blade.tensorrt.backend_name()
        cfg.customize_onnx_opset_version = 12
        cfg.enable_fp16 = True

        with cfg, torch_blade.logging.logger_level_context('DEBUG'):
            print("Go!")
            opt_model = torch_blade.optimize(traced_model.cuda(),
                                             False,
                                             model_inputs=inputs)

        print("Optimize finish!")
        output_pt_file = "trans-xl_trt_opt.pt"
        torch.jit.save(opt_model, output_pt_file)
        print("Done!")
        model = torch.jit.load(output_pt_file).eval()

    # warmup
    for i in range(100):
        output = model(inputs)

    # normal measure
    tic = time.time()
    for i in range(100):
        output = model(inputs)
    rt_ms = (time.time() - tic) / 100
    print(f'average exec time: {rt_ms} s')

    cu_prof_start()
    for i in range(100):
        output = model(inputs)
    cu_prof_stop()


if __name__ == '__main__':
    # `optimize_config` can be 'TRT', 'DISC' or None.
    run(optimize_config='DISC')
