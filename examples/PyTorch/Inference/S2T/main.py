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

import torch
import time
import numpy as np
import ctypes
from transformers import Speech2TextForConditionalGeneration

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


class ModelWrapper(torch.nn.Module):

    def __init__(self, original_model, amp: bool):
        super().__init__()
        self.model = original_model
        self.amp = amp

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs,
    ):
        if self.amp is True:
            with torch.cuda.amp.autocast():
                result = self.model(input_features, attention_mask,
                                    decoder_input_ids, **kwargs)
        else:
            result = self.model(input_features, attention_mask,
                                decoder_input_ids, **kwargs)
        keys = result.keys()
        return tuple([result[key] for key in keys])


def evaluate_torch(model, inputs):
    # warmup
    for i in range(20):
        model(inputs[0], inputs[1], inputs[2])

    iters = 100
    tic = time.time()
    for i in range(iters):
        model(inputs[0], inputs[1], inputs[2])
    avg_time = (time.time() - tic) / iters
    print("average time in {} iterations: {} seconds".format(iters, avg_time))

    # profile start
    cu_prof_start()
    model(inputs[0], inputs[1], inputs[2])
    cu_prof_stop()


def disc_optimize(model, inputs, out_file: str):
    torch_config = torch_blade.config.Config()
    torch_config.enable_mlir_amp = False  # disable mix-precision

    traced_model = torch.jit.trace(model.cuda().eval(), inputs,
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
    batch_size = 1
    sequence_length = 584
    feature_size = 80
    model = Speech2TextForConditionalGeneration.from_pretrained(
        "facebook/s2t-small-librispeech-asr").eval()
    input_features = torch.rand(batch_size, sequence_length, feature_size)
    attention_mask = torch.rand(batch_size, sequence_length).long()
    decoder_input_ids = torch.tensor([[2]] * batch_size)
    inputs = [
        input_features.cuda(),
        attention_mask.cuda(),
        decoder_input_ids.cuda()
    ]

    # Trace the model. Batch dim is static and seq_len dim is dynamic after
    # tracing.
    s2t_model_amp = ModelWrapper(model.cuda(), True)

    # Run naive torch.
    model = s2t_model_amp
    evaluate_torch(model, inputs)

    # Run BladeDISC optimization.
    disc_optimize(s2t_model_amp, inputs, 's2t_amp.disc.pt')
    model = torch.jit.load('s2t_amp.disc.pt').cuda().eval()
    evaluate_torch(model, inputs)


if __name__ == '__main__':
    run()
