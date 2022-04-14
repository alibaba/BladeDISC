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

# os.environ["TORCH_BLADE_DEBUG_LOG"] = "on"
os.environ["DISC_ENABLE_STITCH"] = "true"

import torch
import time
import numpy as np
import ctypes

import torch_blade
# import torch_blade.tensorrt

_cudart = ctypes.CDLL('libcudart.so')


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)


class wrapper(torch.nn.Module):

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs,
    ):
        result = self.model(input_features, attention_mask, decoder_input_ids,
                            **kwargs)
        keys = result.keys()
        return tuple([result[key] for key in keys])


def printTensorShapesInTuple(value):
    if type(value) is tuple:
        for val in value:
            printTensorShapesInTuple(val)
    elif type(value) is list:
        for val in value:
            printTensorShapesInTuple(val)
    else:
        print(value.shape)


def run(optimize_config: str = None):
    from transformers import Speech2TextForConditionalGeneration

    batch_size = 1
    sequence_length = 584
    feature_size = 80
    model = Speech2TextForConditionalGeneration.from_pretrained(
        "facebook/s2t-small-librispeech-asr").eval()
    input_features = torch.rand(batch_size, sequence_length, feature_size)
    attention_mask = torch.rand(batch_size, sequence_length).long()
    decoder_input_ids = torch.tensor([[2]] * batch_size)

    # Trace the model. Batch dim is static and seq_len dim is dynamic after
    # tracing.
    traced_model = torch.jit.trace(wrapper(model.cuda()), [
        input_features.cuda(),
        attention_mask.cuda(),
        decoder_input_ids.cuda()
    ],
                                   strict=False)
    traced_model = traced_model.eval()

    model = traced_model
    if optimize_config is 'disc':
        # Optimize with BladeDISC
        torch_config = torch_blade.config.Config()
        torch_config.enable_mlir_amp = True  # enable mix-precision
        torch_config.enable_force_to_cuda = True
        with torch.no_grad(), torch_config:
            # It will return an optimized TorchScript
            optimized_ts = torch_blade.optimize(traced_model.cuda(),
                                                allow_tracing=True,
                                                model_inputs=tuple([
                                                    input_features.cuda(),
                                                    attention_mask.cuda(),
                                                    decoder_input_ids.cuda()
                                                ]))

        output_pt_file = "s2t_disc_opt.pt"
        torch.jit.save(optimized_ts, output_pt_file)
        model = torch.jit.load(output_pt_file).eval()
    elif optimize_config is 'trt':
        cfg = torch_blade.Config.get_current_context_or_new().clone()
        cfg.optimization_pipeline = torch_blade.tensorrt.backend_name()
        cfg.customize_onnx_opset_version = 12
        cfg.enable_fp16 = True

        with cfg, torch_blade.logging.logger_level_context('DEBUG'):
            print("Go!")
            opt_model = torch_blade.optimize(traced_model.cuda(),
                                             False,
                                             model_inputs=tuple([
                                                 input_features.cuda(),
                                                 attention_mask.cuda(),
                                                 decoder_input_ids.cuda()
                                             ]))

        print("Optimize finish!")
        output_pt_file = "s2t_trt_opt.pt"
        torch.jit.save(opt_model, output_pt_file)
        print("Done!")
        model = torch.jit.load(output_pt_file).eval()

    # warmup
    for i in range(100):
        #output = optimized_ts(input_features.cuda())
        output = model(input_features.cuda(), attention_mask.cuda(),
                       decoder_input_ids.cuda())

    # normal measure
    tic = time.time()
    for i in range(100):
        output = model(input_features.cuda(), attention_mask.cuda(),
                       decoder_input_ids.cuda())
    rt_ms = (time.time() - tic) / 100
    print(f'average exec time: {rt_ms} s')

    cu_prof_start()
    for i in range(100):
        output = model(input_features.cuda(), attention_mask.cuda(),
                       decoder_input_ids.cuda())
    cu_prof_stop()

    if False:
        printTensorShapesInTuple(output)
        print(f"data for tensor {output[0].shape}")
        print(output[0])
        print(output[-1])

    if False:
        vanila = torch.jit.load(input_pt)
        output2 = vanila(input_features.cuda(), attention_mask.cuda(),
                         decoder_input_ids.cuda())
        print("\n\ntorch-script output tensor shapes:")
        printTensorShapesInTuple(output2)
        print(f"data for tensor {output2[0].shape}")
        print(output2[0])
        print(f"data for tensor {output2[-1].shape}")
        print(output2[-1])

    if False:
        print("Blade TRT:")
        blade_trt_out = blade_trt_opt(input_pt, 'blade-trt.pt', input_features,
                                      attention_mask, decoder_input_ids)
        printTensorShapesInTuple(blade_trt_out)
        print(blade_trt_out[0])
        print(blade_trt_out[-1])


if __name__ == '__main__':
    # `optimize_config` can be 'trt', 'disc' or None.
    run(optimize_config='disc')
