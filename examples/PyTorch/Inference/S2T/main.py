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
os.environ["DISC_ENABLE_STITCH"] = "true"
os.environ["DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE"] = "true"

import torch
import time

import volksdep.converters
import numpy as np

import torch_blade

import ctypes
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
        self.model=original_model

   def forward(
           self,
           input_features = None,
           attention_mask = None,
           decoder_input_ids = None,
           **kwargs,
           ):
        #generated_ids = self.model.generate(inputs)
        result = self.model(input_features, attention_mask, decoder_input_ids, **kwargs)
        keys = result.keys()
        return tuple([result[key] for key in keys])


def run_trt_engine(trt_engine_name):
    trt_model = volksdep.converters.load(trt_engine_name)

    batch_size = 1
    sequence_length = 584
    feature_size = 80
    input_features = torch.rand(batch_size, sequence_length, feature_size)
    attention_mask = torch.rand(batch_size, sequence_length).long()
    decoder_input_ids = torch.tensor([[2]] * batch_size)
    inputs = tuple([input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda()])
    #inputs = get_torch_test_data(batch, seq_len, cuda = True)

    # warmup
    for i in range(20):
        trt_output = trt_model(inputs)
    iters = 100
    tic = time.time()
    for i in range(iters):
        trt_model(inputs)
    avg_time = (time.time() - tic) / iters
    print("average time in {} iterations: {} seconds".format(iters, avg_time))

    exec_time = []
    for i in range(50):
        tic = time.time()
        trt_model(inputs)
        delta = time.time() - tic
        exec_time.append(delta)
    print("medium execution time of 50 iterations: {} seconds.".format(np.median(exec_time)))

    # profile start
    cu_prof_start()
    for i in range(100):
        trt_model(inputs)
    cu_prof_stop()
    # profile end


def export_torch_to_onnx(input_pt : str, output_file : str):
    model = torch.jit.load(input_pt)
    model = model.eval().cuda()

    batch_size = 1
    sequence_length = 584
    feature_size = 80
    input_features = torch.rand(batch_size, sequence_length, feature_size)
    attention_mask = torch.rand(batch_size, sequence_length).long()
    decoder_input_ids = torch.tensor([[2]] * batch_size)
    example_outputs = model(input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda())
    test_data = (input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda())

    input_names = ['input_features', 'attention_mask', 'decoder_input_ids']
    output_names = ['logits', 'past_key_values', 'encoder_last_hidden_state']

    dynamic_axes={'input_features' : {0 : 'batch_size', 1 : 'sequence_length'},
                  'attention_mask' : {0 : 'batch_size', 1 : 'sequence_length'},
                  'decoder_input_ids' : {0 : 'batch_size'},
                  }
    # Failed in torch 1.7.1. Success in torch 1.8.1
    torch.onnx.export(model, test_data, output_file, opset_version=12,
                      input_names = input_names, output_names = output_names,
                      dynamic_axes = dynamic_axes, example_outputs = example_outputs)


def traceS2TAndSave(filename : str):
    from transformers import Speech2TextForConditionalGeneration
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    model = model.eval()
    batch_size = 1
    sequence_length = 584
    feature_size = 80
    input_features = torch.rand(batch_size, sequence_length, feature_size)
    attention_mask = torch.rand(batch_size, sequence_length).long()
    decoder_input_ids = torch.tensor([[2]] * batch_size)
    #inputs = torch.rand(batch_size, sequence_length, feature_size)
    # It turns out that batch dim is static and seq_len dim is dynamic after tracing.
    traced_model = torch.jit.trace(wrapper(model.cuda()), [input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda()], strict = False)
    #traced_model = torch.jit.trace(wrapper(model.cuda()), inputs.cuda(), strict = False)
    torch.jit.save(traced_model, filename)


def DISCOptimize(input_pt : str, output_pt_to_save : str):
    traced_model = torch.jit.load(input_pt)
    traced_model = traced_model.eval()

    batch_size = 1
    #sequence_length = 64
    sequence_length = 584
    #sequence_length = 1024 
    feature_size = 80
    input_features = torch.rand(batch_size, sequence_length, feature_size)
    attention_mask = torch.rand(batch_size, sequence_length).long()
    decoder_input_ids = torch.tensor([[2]] * batch_size)

    # torch_config = torch_blade.config.Config()
    # torch_config.enable_mlir_amp = True # disable mix-precision
    # torch_config.enable_force_to_cuda = True
    # with torch.no_grad(), torch_config:
    #     # BladeDISC torch_blade optimize will return an optimized TorchScript
    #     optimized_ts = torch_blade.optimize(traced_model.cuda(), allow_tracing=True,
    #                                         model_inputs=tuple([input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda()]))

    # torch.jit.save(optimized_ts, output_pt_to_save)
    optimized_ts = torch.jit.load(output_pt_to_save)
    optimized_ts = optimized_ts.cuda()

    # warmup
    for i in range(100):
        #output = optimized_ts(input_features.cuda())
        output = optimized_ts(input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda())

    # normal measure
    tic = time.time()
    for i in range(100):
        output = optimized_ts(input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda())
    rt_ms = (time.time() - tic) / 100
    print(f'exec time: {rt_ms} s')

    cu_prof_start()
    for i in range(100):
        output = optimized_ts(input_features.cuda(), attention_mask.cuda(), decoder_input_ids.cuda())
    cu_prof_stop()


    #print(output)


if __name__ == '__main__':
    # A traced model is already maintained at:
    # http://zhengzhen.oss-cn-hangzhou-zmf.aliyuncs.com/model-data/Speech/PyTorch-S2T-Fairseq/traced-model.pt
    #export_torch_to_onnx('one-step-forward.pt', 'one-step-forward.onnx')
    #traceS2TAndSave('one-step-forward.pt')

    #DISCOptimize('one-step-forward.pt', 'disc-opt.one-step-forward.pt')
    #DISCOptimize('one-step-forward.pt', 'disc-opt.stitch.one-step-forward.pt')
    DISCOptimize('one-step-forward.pt', 'disc-opt.stitch.tlp.one-step-forward.pt')

    #run_trt_engine('one-step-forward.trt')
