#!/usr/bin/python3
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

import argparse
import os

# Enable stitch fusion optimization.
os.environ["DISC_ENABLE_STITCH"] = "true"

import time
import ctypes
import numpy as np
import torch
import torch.onnx
from transformers import BertModel, BertConfig, TFBertModel

import torch_blade
import torch_blade.tensorrt
import torch_blade.utils as utils

# Tools for profiling, to be removed in the final release.
_cudart = ctypes.CDLL('libcudart.so')


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)


class BertModelAMP(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        with torch.cuda.amp.autocast():
            super().__init__(config, add_pooling_layer)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        with torch.cuda.amp.autocast():
            return super().forward(input_ids, attention_mask, token_type_ids,
                                   position_ids, head_mask, inputs_embeds,
                                   encoder_hidden_states,
                                   encoder_attention_mask, past_key_values,
                                   use_cache, output_attentions,
                                   output_hidden_states, return_dict)


def get_torch_test_data(batch: int = 1,
                        seq_len: int = 128,
                        cuda: bool = False):
    inp0 = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inp1 = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inp2 = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inputs = [inp0, inp1, inp2]
    if cuda:
        inputs = [inp.cuda() for inp in inputs]
    return inputs


def get_torch_bert_model(num_hidden_layers: int = 12,
                         num_attention_heads: int = 12,
                         hidden_size: int = 768,
                         amp: bool = False):
    configuration = BertConfig()  # bert-base-uncased style configuration
    configuration.num_hidden_layers = num_hidden_layers
    configuration.num_attention_heads = num_attention_heads
    configuration.hidden_size = hidden_size

    model = BertModelAMP(configuration).cuda().eval() if amp else BertModel(
        configuration).cuda().eval()
    return model


def get_torch_bert_large_model(amp: bool = False):
    model = get_torch_bert_model(num_hidden_layers=24,
                                 num_attention_heads=16,
                                 hidden_size=1024,
                                 amp=amp)
    return model


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


def export_torch_to_onnx(model, inputs, output_file: str):
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    output_names = ['last_hidden_state', 'pooler_output']
    dynamic_axes = {
        'input_ids': {
            0: 'batch_size',
            1: 'seq_length'
        },
        'attention_mask': {
            0: 'batch_size',
            1: 'seq_length'
        },
        'token_type_ids': {
            0: 'batch_size',
            1: 'seq_length'
        },
        'last_hidden_state': {
            0: 'batch_size',
            1: 'seq_length'
        },
        'pooler_output': {
            0: 'batch_size'
        }
    }
    torch.onnx.export(model,
                      tuple(inputs),
                      output_file,
                      opset_version=10,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)


def save_trt_engine(onnx_file: str,
                    trt_engine_name: str,
                    opt_shape: list,
                    min_shape: list = None,
                    max_shape: list = None,
                    fp16: bool = False,
                    int8: bool = False):
    cmd = f'trtexec --onnx={onnx_file} --saveEngine={trt_engine_name} --workspace=256'
    type_str = 'fp16' if fp16 else 'fp32'
    if fp16:
        cmd += ' --fp16'
    if int8:
        cmd += ' --int8'
    shape_str = f'{opt_shape[0]}x{opt_shape[1]}'
    cmd += f' --optShapes=input_ids:{shape_str},attention_mask:{shape_str},token_type_ids:{shape_str}'
    if min_shape:
        shape_str = f'{min_shape[0]}x{min_shape[1]}'
        cmd += f' --minShapes=input_ids:{shape_str},attention_mask:{shape_str},token_type_ids:{shape_str}'
    if max_shape:
        shape_str = f'{max_shape[0]}x{max_shape[1]}'
        cmd += f' --maxShapes=input_ids:{shape_str},attention_mask:{shape_str},token_type_ids:{shape_str}'
    print(cmd)
    out = os.popen(cmd).read().split('\n')
    for line in out:
        print(line)


def run_trt_engine(trt_engine_name: str, inputs):
    import volksdep.converters
    trt_model = volksdep.converters.load(trt_engine_name)

    # warmup
    for i in range(20):
        trt_model(inputs)
    iters = 100
    tic = time.time()
    for i in range(iters):
        trt_model(inputs)
    avg_time = (time.time() - tic) / iters
    print("average time in {} iterations: {} seconds".format(iters, avg_time))

    # profile start
    _cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
    res = _cudart.cudaProfilerStart()
    trt_model(inputs)
    res = _cudart.cudaProfilerStop()
    # profile end


def blade_trt_optimize(model, inputs, fp16: bool, is_static: bool,
                       out_file: str):
    cfg = torch_blade.Config.get_current_context_or_new().clone()
    cfg.optimization_pipeline = torch_blade.tensorrt.backend_name()
    cfg.customize_onnx_opset_version = 12
    cfg.enable_fp16 = fp16

    traced_model = torch.jit.trace(model.cuda().eval(), inputs,
                                   strict=False).cuda().eval()

    with cfg, torch_blade.logging.logger_level_context('INFO'):
        if is_static is True:
            opt_model = torch_blade.optimization._static_optimize(
                traced_model, False, model_inputs=tuple(inputs))
        else:
            opt_model = torch_blade.optimize(traced_model,
                                             False,
                                             model_inputs=tuple(inputs))
    torch.jit.save(opt_model, out_file)


def run():
    parser = argparse.ArgumentParser(prog = 'BladeDISC Bert example')
    parser.add_argument('--disc-only', help = "Run BladeDISC only",
                    action='store_true')  # on/off flag
    args = parser.parse_args()

    batch = 1
    seq = 64
    bert_large = get_torch_bert_large_model(amp=False)
    bert_large_amp = get_torch_bert_large_model(amp=True)
    inputs = get_torch_test_data(batch=batch, seq_len=seq, cuda=True)

    # Run naive torch.
    print("Naive PyTorch.")
    model = bert_large_amp
    evaluate_torch(model, inputs)

    if utils.torch_version_number() >= utils.parse_version("1.14.0"):
        print("BladeDISC PyTorch 2.0 Optimization.")
        evaluate_torch(torch.compile(bert_large_amp, backend="disc"), inputs)

    # Run BladeDISC optimization.
    print("BladeDISC Optimization.")
    disc_optimize(bert_large_amp, inputs, 'bert_large_amp.disc.pt')
    model = torch.jit.load('bert_large_amp.disc.pt').cuda().eval()
    evaluate_torch(model, inputs)

    if args.disc_only:
        return

    # Run TensorRT with `trtexec`. Static shape optimization.
    print("Official TensorRT Static Optimization.")
    export_torch_to_onnx(bert_large, inputs, 'bert_large.onnx')
    save_trt_engine('bert_large.onnx',
                    f'bert_large.fp16.{batch}x{seq}.trt', [batch, seq],
                    fp16=True)
    run_trt_engine(f'bert_large.fp16.{batch}x{seq}.trt', inputs)

    # Run TorchBlade-TensorRT optimization.
    print("TorchBlade-TensorRT Optimization.")
    blade_trt_optimize(bert_large, inputs, True, True, 'bert_large_amp.trt.pt')
    model = torch.jit.load('bert_large_amp.trt.pt').cuda().eval()
    evaluate_torch(model, inputs)


if __name__ == '__main__':
    run()
