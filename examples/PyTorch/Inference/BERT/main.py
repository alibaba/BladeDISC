#!/usr/bin/python
import os

os.environ["TORCH_BLADE_DEBUG_LOG"] = "off"
# Enable stitch fusion optimization.
os.environ["DISC_ENABLE_STITCH"] = "true"
os.environ["DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE"] = "true"


import time
import ctypes
import numpy as np
import torch
import torch.onnx
from transformers import BertModel, BertConfig, TFBertModel

import torch_blade

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
            return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                    inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values,
                    use_cache, output_attentions, output_hidden_states, return_dict)


def get_torch_test_data(batch : int = 1, seq_len : int = 128, cuda : bool = False):
    inp0 = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inp1 = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inp2 = torch.zeros([batch, seq_len], dtype=torch.int).long()
    inputs = [inp0, inp1, inp2]
    if cuda:
        inputs = [inp.cuda() for inp in inputs]
    return inputs


def get_torch_bert_model(num_hidden_layers : int = 12, num_attention_heads : int = 12,
                         hidden_size : int = 768, amp : bool = False):
    configuration = BertConfig() # bert-base-uncased style configuration
    configuration.num_hidden_layers = num_hidden_layers
    configuration.num_attention_heads = num_attention_heads
    configuration.hidden_size = hidden_size

    model = BertModelAMP(configuration).cuda().eval() if amp else BertModel(configuration).cuda().eval()
    return model


def get_torch_bert_large_model(amp : bool = False):
    model = get_torch_bert_model(num_hidden_layers = 24, num_attention_heads = 16,
                         hidden_size = 1024, amp = amp)
    return model


def save_torch_jit(model, file_name : str):
    inputs = get_torch_test_data(cuda = True)
    traced_model = torch.jit.trace(model.cuda(), inputs, strict=False)
    torch.jit.save(traced_model, file_name)


def disc_optimize_torch(in_file : str, out_file : str):
    torch_config = torch_blade.config.Config()
    torch_config.enable_mlir_amp = False # disable mix-precision

    inputs = get_torch_test_data(cuda = True)
    model = torch.jit.load(in_file).cuda().eval()

    torch._C._jit_pass_inline(model._c.forward.graph)
    torch._C._jit_pass_remove_dropout(model._c)

    with torch.no_grad(), torch_config:
      # BladeDISC torch_blade optimize will return an optimized TorchScript
      optimized_ts = torch_blade.optimize(model, allow_tracing=True, model_inputs=tuple(inputs))
    torch.jit.save(optimized_ts, out_file)


def run_torch_script(model, batch : int = 1 , seq_len : int = 64, in_data = None):
    if in_data is None: 
        inputs = get_torch_test_data(batch, seq_len, cuda = True)
    else:
        inputs = in_data

    # warmup
    for i in range(20):
        model(*tuple(inputs))
    iters = 100
    tic = time.time()
    for i in range(iters):
        model(*tuple(inputs))
    avg_time = (time.time() - tic) / iters
    print("average time in {} iterations: {} seconds".format(iters, avg_time))

    exec_time = []
    for i in range(50):
        tic = time.time()
        model(*tuple(inputs))
        delta = time.time() - tic
        exec_time.append(delta)
    print("medium execution time of 50 iterations: {} seconds.".format(np.median(exec_time)))

    # profile start
    _cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
    res = _cudart.cudaProfilerStart()
    model(*tuple(inputs))
    res = _cudart.cudaProfilerStop()
    # profile end


def export_torch_to_onnx(model, output_file : str):
    inputs = get_torch_test_data(cuda = True)
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    output_names = ['last_hidden_state', 'pooler_output']
    dynamic_axes={'input_ids' : {0 : 'batch_size', 1 : 'seq_length'},
                  'attention_mask' : {0 : 'batch_size', 1 : 'seq_length'},
                  'token_type_ids' : {0 : 'batch_size', 1 : 'seq_length'},
                  'last_hidden_state' : {0 : 'batch_size', 1 : 'seq_length'},
                  'pooler_output' : {0 : 'batch_size'}
                  }
    torch.onnx.export(model, tuple(inputs), output_file, opset_version=10,
                      input_names = input_names, output_names = output_names,
                      dynamic_axes = dynamic_axes)


def save_trt_engine(onnx_file : str, trt_engine_name : str, opt_shape : list,
        min_shape : list = None, max_shape : list = None, fp16 : bool = False, int8 : bool = False):
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


def run_trt_engine(trt_engine_name : str, batch : int = 1, seq_len : int = 64):
    import volksdep.converters
    trt_model = volksdep.converters.load(trt_engine_name)

    inputs = get_torch_test_data(batch, seq_len, cuda = True)

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
    _cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
    res = _cudart.cudaProfilerStart()
    trt_model(inputs)
    res = _cudart.cudaProfilerStop()
    # profile end


def run(optimize_config : str = None):
    batch = 1
    seq = 64
    bert_large_amp = get_torch_bert_large_model(amp = True)
    inputs = get_torch_test_data(batch = batch, seq_len = seq, cuda = True)

    model = bert_large_amp
    if optimize_config is 'disc':
        # BladeDISC optimization.
        save_torch_jit(bert_large_amp, 'bert_large_amp.pt')
        disc_optimize_torch('bert_large_amp.pt', 'bert_large_amp.opt.pt')
        model = torch.jit.load(script_file).cuda().eval()
    run_torch(model, batch = batch, seq_len = seq, in_data = inputs)


    # TRT Optimization. To be removed in the final release.
    # To use TRT amp rather than Torch AMP.
    bert_large = get_torch_bert_large_model(amp = False)
    export_torch_to_onnx(bert_large, 'bert_large.onnx')
    save_trt_engine('bert_large.onnx', f'bert_large.fp16.{batch}x{seq}.trt', [batch, seq], fp16 = True)
    run_trt_engine(f'bert_large.fp16.{batch}x{seq}.trt', batch = batch, seq_len = seq)


if __name__ == '__main__':
    # `optimize_config` can be 'disc' or None.
    run(optimize_config = 'disc')
