#!/usr/bin/python
import os

os.environ["TORCH_BLADE_DEBUG_LOG"] = "off"
#os.environ["TRT_LOG_LEVEL"] = "INFO"
os.environ["DISC_ENABLE_STITCH"] = "true"
os.environ["DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE"] = "true"
#os.environ["DEBUG_MAX_FUSION_NUMBERS"] = "196"
#os.environ["TF_CPP_VMODULE"] = "disc_compiler=1"
#os.environ["TAO_CPP_VMODULE"] = "disc_compiler=1"

import time
import ctypes
import numpy as np
import torch
import torch.onnx
from transformers import BertModel, BertConfig, TFBertModel
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants, constants
from tensorflow.core.protobuf import meta_graph_pb2

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


def print_torch_jit(model):
    inputs = get_torch_test_data(cuda = True)
    traced_model = torch.jit.trace(model.cuda(), inputs, strict=False)
    # print(traced_model.graph)


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


def run_torch_script(script_file : str, batch : int = 1 , seq_len : int = 64, in_data = None):
    model = torch.jit.load(script_file).cuda().eval()

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

    # print(model(*tuple(inputs)))


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


def print_onnx(model_file : str):
    import onnx

    # Load the ONNX model
    model = onnx.load(model_file)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def get_torch_bert_large_model(amp : bool = False):
    model = get_torch_bert_model(num_hidden_layers = 24, num_attention_heads = 16,
                         hidden_size = 1024, amp = amp)
    return model


def get_torch_bert_base_model(amp : bool = False):
    model = get_torch_bert_model(num_hidden_layers = 12, num_attention_heads = 12,
                         hidden_size = 768, amp = amp)
    return model


def run_torch_model(model, batch : int = 1, seq_len : int = 64):
    inputs = get_torch_test_data(batch, seq_len, cuda = True)
    model(*tuple(inputs))


def get_tf_test_data(batch : int = 1, seq_len : int = 64, cuda : bool = False):
    inp0 = np.zeros([batch, seq_len], dtype=int)
    inp1 = np.zeros([batch, seq_len], dtype=int)
    inp2 = np.zeros([batch, seq_len], dtype=int)
    inputs = [inp0, inp1, inp2]
    return input


def get_tf_bert_model(num_hidden_layers : int = 12, num_attention_heads : int = 12,
                      hidden_size : int = 768, amp : bool = False):
    # Initializing a BERT bert-base-uncased style configuration
    configuration = BertConfig()
    configuration.num_hidden_layers = num_hidden_layers
    configuration.num_attention_heads = num_attention_heads
    configuration.hidden_size = hidden_size

    model = TFBertModel(configuration)

    return model


def get_tf_bert_large_model(amp : bool = False):
    model = get_tf_bert_model(num_hidden_layers = 24, num_attention_heads = 16,
                         hidden_size = 1024, amp = amp)
    return model


def get_tf_bert_base_model(amp : bool = False):
    model = get_tf_bert_model(num_hidden_layers = 12, num_attention_heads = 12,
                         hidden_size = 768, amp = amp)
    return model

def load_tf_bert_model(path : str):
    return tf.saved_model.load(path)


def execute_tf_opt():
  data = get_tf_test_data(batch = 1, seq_len = 64)
  model = get_tf_bert_large_model()
  tf.saved_model.save(model, 'saved')

  test_data = {
    "serving_default_attention_mask:0" : np.zeros([1, 64], dtype = int),
    "serving_default_input_ids:0" : np.zeros([1, 64], dtype = int),
    "serving_default_token_type_ids:0" : np.zeros([1, 64], dtype = int),
  }

  session_config = tf.ConfigProto()
  session_config.log_device_placement = True
  session_config.allow_soft_placement = True
  #session_config.intra_op_parallelism_threads=int(16/process_num)
  #session_config.inter_op_parallelism_threads=int(16/process_num)
  session_config.gpu_options.allow_growth = True
  session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  session_config.graph_options.rewrite_options.auto_mixed_precision = 1

  with tf.Session(config=session_config) as sess:
    meta_graph_def = tf.saved_model.loader.load(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    './saved'
    )
    signature_def = meta_graph_def.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    inputs = signature_def.inputs
    outputs = signature_def.outputs

    fetch_dict = {}
    for name, tensor in outputs.items():
      fetch_dict[name] = tf.get_default_graph().get_tensor_by_name(tensor.name)

    # warmup 
    for i in range(50):
      outs = sess.run(fetch_dict, feed_dict=test_data)

    step_to_profile = 80
    all_times = []
    for i in range(100):
      s = time.time()
      # if i == step_to_profile:
      #   cu_prof_start()
      outs = sess.run(fetch_dict, feed_dict=test_data)
      e = time.time()
      # print(f'outs = {outs}')
      # if i == step_to_profile:
      #   cu_prof_stop()
      # print(f'step #{i}: {e - s} s')
      all_times.append(e - s)
    print(f'mean: {np.mean(all_times)}, median: {np.median(all_times)}, max: {np.max(all_times)}, min: {np.min(all_times)}')


def execute_torch_opt():
    batch = 1
    seq = 64
    bert_large_amp = get_torch_bert_large_model(amp = True)

    # BladeDISC optimization.
    save_torch_jit(bert_large_amp, 'bert_large_amp.pt')
    inputs = get_torch_test_data(batch = batch, seq_len = seq, cuda = True)
    disc_optimize_torch('bert_large_amp.pt', 'bert_large_amp.stitch.pt')
    run_torch_script('bert_large_amp.stitch.pt', batch = batch, seq_len = seq, in_data = inputs)

    # TRT Optimization.
    # To use TRT amp rather than Torch AMP.
    bert_large = get_torch_bert_large_model(amp = False)
    export_torch_to_onnx(bert_large, 'bert_large.onnx')
    save_trt_engine('bert_large.onnx', f'bert_large.fp16.{batch}x{seq}.trt', [batch, seq], fp16 = True)
    run_trt_engine(f'bert_large.fp16.{batch}x{seq}.trt', batch = batch, seq_len = seq)
    # int8 does not work well.
    #save_trt_engine('bert_large.onnx', f'bert_large.int8.{batch}x{seq}.trt', [batch, seq], fp16 = False, int8 = True)


if __name__ == '__main__':
  execute_torch_opt()
  #execute_tf_opt()
