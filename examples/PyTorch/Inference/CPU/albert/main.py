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
import time
import torch
import timeit
import pandas as pd
import numpy as np
from contextlib import contextmanager
import itertools
import onnxruntime

import torch_blade
from torch_blade import optimize
from torch_blade.testing.common_utils import assert_almost_equal

os.environ["DISC_CPU_LARGE_CONCAT_NUM_OPERANDS"] = "4"
os.environ["DISC_CPU_ENABLE_WEIGHT_PRE_PACKING"] = "1"

def export_torch_to_onnx(model, output_file : str, inputs):
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

@contextmanager
def opt_disc_config():
    torch_config = torch_blade.config.Config()
    try:
        with torch_config:
             yield
    finally:
        pass

@torch.no_grad()
def do_optimize(model, inputs, backend):
    optimized_model = optimize(
        model,
        allow_tracing=True,
        model_inputs=tuple(inputs),
    )

    with open(f'model.code.py', 'w') as writer:
        writer.write(str(optimized_model.graph))
    return optimized_model

results = []
def printStats(backend, timings, batch_size=1):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = (
        "\n%s =================================\n"
        "batch size=%d, num iterations=%d\n"
        "  Median FPS: %.1f, mean: %.1f\n"
        "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
    ) % (
        backend,
        batch_size,
        steps,
        speed_med,
        speed_mean,
        time_med,
        time_mean,
        time_99th,
        time_std,
    )

    print(msg)
    meas = {
        "Backend": backend,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median(sec)": time_med,
        "Mean(sec)": time_mean,
        "99th_p": time_99th,
        "std_dev": time_std,
    }
    results.append(meas)

def benchmark_ort(model, inputs, backend, batch_size):
    device_name = 'cpu'
    export_torch_to_onnx(model, "albert-base-v2.onnx", inputs)
    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = 'albert-base-v2-opt.onnx'
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession('albert-base-v2.onnx', sess_options, providers=['CPUExecutionProvider'])
    ort_inputs = {
      'input_ids':  inputs[0].cpu().numpy(),
      'attention_mask': inputs[1].cpu().numpy(),
      'token_type_ids': inputs[2].cpu().numpy()
    }
    for i in range(100):
        ort_outputs = session.run(None, ort_inputs)
    timings = []
    start = time.time()
    for i in range(200):
        start_time = timeit.default_timer()
        ort_outputs = session.run(None, ort_inputs)
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)
    printStats("onnxruntime", timings, batch_size)

@torch.no_grad()
def benchmark(model, inp, backend, batch_size):
    for i in range(100):
        model(*inp)
    timings = []

    for i in range(200):
        start_time = timeit.default_timer()
        model(*inp)
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)

    printStats(backend, timings, batch_size)

def get_test_data(batch=1, seq_len=12):
    inp_ids = torch.zeros([batch, seq_len], dtype=torch.int)
    inp_mask0 = torch.zeros([batch, seq_len], dtype=torch.int)
    inp_mask1 = torch.zeros([batch, seq_len], dtype=torch.int)
    return (inp_ids.to(torch.int32), inp_mask0.to(torch.int32), inp_mask1.to(torch.int32))

def load_model():
    from transformers import AlbertModel
    model = AlbertModel.from_pretrained("albert-base-v2", torchscript=True)
    return model.eval()

def collect_tensors(data):
    if isinstance(data, torch.Tensor):
        return [data]
    elif isinstance(data, list):
        return list(itertools.chain(*[collect_tensors(d) for d in data]))
    elif isinstance(data, dict):
        sorted_pairs = sorted(data.items(), key=lambda x: x[0])
        sorted_list = [v for k, v in sorted_pairs]
        return collect_tensors(sorted_list)
    elif isinstance(data, tuple):
        return collect_tensors(list(data))
    else:
        return []

def check_results(results0, results1):
    results0 = collect_tensors(results0)
    results1 = collect_tensors(results1)

    try:
        assert_almost_equal(results0, results1, atol=2e-2, rtol=1e-5)
        print("Accuraccy check passed")
    except Exception as err:
        print(err)

if __name__ == '__main__':
    batch = 1
    seq_len = 24
    model = load_model()
    inputs = get_test_data(batch, seq_len)

    benchmark(model, inputs, "torch", batch)
    backend = "DISC"
    with opt_disc_config():
        opt_model = do_optimize(model, inputs, backend)
    benchmark(opt_model, inputs, backend, batch)
    benchmark_ort(model, inputs, "onnxruntime", batch)

    # Generate report
    print("Model Summary:")
    summary = pd.DataFrame(results)
    print(summary.to_markdown())

    output = model(*inputs)
    test_result = opt_model(*inputs)
    check_results(output, test_result)

