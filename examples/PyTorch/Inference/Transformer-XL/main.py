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

# set TORCH_BLADE_DEBUG_LOG=on
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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = TransfoXLModel.from_pretrained('transfo-xl-wt103', torchscript=True).to(device)
#tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)
inputs = torch.tensor([[14049, 2, 617, 3225, 23, 16072]]).long().to(device)

model.eval()
traced_model = torch.jit.trace(model, inputs, strict = False)

#last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
#
torch_config = torch_blade.config.Config()
torch_config.enable_mlir_amp = False # disable mix-precision
with torch.no_grad(), torch_config:
  # BladeDISC torch_blade optimize will return an optimized TorchScript
  optimized_ts = torch_blade.optimize(traced_model, allow_tracing=True, model_inputs=inputs)

# The optimized module could be saved as a TorchScript
torch.jit.save(optimized_ts, "opt.disc.stitch.pt")

optimized_ts = torch.jit.load("opt.disc.stitch.pt")

cu_prof_start()
outputs = optimized_ts(inputs)
cu_prof_stop()
print(outputs)
