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

import sys
import os
import onnx
import torch
import torch_blade

import foldacc.optimization.distributed.utils 

model_path = sys.argv[1]
data_path = sys.argv[2]
save_path = sys.argv[3]
device = sys.argv[4]

torch.cuda.set_device(int(device))

model = torch.jit.load(model_path)
inputs = torch.load(data_path)

model = torch_blade.optimize(model, allow_tracing=True, model_inputs=inputs)

torch.jit.save(model, save_path)