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