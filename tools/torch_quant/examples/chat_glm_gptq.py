# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from torch import nn
from torch_quant.experiment import GPTQuantizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

logging.basicConfig(level=logging.DEBUG)

from transformers.dynamic_module_utils import get_class_from_dynamic_module


def do_inference_with_fixed_seed(model, tokenizer, prompt):
    set_seed(42)
    response, _ = model.chat(tokenizer, prompt, history=[])
    return response


prompt = "晚上睡不着应该怎么办"

# If local mode is used, change it to the folder that contains model files
target_model = "THUDM/chatglm-6b"

# The basic block within which all layers are calibrated together
target_block = get_class_from_dynamic_module(target_model, "modeling_chatglm.py", "GLMBlock")

tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
model = AutoModel.from_pretrained(target_model, trust_remote_code=True, resume_download=True).half().cuda()

# Get the output of the original model
response = do_inference_with_fixed_seed(model, tokenizer, prompt)
print(response)

# Get the GPTQuantizer, the block means that the GLMBlock is calibrated one-by-one
# and the last lm_head (of type nn.Linear) is calibrated alone
quantizer = GPTQuantizer(block=[target_block, nn.Linear])

# prepare the model for the quantization process
calib_model = quantizer.calib(model)

# Since we do not get the graph of the model (e.g. torchscript, fx graph), we must
# do inference on the model once and record the block order
with quantizer.record_order():
    do_inference_with_fixed_seed(model, tokenizer, prompt)

# Do calibration on the model. In each iter, one block will be quantized using GPTQ and
# you can use other prompts.
for i in tqdm(range(quantizer.calibration_iters)):
    with quantizer.start_calib_iter(i):
        response, history = model.chat(tokenizer, prompt, history=[])

# Get the result of the weight fake-quantized model
response = do_inference_with_fixed_seed(model, tokenizer, prompt)
print(response)
