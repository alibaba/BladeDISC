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

import contextlib
import os
import time
from inspect import signature

import torch
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from torch_blade.config import Config
from torch_blade.optimization import optimize
from torch_quant.quantizer import Backend, Quantizer
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed
)
from transformers.utils.fx import HFTracer


@contextlib.contextmanager
def set_env(**environ):
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@torch.no_grad()
def benchmark(model, inp):
    warmup = 5
    bench = 10
    for _ in range(warmup):
        model(*inp)

    start = time.time()
    for _ in range(bench):
        model(*inp)
    end = time.time()
    ms = (end - start) * 1000 / bench
    print(f"avg ms: {ms}")


@torch.no_grad()
def evaluate(model, dataloader, device):
    metric = load("glue", "mrpc")
    model.eval().to(device)
    for batch in tqdm(dataloader, desc='Running evaluation'):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        )
        output = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        predictions = torch.argmax(output, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print("************************")
    print("evaluation result: ")
    for k, v in metric.compute().items():
        print(f"  {k: <{10}} = {round(v, 4):>{8}}")
    print("************************")


@torch.no_grad()
def calibrate(model, dataloader, device):
    calib_step = 10
    model.eval().to(device)
    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        )
        if idx == calib_step:
            return


def get_dummy_input():
    batch = 6
    seq_len = 128
    input_ids = torch.randint(
        100, 1000, (batch, seq_len), dtype=torch.int64, device="cpu")
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    dummy_input = (input_ids, attention_mask, token_type_ids)
    return dummy_input


set_seed(0)
MODEL_ID = "M-FAC/bert-mini-finetuned-mrpc"
TASK_NAME = "mrpc"
REVISION = "main"
device = "cpu"

print('load dataset...')
raw_datasets = load_dataset("glue", TASK_NAME)
print('prepare bert-mini model...')
config = AutoConfig.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    torchscript=True
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    revision=REVISION,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=config,
    revision=REVISION,
)


def preprocess_function(x):
    return tokenizer(x['sentence1'], x['sentence2'], padding='max_length', max_length=128, truncation=True)


raw_datasets = raw_datasets.map(preprocess_function, batched=True)
eval_dataset = raw_datasets["validation"]
eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=default_data_collator)
# Firstly, we evaluate the accuracy of the origin fp32 model.
# You will get the result like:
# ************************
# evaluation result:
#   accuracy   =   0.8113
#   f1         =   0.8651
# ************************
evaluate(model, eval_dataloader, device)

# optimize the model using DISC on fp32 precision
dummy_input = get_dummy_input()
fp32_torchscript_model = torch.jit.trace(model, dummy_input)
fp32_disc_model = optimize(fp32_torchscript_model, True, dummy_input)
# Then, we evaluate the accuracy of the DISC optimized fp32 model.
# You will get the result like:
# ************************
# evaluation result:
#   accuracy   =   0.8113
#   f1         =   0.8651
# ************************
evaluate(fp32_disc_model, eval_dataloader, device)


# optimize the model using DISC on INT8 precision
class HFTracerWrapper(HFTracer):
    def __init__(self, input_names, **kwargs):
        super().__init__(**kwargs)
        self.input_names = input_names

    def trace(self, root, **kwargs):
        sig = signature(root.forward)
        concrete_args = {
            p.name: p.default for p in sig.parameters.values() if p.name not in self.input_names
        }
        return super().trace(root, concrete_args=concrete_args, **kwargs)


tracer = HFTracerWrapper(["input_ids", "attention_mask", "token_type_ids"])
quantizer = Quantizer(tracer=tracer, backend=Backend.DISC)
# Convert the nn.Module to fx.Module and modify the inference
# graph to meet the needs of convert it the int8 model.
calib_model = quantizer.calib(model)
# Do calibration on the fx.Module and collect the quantization information.
calibrate(calib_model, eval_dataloader, device)

# Convert the fx.Module to torchscript with fake-quant on it.
fake_quant_model = quantizer.quantize(model)
traced_model = torch.jit.trace(fake_quant_model, dummy_input)

# Convert the torchscript with fake-quant to the real quantized DISC optimized model.
# Get the pdll files for optimization, which define the quantization patterns.
level = 3
par_dir = (os.path.pardir,) * level
disc_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *par_dir))
pdll_files = os.path.join(disc_root_path, "pytorch_blade/tests/disc/pdl/pdll_files/")
disc_torch_pdll_files = [
    os.path.join(pdll_files, "common/fake_quant.pdll"),
    os.path.join(pdll_files, "cpu/dequant_gemm_quant.pdll")
]
env_var = {}
env_var["DISC_TORCH_PDL_FILES"] = ",".join(disc_torch_pdll_files)
# Set some config to enable quantization optimization
cfg = Config()
cfg.enable_int8 = True
cfg.disc_cluster_max_iter_count = 100
# optimize the model through the DISC optimization interface
with set_env(**env_var), cfg:
    int8_disc_model = optimize(traced_model, True, dummy_input)
    print(int8_disc_model.inlined_graph)
# evaluate the real quantized DISC model
evaluate(int8_disc_model, eval_dataloader, device)

# benchmark the origin model, DISC fp32 model and disc int8 model
benchmark(model, dummy_input)
benchmark(fp32_disc_model, dummy_input)
benchmark(int8_disc_model, dummy_input)
