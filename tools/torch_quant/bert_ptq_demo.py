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
import logging
import time
from inspect import signature

import torch
import torch_blade
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from torch_quant.quantizer import Backend, Quantizer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.fx import HFTracer

logging.basicConfig(level=logging.DEBUG)

# TODO(litan.ls): use a smaller demo model
MODEL_ID = 'bert-base-cased-finetuned-mrpc'


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


def calibrate(model, dataloader, device):
    model.eval().to(device)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
            )


def evaluate(model, dataloader, device):
    metric = load("glue", "mrpc")
    n_steps = len(dataloader)
    progress_bar = tqdm(total=n_steps, desc='batches')
    model.eval().to(device)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
            )
        predictions = torch.argmax(outputs[0], dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)
    return metric.compute()


def gen_dummy_inputs(device, batch_size=1, seq_len=128):
    input_ids = torch.randint(
        100, 1000, (batch_size, seq_len), dtype=torch.int64, device=device)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    return (input_ids, attention_mask, token_type_ids)


def benchmark(model, device, n_warmup=10, n_repeat=20):
    inputs = gen_dummy_inputs(device)
    for _ in range(n_warmup):
        with torch.no_grad():
            model(*inputs)
    start = time.time()
    for _ in range(n_repeat):
        with torch.no_grad():
            model(*inputs)
    end = time.time()
    print(f'benchmark: avg={(end-start)/n_repeat*1000:.2f}ms')


def ptq(calib_dataloader, device):
    print('load model...')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torchscript=True)
    # model.config.problem_type is set in first forward calling, and is required for fx tracing
    # we call calibrate() here only as a dummy forward calling.
    calibrate(model, calib_dataloader, device)

    tracer = HFTracerWrapper(
        ["input_ids", "attention_mask", "token_type_ids"])
    quantizer = Quantizer(tracer=tracer, backend=Backend.DISC)
    print('calibrate model...')
    calib_model = quantizer.calib(model)
    calibrate(calib_model, calib_dataloader, device)

    quant_model = quantizer.quantize(model)
    dummy_inputs = gen_dummy_inputs(device)
    traced = torch.jit.trace(quant_model, dummy_inputs)
    torch.jit.save(traced, 'fake_quant.jit')
    blade_config = torch_blade.config.Config()
    blade_config.enable_int8 = True
    blade_config.disc_cluster_max_iter_count = 200
    with torch.no_grad(), blade_config:
        quant_model = torch_blade.optimize(traced, model_inputs=dummy_inputs)
        torch.jit.save(quant_model, 'blade_quant.jit')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit-model')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--ptq', action='store_true')
    args = parser.parse_args()

    print('load tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')

    def tokenize(x):
        return tokenizer(x['sentence1'], x['sentence2'], padding='max_length', truncation=True)
    print('load dataset...')
    dataset = load_dataset("glue", "mrpc")
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(['sentence1', 'sentence2', 'idx'])
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format('torch')

    device = torch.device('cpu')

    if args.ptq:
        calib_dataloader = DataLoader(dataset['train'].shuffle(
            seed=42).select(range(8)), batch_size=8)
        ptq(calib_dataloader, device)

    if args.jit_model:
        model = torch.jit.load(args.jit_model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, torchscript=True)
    model.to(device)

    if args.eval:
        test_dataloader = DataLoader(dataset['test'], batch_size=8)
        metric = evaluate(model, test_dataloader, device)
        print(f'eval metric: {metric}')

    if args.benchmark:
        benchmark(model, device)
