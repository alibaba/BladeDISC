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

import logging
from inspect import signature

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torch_quant.quantizer import Backend, Quantizer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.fx import HFTracer

logging.basicConfig(level=logging.DEBUG)


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
            model(**batch)


def evaluate(model, dataloader, device):
    metric = load_metric("glue", "mrpc")
    n_steps = len(dataloader)
    progress_bar = tqdm(total=n_steps, desc='batches')
    model.eval().to(device)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.argmax(outputs["logits"], dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)
    return metric.compute()


if __name__ == "__main__":
    print('load tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")

    def tokenize(x):
        return tokenizer(x['sentence1'], x['sentence2'], padding='max_length', truncation=True)
    print('load dataset...')
    dataset = load_dataset("glue", "mrpc")
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(['sentence1', 'sentence2', 'idx'])
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format('torch')
    calib_dataloader = DataLoader(dataset['train'].shuffle(
        seed=42).select(range(8)), batch_size=8)
    test_dataloader = DataLoader(dataset['test'], batch_size=8)
    cpu = torch.device('cpu')

    print('load model...')
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased-finetuned-mrpc")
    # model.config.problem_type is set in first forward calling, and is required for fx tracing
    print('test original model...')
    metric = evaluate(model, test_dataloader, cpu)
    print(f'original test metric: {metric}')

    print('prepare ptq model...')
    tracer = HFTracerWrapper(
        ["input_ids", "attention_mask", "token_type_ids", "labels"])
    quantizer = Quantizer(tracer=tracer, backend=Backend.FBGEMM)
    calib_model = quantizer.calib(model)
    calibrate(calib_model, calib_dataloader, cpu)

    quant_model = quantizer.quantize(model)
    metric = evaluate(quant_model, test_dataloader, cpu)
    print(f'quant test metric: {metric}')
