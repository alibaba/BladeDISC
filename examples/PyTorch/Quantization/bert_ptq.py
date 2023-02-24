import contextlib
import os
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

set_seed(0)
MODEL_ID = "M-FAC/bert-mini-finetuned-mrpc"
TASK_NAME = "mrpc"
REVISION = "main"

print('load dataset...')
raw_datasets = load_dataset("glue", TASK_NAME)
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
evaluate(model, eval_dataloader, "cpu")


# quantization
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

input_ids = torch.randint(
    100, 1000, (1, 128), dtype=torch.int64, device="cpu")
attention_mask = torch.ones_like(input_ids)
token_type_ids = torch.zeros_like(input_ids)
dummy_input = (input_ids, attention_mask, token_type_ids)
traced_model = torch.jit.trace(model, dummy_input)
torch.jit.save(traced_model, "bert_mini.fp32.torchscript")

opt_model = optimize(traced_model, True, dummy_input)
print(opt_model.inlined_graph)
torch.jit.save(opt_model, "bert_mini.fp32.disc.torchscript")


quantizer = Quantizer(tracer=tracer, backend=Backend.DISC)
calib_model = quantizer.calib(model)
calibrate(calib_model, eval_dataloader, "cpu")

fake_quant_model = quantizer.quantize(model)
evaluate(fake_quant_model, eval_dataloader, "cpu")

traced_model = torch.jit.trace(fake_quant_model, dummy_input)

cfg = Config()
cfg.enable_int8 = True
cfg.disc_cluster_max_iter_count = 100
DISC_TORCH_PDL_FILES = [
    "/workspace/codes/BladeDISC/pytorch_blade/tests/disc/pdl/pdll_files/common/fake_quant.pdll",
    "/workspace/codes/BladeDISC/pytorch_blade/tests/disc/pdl/pdll_files/cpu/dequant_gemm_quant.pdll"
]
env_var = {}
env_var["DISC_TORCH_PDL_FILES"] = ",".join(DISC_TORCH_PDL_FILES)
with set_env(**env_var), cfg:
    opt_model = optimize(traced_model, True, dummy_input)
    print(opt_model.inlined_graph)
torch.jit.save(opt_model, "bert_mini.int8.disc.torchscript")
evaluate(opt_model, eval_dataloader, "cpu")
