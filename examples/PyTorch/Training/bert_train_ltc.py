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
import inspect
import os
import sys
import time
from datetime import timedelta
import timeit
import pandas as pd
import numpy as np

import torch
from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import torch_blade
import torch_blade.utils as utils
import ctypes
import torch._lazy as ltc
import torch._lazy.metrics as ltc_metrics
utils.disable_pytorch_jit()
torch_blade.init_ltc_disc_backend()
torch.manual_seed(2)


torch.backends.cuda.matmul.allow_tf32 = True

# You will download around 84G dataset if you run this end to end training/evaluation example.

os.environ["TOKENIZERS_PARALLELISM"] = "false"

results = []
def printStats(backend, timings, precision):
    times = np.array(timings)
    steps = len(times)
    speeds = 1.0/times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = (
        "\n%s =================================\n"
        "num iterations=%d\n"
        "  Median FPS: %.1f, mean: %.1f\n"
        "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
    ) % (
        backend,
        steps,
        speed_med,
        speed_mean,
        time_med,
        time_mean,
        time_99th,
        time_std,
    )
    meas = {
        "Backend": backend,
        "precision": precision,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median(ms)": time_med,
        "Mean(ms)": time_mean,
        "99th_p": time_99th,
        "std_dev": time_std,
    }
    results.append(meas)


def data_processing(num_samples, batch_size):
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].select(range(num_samples))
    small_eval_dataset = tokenized_datasets["test"].select(range(num_samples))

    train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def training_iter_fn(batch, model, optimizer, scaler):
    with torch.cuda.amp.autocast():
        outputs = model(**batch)
        loss = outputs.loss
    scaler.scale(loss).backward()
    return loss


def get_device(backend):
    # reference to Lazy device if enable LazyTensor Core
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if backend == "ltc-disc":
        device = torch.device("lazy")
    return device


def model_training_evaluation(
        backend, train_dataloader, eval_dataloader, model, optimizer, num_epochs, evaluation):
    device = get_device(backend)

    model.to(device)
    model.train()
    total_iter = 0
    timings = []
    profiling_warmup = 20

    tr_loss = 0.0
    loss_history = []
    for epoch in range(num_epochs):
        end_time = timeit.default_timer()
        for i, batch in enumerate(train_dataloader, 0):
            total_iter += 1
            start_time = end_time
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            if backend == "ltc-disc":
                torch._lazy.mark_step()
            loss_history.append(loss.item())
            tr_loss += loss.item()
            if i % 5 == 0:
                tr_loss = torch.tensor(0.0).to(device)

            end_time = timeit.default_timer()
            meas_time = end_time - start_time
            print("epoch: {}, batch: {}, loss: {}, elapsed:{}".format(epoch, i, loss.item(), meas_time))
            timings.append(meas_time)

    printStats(backend or "PyTorch", timings[1:], "fp32")
    if evaluation:
        metric = load_metric("accuracy")
        model.eval()
        opt_model = model
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = opt_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        return loss_history, metric.compute()
    else:
        return loss_history, None


def check_loss(ref_loss, res_loss):
    assert len(ref_loss) == len(res_loss)
    length = len(ref_loss)
    x = min(length, 10)
    if sum(res_loss[-x:]) / 10 <= sum(ref_loss[-x:]) / 10 + 1e-1:
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="TorchDynamo end to end training/evaluation benchmark"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of samples to train/eval (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--acc-backend",
        default="eager",
        help="train model with accelerator backend[eager, ltc-disc] (default: eager)",
    )

    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="train model using a given optimizer (default: Adam)",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="running evaluation after model training",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dataloader, eval_dataloader = data_processing(
        args.num_samples, args.batch_size
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer_cls = getattr(sys.modules["torch.optim"], args.optimizer)
    if "capturable" in inspect.signature(optimizer_cls).parameters.keys():
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, capturable=True)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    if args.acc_backend == "ltc-disc":
        actual_loss, accuracy = model_training_evaluation(
            args.acc_backend,
            train_dataloader,
            eval_dataloader,
            model,
            optimizer,
            args.epochs,
            args.evaluation
        )

    native_start = time.time()
    expect_loss, accuracy = model_training_evaluation(
        None,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation
    )
    native_end = time.time()

    check_success = check_loss(actual_loss, expect_loss)
    if args.evaluation:
        print(f"Model accuracy: {accuracy}")
    native_elapsed = native_end - native_start
    print(
        f"Train model on {args.epochs} epochs with backend {args.acc_backend} and optimizer {args.optimizer}:"
    )
    print(f"PyTorch spent {timedelta(seconds=native_elapsed/args.epochs)} per epoch")

    # Generate report
    print("Model Summary:")
    summary = pd.DataFrame(results)
    print(summary.to_markdown())

    assert check_success, "PyTorch LTC end to end training loss is greater than native Pytorch"


if __name__ == "__main__":
    main()
