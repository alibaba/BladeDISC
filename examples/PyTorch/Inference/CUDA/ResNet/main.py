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
import random
import torch
import torch_blade
import torch_blade.utils as utils
from torchvision import models

import pandas as pd
import numpy as np
import timeit

import os
os.environ["TORCH_DISC_USE_TORCH_MLIR"] = "true"
os.environ["DISC_ENABLE_STITCH"] = "true"
results = []

def printStats(backend, timings, precision, batch_size=1):
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
    # print(msg)
    meas = {
        "Backend": backend,
        "batch": batch_size,
        "precision": precision,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median(ms)": time_med,
        "Mean(ms)": time_mean,
        "99th_p": time_99th,
        "std_dev": time_std,
    }
    results.append(meas)


@torch.no_grad()
def benchmark(backend, precision, model, inp, batch_size):

    utils.disable_pytorch_jit()
    for i in range(100):
        model(*inp)
    torch.cuda.synchronize()
    timings = []
    for i in range(200):
        start_time = timeit.default_timer()
        output = model(*inp)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        meas_time = end_time - start_time
        timings.append(meas_time)

    printStats(backend, timings, precision, batch_size)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Blade ResNet Example')
    parser.add_argument('--fp16', action="store_true", default=False,
                        help='Enable fp16 optimization')
    parser.add_argument('--amp', action="store_true", default=False,
                        help='Enable amp optimization')
    parser.add_argument('--batch', type=int, default=32,
                        help='The input batch size')
 
    args = parser.parse_args()

    model = models.resnet50().float().cuda().eval()
    dummy = torch.rand(args.batch, 3, 224, 224).cuda()
    opt_cfg = torch_blade.Config()
    opt_cfg.enable_fp16 = args.fp16
    precision = 'fp32'
    if opt_cfg.enable_fp16:
        model.half()
        dummy = dummy.half()
        precision = 'fp16'

    ts_model = torch.jit.script(model)
    precision = 'amp' if args.amp else precision
    with torch.cuda.amp.autocast(args.amp), opt_cfg:
        benchmark("eager", precision, model, (dummy, ), args.batch)
        benchmark("script", precision, ts_model, (dummy, ), args.batch)
        if utils.torch_version_number() >= utils.parse_version("1.14.0"):
            benchmark("dynamo_disc", precision, torch.compile(model, backend='disc'), (dummy, ), args.batch)
        opt_model = torch_blade.optimize(model, allow_tracing=True,model_inputs=(dummy,))
        benchmark("script_disc", precision, opt_model, (dummy, ), args.batch)

    # Generate report
    print("Model Summary:")
    summary = pd.DataFrame(results)
    print(summary.to_markdown())

