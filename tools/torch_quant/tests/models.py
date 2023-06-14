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

from functools import partial

import torch
import torch.nn as nn
from torch_quant.graph import GraphModContext
from torch_quant.module import fx_trace
from torch_quant.observer import Observer


def create_ctx(model: nn.Module) -> GraphModContext:
    mapping = fx_trace(model)
    dummy_observer = partial(Observer, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    ctx = GraphModContext(
        gm=mapping[''].gm,
        root=mapping[''].m,
        act_ob_ctr=dummy_observer,
        w_ob_ctr=dummy_observer,
        bias_ob_ctr=dummy_observer,
    )
    return ctx


class SubModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SimpleLinearWithBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


class SimpleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3)
        self.sub = SubModule()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(4, 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.sub(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class LinearReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


class UntraceableSubModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu = LinearReLU()

    def forward(self, x):
        if torch.max(x) < 0:
            x = x + 1
        x = self.linear_relu(x)
        return x


class UntraceableSimpleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.traceable_sub = SimpleModule()
        self.untraceable_sub = UntraceableSubModule()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        x = self.traceable_sub(x)
        x = self.untraceable_sub(x)
        x = self.linear(x)
        return x
