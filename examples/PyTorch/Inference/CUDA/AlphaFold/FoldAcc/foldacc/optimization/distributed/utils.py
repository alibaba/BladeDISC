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

import os

import torch
import torch.nn as nn

from foldacc.model.distributed.comm import Gather, Scatter, AlltoAll

torch.classes.load_library(f"{os.path.dirname(__file__)}/../../../foldacc_custom.so")

class Gather_save(nn.Module):
    def __init__(self, dim = 0, world_size = 1):
        super(Gather_save, self).__init__()
        self.dim = dim
        self.world_size = world_size
        self.op = torch.classes.foldacc.FoldAccGather(dim, world_size)
    
    def forward(self, x):
        return self.op.forward(x)

class Scatter_save(nn.Module):
    def __init__(self, dim = 0, world_size = 1):
        super(Scatter_save, self).__init__()
        self.dim = dim
        self.world_size = world_size
        self.op = torch.classes.foldacc.FoldAccScatter(dim, world_size)

    def forward(self, x):
        return self.op.forward(x)

class AlltoAll_save(nn.Module):
    def __init__(self, in_dim = 0, out_dim = 1, world_size = 1):
        super(AlltoAll_save, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.world_size = world_size
        self.op = torch.classes.foldacc.FoldAccAlltoAll(in_dim, out_dim, world_size)

    def forward(self, x):
        return self.op.forward(x)

class Gather_load(nn.Module):
    def __init__(self, dim = 0):
        super(Gather_load, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return Gather.apply(x, self.dim)

class Scatter_load(nn.Module):
    def __init__(self, dim = 0):
        super(Scatter_load, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return Scatter.apply(x, self.dim)

class AlltoAll_load(nn.Module):
    def __init__(self, in_dim = 0, out_dim = 1):
        super(AlltoAll_load, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, x):
        return AlltoAll.apply(x, self.in_dim, self.out_dim)
