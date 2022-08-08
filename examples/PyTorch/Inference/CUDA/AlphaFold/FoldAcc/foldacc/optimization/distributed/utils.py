import os

import torch
import torch.nn as nn

from foldacc.model.distributed.comm import Gather, Scatter, All_to_All

torch.classes.load_library(f"{os.path.dirname(__file__)}/../../../foldacc_custom.so")

class Gather_save(nn.Module):
    def __init__(self, dim = 0, world_size = 1):
        super(Gather_save, self).__init__()
        self.dim = dim
        self.world_size = world_size
        self.op = torch.classes.paifold.PaiGather(dim, world_size)
    
    def forward(self, x):
        return self.op.forward(x)

class Scatter_save(nn.Module):
    def __init__(self, dim = 0, world_size = 1):
        super(Scatter_save, self).__init__()
        self.dim = dim
        self.world_size = world_size
        self.op = torch.classes.paifold.PaiScatter(dim, world_size)

    def forward(self, x):
        return self.op.forward(x)

class AlltoAll_save(nn.Module):
    def __init__(self, in_dim = 0, out_dim = 1, world_size = 1):
        super(AlltoAll_save, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.world_size = world_size
        self.op = torch.classes.paifold.PaiAlltoAll(in_dim, out_dim, world_size)

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
        return All_to_All.apply(x, self.in_dim, self.out_dim)
