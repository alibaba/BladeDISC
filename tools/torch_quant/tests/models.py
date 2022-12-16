import torch
import torch.nn as nn

from torch_quant.graph import GraphModContext
from torch_quant.module import fx_trace


def create_ctx(model: nn.Module) -> GraphModContext:
    mapping = fx_trace(model)
    return GraphModContext(mapping[''].gm, mapping[''].m)


class SubModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
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
