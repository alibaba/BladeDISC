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

from typing import Tuple
import math

import torch
import torch.distributed as dist
from torch import Tensor

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], torch.distributed.get_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[torch.distributed.get_rank()]

    return output

def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    if dim == 0:
        output_shape = list(tensor.shape)
        output_shape[dim] *= torch.distributed.get_world_size()
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(torch.distributed.get_world_size(), dim=dim)
        dist.all_gather(list(tensor_list),
                        tensor,
                        async_op=False)
    else:
        tensor_list = [
            torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        dist.all_gather(tensor_list,
                        tensor,
                        async_op=False)
        output = torch.cat(tensor_list, dim=dim)

    return output

def scatter(input: Tensor, dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input

    input = Scatter.apply(input, dim)
    return input

def gather(input: Tensor, dim: int = -1, dtype=torch.float) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input
    
    input_dtype = input.dtype
    input = input.to(dtype)
    input = Gather.apply(input, dim)
    input = input.to(input_dtype)
    return input

def _all_to_all(tensor: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[in_dim], torch.distributed.get_world_size())
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]
    output_tensor_list = [torch.ones_like(tensor_) for tensor_ in input_tensor_list]
    dist.all_to_all(output_tensor_list,
                    input_tensor_list,
                    async_op=False)
    output = torch.cat(output_tensor_list, dim=out_dim)

    return output

def col_to_row(input_: Tensor, dtype=torch.float) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input_
    input_dtype = input_.dtype
    input_ = input_.to(dtype)
    input_ = AlltoAll.apply(input_, 0, 1)
    input_ = input_.to(input_dtype)
    return input_

def row_to_col(input_: Tensor, dtype=torch.float) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input_
    input_dtype = input_.dtype
    input_ = input_.to(dtype)
    input_ = AlltoAll.apply(input_, 1, 0)
    input_ = input_.to(input_dtype)
    return input_

class Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Scatter", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: "Scatter", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None

class Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Gather", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: "Gather", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None

class AlltoAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "AlltoAll", input_: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input_, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx: "AlltoAll", grad_output: Tensor) -> Tuple[Tensor]:
        saved_tensors = ctx.saved_tensors[0]
        return _all_to_all(grad_output, in_dim=int(saved_tensors[1]),
                           out_dim=int(saved_tensors[0])), None, None
