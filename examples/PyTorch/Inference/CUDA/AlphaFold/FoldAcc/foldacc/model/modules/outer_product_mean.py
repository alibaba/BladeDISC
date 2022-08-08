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

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldacc.model.modules.ops import Linear
from foldacc.model.modules.utils import chunk_layer
from foldacc.model.distributed.comm import gather, scatter, col_to_row, row_to_col

class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3, comm_dtype=torch.float):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.comm_dtype = comm_dtype

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)

        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

        self.layer_norm_out = None

    def _opm(self, a, b, norm):
        a_norm = torch.mean(norm, -2)
        a = a / (a_norm.unsqueeze(-2) + 1e-4)

        aa = a.transpose(-1, -2).reshape(*a.shape[:-3], a.shape[-3]*a.shape[-1], a.shape[-2])
        bb = b.transpose(-2, -3).reshape(*b.shape[:-3], b.shape[-2], b.shape[-3]*b.shape[-1])
        outer = torch.matmul(aa, bb)
        outer = outer.reshape(*a.shape[:-3], a.shape[-3], a.shape[-1], b.shape[-3], b.shape[-1]).transpose(-3, -2)

        outer = outer.reshape(outer.shape[:-2] + (-1,))
        
        outer = outer / (norm / (a_norm.unsqueeze(-2) + 1e-7) + 1e-4)
        outer = F.linear(outer, self.linear_out.weight) + self.linear_out.bias.reshape(*[1 for i in range(len(norm.shape[:-1]))], -1) / (norm + 1e-3)

        return outer

    def apply_unifold_original_mode(self):
        self.act = nn.GELU()
        self.linear_z = Linear(self.c_z, self.c_z, init="final")
        self.layer_norm_out = nn.LayerNorm(self.c_z)

    def _chunk(self, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        norm: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        norm_reshape = norm.reshape((-1,) + norm.shape[-3:])
        out = []
        for a_prime, b_prime, n_prime in zip(a_reshape, b_reshape, norm_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime, "norm": n_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if(len(out) == 1):
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(m.shape[:-1]).to(m.dtype).cuda()

        # [*, N_seq, N_res, C_m]
        m = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        a = self.linear_1(m)
        b = self.linear_2(m)

        b = gather(b, dim=1, dtype=self.comm_dtype)

        mask = mask.unsqueeze(-1)
        if self.layer_norm_out is not None:
            mask = mask * (mask.shape[-2] ** -0.5)

        mask_col = scatter(mask, dim=1)
        a *= mask_col
        b *= mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, 1]
        mask1 = mask_col.transpose(-1, -3).transpose(-1, -2)
        mask2 = mask.transpose(-1, -3)
        norm = torch.matmul(mask2, mask1).transpose(-1, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, norm, chunk_size)
        else:
            outer = self._opm(a, b, norm)

        if self.layer_norm_out is not None:
            outer = self.act(outer)
            outer = self.layer_norm_out(outer)
            outer = self.linear_z(outer)

        return outer
