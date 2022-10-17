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

from functools import partialmethod, partial
from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from foldacc.model.modules.ops import Linear, LayerNorm, Attention
from foldacc.model.modules.utils import permute_final_dims, chunk_layer
from foldacc.model.distributed.comm import gather, scatter, col_to_row, row_to_col

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True, comm_dtype=torch.float, low_mem=False):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.comm_dtype = comm_dtype
        self.low_mem = low_mem

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _low_mem_mul(self, z, mask, weight_ag, bias_ag, weight_ap, bias_ap, weight_bg, bias_bg, weight_bp, bias_bp):
        a = F.linear(z, weight_ag, bias_ag)
        a.sigmoid_()
        a *= F.linear(z, weight_ap, bias_ap)
        a *= mask

        b = F.linear(z, weight_bg, bias_bg)
        b.sigmoid_()
        b *= F.linear(z, weight_bp, bias_bp)
        b *= mask

        if self._outgoing:
            b = gather(b, dim=0, dtype=self.comm_dtype)
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = gather(a, dim=1, dtype=self.comm_dtype)
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        return torch.bmm(a, b)

    def _chunk(self, z, mask, weight_ag, bias_ag, weight_ap, bias_ap, weight_bg, bias_bg, weight_bp, bias_bp, chunk_size):
        return chunk_layer(
            partial(self._low_mem_mul, z=z, mask=mask),
            {
                "weight_ag":weight_ag,
                "bias_ag":bias_ag,
                "weight_ap":weight_ap,
                "bias_ap":bias_ap,
                "weight_bg":weight_bg,
                "bias_bg":bias_bg,
                "weight_bp":weight_bp,
                "bias_bp":bias_bp
            },
            chunk_size=chunk_size,
            no_batch_dims=1
        )

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size = None
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = torch.ones(z.shape[:-1]).to(device=z.device, dtype=z.dtype)

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)

        if self.low_mem and chunk_size is not None:
            x = self._chunk(
                z, 
                mask,
                self.linear_a_g.weight,
                self.linear_a_g.bias,
                self.linear_a_p.weight,
                self.linear_a_p.bias,
                self.linear_b_g.weight,
                self.linear_b_g.bias,
                self.linear_b_p.weight,
                self.linear_b_p.bias,
                chunk_size
            )
        else:
            a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
            a = a * mask
            b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
            b = b * mask

            if self._outgoing:
                b = gather(b, dim=0, dtype=self.comm_dtype)
                a = permute_final_dims(a, (2, 0, 1))
                b = permute_final_dims(b, (2, 1, 0))
            else:
                a = gather(a, dim=1, dtype=self.comm_dtype)
                a = permute_final_dims(a, (2, 1, 0))
                b = permute_final_dims(b, (2, 0, 1))

            x = torch.bmm(a, b)

        x = permute_final_dims(x, (1, 2, 0))
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        z = x * g

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class TriangleAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, starting, inf=1e9, comm_dtype=torch.float, low_mem=False
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        self.comm_dtype = comm_dtype
        self.low_mem = low_mem

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads, inf=inf
        )

    def _chunk(self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self.mha,
            {
                "q_x": x, 
                "kv_x": x,
                "biases": biases, 
            },
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2])
        )

    def forward(self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if mask is None:
            # [*, I, J]
            mask = torch.ones(x.shape[:-1]).to(device=x.device, dtype=x.dtype)

        # Shape annotations assume self.starting. Else, I and J are flipped
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.mha.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = self.linear(x)
        triangle_bias = gather(triangle_bias, dim=0, dtype=self.comm_dtype)
        triangle_bias = permute_final_dims(triangle_bias, (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size)
        else:
            x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


class TriangleAttentionStartingNode(TriangleAttention):
    """
    Implements Algorithm 13.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
