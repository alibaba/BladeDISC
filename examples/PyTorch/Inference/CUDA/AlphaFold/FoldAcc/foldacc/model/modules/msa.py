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

import torch
import torch.nn as nn

from typing import Tuple, Optional, List

from foldacc.model.modules.ops import (
    Linear,
    LayerNorm,
    Attention,
    GlobalAttention,
    DropoutRowwise,
    DropoutColumnwise,
    Transition
)
from foldacc.model.modules.utils import permute_final_dims, chunk_layer
from foldacc.model.modules.outer_product_mean import OuterProductMean
from foldacc.model.modules.triangle import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from foldacc.model.distributed.comm import gather, scatter, col_to_row, row_to_col

class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
        comm_dtype=torch.float
    ):
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf
        self.comm_dtype = comm_dtype

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads, inf=inf
        )

    def _chunk(self, 
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self.mha,
            {
                "q_x": m, 
                "kv_x": m,
                "biases": biases, 
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m = self.layer_norm_m(m)

        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            mask = torch.ones(
                m.shape[:-3] + (n_seq, n_res),
            ).to(device=m.device, dtype=m.dtype)

        mask_bias = (self.mha.inf * (mask - 1))[..., :, None, None, :]

        if (self.pair_bias and
            z is not None and
            self.layer_norm_z is not None and
            self.linear_z is not None
        ):
            z = self.layer_norm_z(z)
            z = self.linear_z(z)

            z = gather(z, dim=0, dtype=self.comm_dtype)

            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        m, mask_bias, z = self._prep_inputs(m, z, mask)

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)

        if chunk_size is not None:
            m = self._chunk(m, biases, chunk_size)
        else:
            m = self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
            )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9, comm_dtype=torch.float):
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
            comm_dtype=comm_dtype
        )


class MSAColumnAttention(nn.Module):
    def __init__(self, c_m, c_hidden, no_heads, inf=1e9):
        super(MSAColumnAttention, self).__init__()
        
        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_in=c_m,
            c_hidden=c_hidden,
            no_heads=no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m = self._msa_att(m, mask=mask, chunk_size=chunk_size)

        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        return m

class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, inf=1e9, eps=1e-10
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.layer_norm_m = nn.LayerNorm(c_in)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf,
            eps=eps,
        )

    def _chunk(self, 
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: torch.Tensor,
    ) -> torch.Tensor:
        return chunk_layer(
            self.global_attention,
            {"m":m, "mask":mask},
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        n_seq, n_res, c_in = m.shape[-3:]

        if mask is None:
            mask = torch.ones(
                m.shape[:-1],
                dtype=m.dtype,
                device=m.device,
            ).detach()

        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        m = self.layer_norm_m(m)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size) 
        else:
            m = self.global_attention(m, mask)

        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        return m


class EvoformerBlockCore(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        outer_product_mean_first = False,
        comm_dtype = torch.float,
        low_mem = False,
        _is_extra_msa_stack: bool = False,
        mask_trans = True,
        act=torch.nn.ReLU,
    ):
        super(EvoformerBlockCore, self).__init__()
        self.outer_product_mean_first = outer_product_mean_first
        self.comm_dtype = comm_dtype
        self.low_mem = low_mem
        self.mask_trans = mask_trans

        self.msa_transition = Transition(
            c_m,
            transition_n,
            act=act
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
            comm_dtype=comm_dtype
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
            comm_dtype=comm_dtype,
            low_mem=low_mem
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
            comm_dtype=comm_dtype,
            low_mem=low_mem
        )

        self.tri_att_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
            comm_dtype=comm_dtype
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
            comm_dtype=comm_dtype
        )

        self.pair_transition = Transition(
            c_z,
            transition_n,
            act=act
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        pair_mask_row = scatter(pair_mask, dim=0)
        pair_mask_col = scatter(pair_mask, dim=1)

        msa_trans_mask = scatter(msa_mask, dim=1).unsqueeze(-1) if self.mask_trans else torch.ones(m.shape[:-1]).to(dtype=m.dtype).cuda().unsqueeze(-1)

        m += self.msa_transition(
            m, msa_trans_mask, chunk_size
        )

        if not self.outer_product_mean_first:
            z += self.outer_product_mean(
                m, mask=msa_mask, chunk_size=chunk_size
            )

        z += self.ps_dropout_row_layer(self.tri_mul_out(z, mask=pair_mask_row, chunk_size=chunk_size))
        z = row_to_col(z, dtype=self.comm_dtype)

        z += self.ps_dropout_row_layer(self.tri_mul_in(z, mask=pair_mask_col, chunk_size=chunk_size))
        z = col_to_row(z, dtype=self.comm_dtype)

        z += self.ps_dropout_row_layer(
            self.tri_att_start(z, mask=pair_mask_row, chunk_size=chunk_size)
        )
        z = row_to_col(z, dtype=self.comm_dtype)

        z += self.ps_dropout_col_layer(
            self.tri_att_end(z, mask=pair_mask_col, chunk_size=chunk_size)
        )

        pair_trans_mask = pair_mask_col.unsqueeze(-1) if self.mask_trans else torch.ones(z.shape[:-1]).to(dtype=z.dtype).cuda().unsqueeze(-1)
        z += self.pair_transition(
            z, pair_trans_mask, chunk_size
        )
        z = col_to_row(z, dtype=self.comm_dtype)

        return m, z

