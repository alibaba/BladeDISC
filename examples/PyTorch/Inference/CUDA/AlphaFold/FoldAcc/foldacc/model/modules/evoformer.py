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

from cgitb import enable
import math
import torch
import torch.nn as nn

from typing import Tuple, Optional, List, Sequence

from foldacc.model.distributed.comm import gather, scatter, col_to_row, row_to_col
from foldacc.model.modules.msa import (
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    MSAColumnGlobalAttention,
    EvoformerBlockCore
)
from foldacc.model.modules.ops import DropoutRowwise, Linear
from foldacc.model.modules.utils import get_padding_size, chunk_layer, padding_feat, split_feat

class EvoformerBlock(nn.Module):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        outer_product_mean_first: bool = False,
        scatter_input: bool = False,
        gather_output: bool = False,
        low_mem: bool = False,
        is_extra: bool = False,
        comm_dtype = torch.float,
        mask_trans = True,
        act=torch.nn.ReLU,
        dtype=torch.float,
        **kwargs,
    ):
        super(EvoformerBlock, self).__init__()

        self.outer_product_mean_first = outer_product_mean_first
        self.scatter_input = scatter_input
        self.gather_output = gather_output
        self.low_mem = low_mem
        self.is_extra = is_extra
        self.comm_dtype = comm_dtype
        self.dtype = dtype

        self.mask_trans = mask_trans

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf
        )

        if not is_extra:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
            )
        else:
            self.msa_att_col = MSAColumnGlobalAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
                eps=eps,
            )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps,
            outer_product_mean_first=outer_product_mean_first,
            low_mem=low_mem,
            mask_trans=mask_trans,
            act=act,
        )

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if torch.distributed.is_initialized():
            dap_size = torch.distributed.get_world_size()
        else:
            dap_size = 1

        msa_length = msa_mask.shape[-2]
        msa_padding_size = get_padding_size(msa_length, dap_size)

        seq_length = pair_mask.shape[-1]
        padding_size = get_padding_size(seq_length, dap_size)

        if self.scatter_input:
            if dap_size != 1:
                m = padding_feat(m, padding_size, msa_padding_size)
                z = padding_feat(z, padding_size, padding_size)

                m = scatter(m, dim=0)
                z = scatter(z, dim=0)

        if dap_size != 1:
            msa_mask = padding_feat(msa_mask, padding_size, msa_padding_size, is_mask=True)
            pair_mask = padding_feat(pair_mask, padding_size, padding_size, is_mask=True)

        msa_mask_row = scatter(msa_mask, dim=0)

        if self.outer_product_mean_first:
            m = row_to_col(m, dtype=self.comm_dtype)
            z += self.core.outer_product_mean(
                m, mask=msa_mask, chunk_size=chunk_size
            )
            m = col_to_row(m, dtype=self.comm_dtype)

        m += self.msa_dropout_layer(
            self.msa_att_row(m, z=z, mask=msa_mask_row, chunk_size=chunk_size)
        )
        
        m = row_to_col(m, dtype=self.comm_dtype)
        msa_mask_col = scatter(msa_mask, dim=1)

        m += self.msa_att_col(m, mask=msa_mask_col, chunk_size=chunk_size)

        m, z = self.core(
            m, 
            z, 
            msa_mask=msa_mask, 
            pair_mask=pair_mask, 
            chunk_size=chunk_size, 
        )

        if torch.distributed.is_initialized():
            m = col_to_row(m, dtype=self.comm_dtype)
        
        if self.gather_output:
            if dap_size != 1:
                m = gather(m, dim=0, dtype=self.comm_dtype)
                z = gather(z, dim=0, dtype=self.comm_dtype)

                m = split_feat(m, msa_padding_size, padding_size)
                z = split_feat(z, padding_size, padding_size)

        return m, z

class Evoformer(nn.Module):
    """
    Evoformer
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        outer_product_mean_first: bool = False,
        low_mem: bool = False,
        comm_dtype = torch.float,
        mask_trans = True,
        act=torch.nn.ReLU,
        dtype=torch.float,
        **kwargs,
    ):
        super(Evoformer, self).__init__()

        self.c_m = c_m
        self.c_z = c_z

        self.blocks = nn.ModuleList()

        for block_id in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                outer_product_mean_first=outer_product_mean_first,
                scatter_input=(block_id == 0),
                gather_output=(block_id == no_blocks - 1),
                low_mem=low_mem,
                comm_dtype=comm_dtype,
                is_extra=False,
                mask_trans=mask_trans,
                act=act,
                dtype=dtype
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        has_batch_dim = False
        if m.shape[0] == 1:
            m = m.squeeze(0)
            z = z.squeeze(0)
            msa_mask = msa_mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)
            has_batch_dim = True

        for block in self.blocks:
            if chunk_size is None:
                m, z = block(m, z, msa_mask, pair_mask)
            else:
                m, z = block(m, z, msa_mask, pair_mask, chunk_size)

            torch.cuda.empty_cache()

        s = self.linear(m[..., 0, :, :])
        
        if has_batch_dim:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)
            s = s.unsqueeze(0)

        return m, z, s
    
    def _forward_offload(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, z = input_tensors
        return self.forward(m, z, msa_mask, pair_mask, chunk_size)

class ExtraMSA(nn.Module):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        outer_product_mean_first: bool = False,
        low_mem: bool = False,
        comm_dtype = torch.float,
        mask_trans=True,
        act=torch.nn.ReLU,
        dtype = torch.float,
        **kwargs,
    ):
        super(ExtraMSA, self).__init__()
        
        self.c_m = c_m
        self.c_z = c_z
        self.dtype = dtype

        self.blocks = nn.ModuleList()
        for block_id in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                scatter_input=(block_id == 0),
                gather_output=(block_id == no_blocks - 1),
                outer_product_mean_first=outer_product_mean_first,
                low_mem=low_mem,
                comm_dtype=comm_dtype,
                is_extra=True,
                mask_trans=mask_trans,
                act=act,
                dtype=dtype,
            )
            self.blocks.append(block)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        has_batch_dim = False
        if m.shape[0] == 1:
            m = m.squeeze(0)
            z = z.squeeze(0)
            msa_mask = msa_mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)
            has_batch_dim = True

        for b in self.blocks:
            if chunk_size is None:
                m, z = b(m, z, msa_mask, pair_mask)
            else:
                m, z = b(m, z, msa_mask, pair_mask, chunk_size)

            torch.cuda.empty_cache()

        if has_batch_dim:
            z = z.unsqueeze(0)

        return z

    def _forward_offload(self,
        input_tensors: Sequence[torch.Tensor],
        chunk_size: int,
        use_lma: bool = False,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        m, z = input_tensors
        return self.forward(m, z, msa_mask, pair_mask, chunk_size)