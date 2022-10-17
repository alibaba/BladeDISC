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

from turtle import forward
from typing import Optional, List, Generator
import math

import torch
import torch.nn as nn

from foldacc.model.modules.ops import (
    Linear,
    LayerNorm,
    Attention,
    DropoutRowwise,
    DropoutColumnwise,
    Transition
)
from foldacc.model.modules.utils import permute_final_dims, get_padding_size, chunk_layer, padding_feat, split_feat
from foldacc.model.modules.triangle import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from foldacc.model.distributed.comm import gather, scatter, col_to_row, row_to_col


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """
    def __init__(self, 
        c_t, 
        c_z, 
        c_hidden, 
        no_heads, 
        inf=1e4, 
        comm_dtype=torch.float, 
        gather_output=True, 
        scatter_input=True,
        attn_o_bias=True,
        dtype=torch.float,
    ):
    
        super(TemplatePointwiseAttention, self).__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.comm_dtype = comm_dtype
        self.gather_output = gather_output
        self.scatter_input = scatter_input
        self.dtype = dtype

        self.mha = Attention(
            self.c_z,
            self.c_t,
            self.c_t,
            self.c_hidden,
            self.no_heads,
            gating=False,
            o_bias=attn_o_bias,
            inf=inf
        )

    def _chunk(self,
        z: torch.Tensor,
        t: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "kv_x": t,
            "biases": biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self, 
        t: torch.Tensor, 
        z: torch.Tensor, 
        template_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:

        has_batch_dim = False
        if t.shape[0] == 1:
            t = t.squeeze(0)
            z = z.squeeze(0)
            template_mask = template_mask.squeeze(0)
            has_batch_dim = True

        if torch.distributed.is_initialized():
            dap_size = torch.distributed.get_world_size()
        else:
            dap_size = 1

        seq_length = z.shape[-3]
        padding_size = get_padding_size(seq_length, dap_size)

        if dap_size != 1:
            t = padding_feat(t, padding_size, padding_size)
            z = padding_feat(z, padding_size, padding_size)

            if self.scatter_input:
                t = scatter(t, dim=1)
                z = scatter(z, dim=0)

        if template_mask is None:
            template_mask = t.ones(t.shape[:-3]).to(t.dtype).cuda()

        bias = self.mha.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        zr = z.reshape(z.shape[0]*z.shape[1], *z.shape[2:])
        tr = t.reshape(t.shape[0]*t.shape[1], *t.shape[2:])
        biasr = bias.reshape(bias.shape[0]*bias.shape[1], *bias.shape[2:])

        biases = [biasr]
        # [*, N_res, N_res, 1, C_z]
        if chunk_size is not None:
            o = self._chunk(zr, tr, biases, chunk_size)
        else:
            o = self.mha(q_x=zr, kv_x=tr, biases=biases)

        o = o.reshape(z.shape[0], z.shape[1], *o.shape[1:])

        # [*, N_res, N_res, C_z]
        o = o.squeeze(-2)
        if dap_size != 1:
            if self.gather_output:
                o = gather(o, dim=0, dtype=self.comm_dtype)

                o = split_feat(o, padding_size, padding_size)

        if has_batch_dim:
            o = o.unsqueeze(0)
        return o

class TemplatePairStackBlock(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        tri_attn_first: bool = True,
        gather_output: bool = False,
        scatter_input: bool = False,
        comm_dtype = torch.float,
        low_mem = False,
        act=torch.nn.ReLU,
        dtype=torch.float,
        final_layer_norm=None,
        **kwargs,
    ):
        super(TemplatePairStackBlock, self).__init__()

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.tri_attn_first = tri_attn_first
        self.inf = inf
        self.gather_output = gather_output
        self.scatter_input = scatter_input
        self.comm_dtype = comm_dtype
        self.low_mem = low_mem
        self.dtype = dtype

        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.final_layer_norm = final_layer_norm

        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
            comm_dtype=comm_dtype,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
            comm_dtype=comm_dtype,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_t,
            self.c_hidden_tri_mul,
            comm_dtype=comm_dtype,
            low_mem=low_mem,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_t,
            self.c_hidden_tri_mul,
            comm_dtype=comm_dtype,
            low_mem=low_mem,
        )

        self.pair_transition = Transition(
            self.c_t,
            self.pair_transition_n,
            act=act
        )

    def forward_loop(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None, 
    ):
        single_templates = [
            t for t in torch.unbind(z, dim=-4)
        ]
        single_templates_masks = [
            m for m in torch.unbind(mask, dim=-3)
        ]

        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]

            single = self.forward_single(single, single_mask, chunk_size)

            single_templates[i] = single.unsqueeze(-4)

        z = torch.cat(single_templates, dim=-4)

        return z

    def forward(self, 
        z: torch.Tensor, 
        mask: torch.Tensor, 
        chunk_size: Optional[int] = None, 
        _mask_trans: bool = False
    ):
        if(mask.shape[-3] == 1):
            expand_idx = list(mask.shape)
            expand_idx[-3] = z.shape[-4]
            mask = mask.expand(*expand_idx)

        if torch.distributed.is_initialized():
            dap_size = torch.distributed.get_world_size()
        else:
            dap_size = 1

        seq_length = mask.shape[-1]
        padding_size = get_padding_size(seq_length, dap_size)

        if not self.low_mem:
            if dap_size != 1:
                mask = padding_feat(mask, padding_size, padding_size, is_mask=True)

                if self.scatter_input:
                    z = padding_feat(z, padding_size, padding_size)
                    z = scatter(z, dim=1)
        
        z = self.forward_loop(z, mask, chunk_size)

        if not self.low_mem:
            if dap_size != 1:
                if self.gather_output:
                    z = gather(z, dim=1, dtype=self.comm_dtype)

                    z = split_feat(z, padding_size, padding_size, has_batch=True)

        return z
    
    def forward_single(self, z, mask, chunk_size=None):
        single, single_mask = z, mask
        _mask_trans = False

        if self.low_mem:
            if torch.distributed.is_initialized():
                dap_size = torch.distributed.get_world_size()
            else:
                dap_size = 1

            if dap_size != 1:
                seq_length = single_mask.shape[-1]
                padding_size = get_padding_size(torch.tensor(seq_length), torch.tensor(dap_size))
                single_mask = padding_feat(single_mask, padding_size, padding_size, is_mask=True)

                if self.scatter_input:
                    single = padding_feat(single, padding_size, padding_size)
                    single = scatter(single, dim=0)

            single_mask_row = scatter(single_mask, dim=0)
            single_mask_col = scatter(single_mask, dim=1)
        else:
            single_mask_row = scatter(single_mask, dim=0)
            single_mask_col = scatter(single_mask, dim=1)

        if self.tri_attn_first:
            single += self.dropout_row(
                self.tri_att_start(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask_row
                ))

            single = row_to_col(single, dtype=self.comm_dtype)
            single += self.dropout_col(
                self.tri_att_end(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask_col
                )
            )
            single = col_to_row(single, dtype=self.comm_dtype)
            single += self.dropout_row(
                self.tri_mul_out(
                    single,
                    mask=single_mask_row,
                    chunk_size=chunk_size
                )
            )
            single = row_to_col(single, dtype=self.comm_dtype)
            single += self.dropout_row(
                self.tri_mul_in(
                    single,
                    mask=single_mask_col,
                    chunk_size=chunk_size
                )
            )
        else:
            single += self.dropout_row(
                self.tri_mul_out(
                    single,
                    mask=single_mask_row,
                    chunk_size=chunk_size
                )
            )
            single = row_to_col(single, dtype=self.comm_dtype)
            single += self.dropout_row(
                self.tri_mul_in(
                    single,
                    mask=single_mask_col,
                    chunk_size=chunk_size
                )
            )
            single = col_to_row(single, dtype=self.comm_dtype)
            single += self.dropout_row(
                self.tri_att_start(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask_row
                ))
            single = row_to_col(single, dtype=self.comm_dtype)
            single += self.dropout_col(
                self.tri_att_end(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask_col
                )
            )

        mask = single_mask_col if _mask_trans else torch.ones(single.shape[:-1]).to(dtype=single.dtype).cuda().unsqueeze(-1)
        single += self.pair_transition(
            single,
            mask,
            chunk_size,
        )
        single = col_to_row(single, dtype=self.comm_dtype)

        if self.low_mem and self.final_layer_norm is not None:
            single = self.final_layer_norm(single)

        if self.low_mem:
            if dap_size != 1:
                if self.gather_output:
                    single = gather(single, dim=0, dtype=self.comm_dtype)
                    single = split_feat(single, padding_size, padding_size)

        return single

class TemplatePairStack(nn.Module):
    """
    Implements Algorithm 16.
    """
    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
        dropout_rate,
        tri_attn_first=True,
        inf=1e9,
        comm_dtype=torch.float,
        low_mem=False,
        return_mean=False,
        act=torch.nn.ReLU,
        dtype=torch.float,
        **kwargs,
    ):
        super(TemplatePairStack, self).__init__()
        self.return_mean = return_mean
        self.c_t = c_t
        self.low_mem = low_mem

        self.layer_norm = LayerNorm(c_t)

        self.blocks = nn.ModuleList()
        for block_id in range(no_blocks):
            block = TemplatePairStackBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                tri_attn_first=tri_attn_first,
                inf=inf,
                scatter_input=(block_id == 0),
                gather_output=(block_id == no_blocks - 1),
                comm_dtype=comm_dtype,
                low_mem=low_mem,
                act=act,
                dtype=dtype,
                final_layer_norm=self.layer_norm if block_id == no_blocks - 1 else None
            )
            self.blocks.append(block)

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = False,
        **kwargs,
    ):
        if type(t) in [Generator, list]:
            sum = 0.0
            count = 0
            new_single_templates = []
            for ti in t:
                for block in self.blocks:
                    if chunk_size is None:
                        ti = block(ti, mask)
                    else:
                        ti = block(ti, mask, chunk_size)

                if self.return_mean:
                    if not self.low_mem:
                        ti = self.layer_norm(ti)
                    sum += ti
                    count += 1
                else:
                    if not self.low_mem:
                        ti = self.layer_norm(ti)
                    new_single_templates.append(ti)
                
                torch.cuda.empty_cache()

            if self.return_mean:
                if count > 0:
                    sum /= count
                    t = sum
                else:
                    t = None
            else:
                t = torch.cat(
                    [s.unsqueeze(-4) for s in new_single_templates], dim=-4
                )
        else:
            has_batch_dim = False

            if t.shape[0] == 1:
                t = t.squeeze(0)
                mask = mask.squeeze(0)
                has_batch_dim = True

            for block in self.blocks:
                if chunk_size is None:
                    t = block(t, mask)
                else:
                    t = block(t, mask, chunk_size)

            if not self.low_mem:
                t = self.layer_norm(t)
            
            if self.return_mean:
                t = torch.mean(t, 0)
            
            if has_batch_dim:
                t = t.unsqueeze(0)

        return t
