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

from foldacc.model.modules.utils import permute_final_dims, get_padding_size
from foldacc.model.modules.ops import Linear
from foldacc.model.distributed.comm import gather, scatter, col_to_row, row_to_col

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hid: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        separate_kv: bool = False,
        bias: bool = True,
        eps: float = 1e-8,
        low_mem = False,
        scatter_input = False,
        gather_output = False,
        dtype = torch.float,
        **kwargs,
    ):
        super(InvariantPointAttention, self).__init__()

        self.d_hid = d_hid
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.eps = eps
        self.low_mem = low_mem
        self.scatter_input = scatter_input
        self.gather_output = gather_output
        self.dtype = dtype

        hc = self.d_hid * self.num_heads
        self.linear_q = Linear(d_single, hc, bias=bias)
        self.separate_kv = separate_kv
        if self.separate_kv:
            self.linear_k = Linear(d_single, hc, bias=bias)
            self.linear_v = Linear(d_single, hc, bias=bias)
        else:
            self.linear_kv = Linear(d_single, 2 * hc, bias=bias)

        hpq = self.num_heads * self.num_qk_points * 3
        self.linear_q_points = Linear(d_single, hpq)
        hpk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        if self.separate_kv:
            self.linear_k_points = Linear(d_single, hpk)
            self.linear_v_points = Linear(d_single, hpv)
        else:
            hpkv = hpk + hpv
            self.linear_kv_points = Linear(d_single, hpkv)

        self.linear_b = Linear(d_pair, self.num_heads)

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))

        concat_out_dim = self.num_heads * (d_pair + self.d_hid + self.num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, d_single, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        f,
        square_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.linear_q(s)

        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        if self.separate_kv:
            k = self.linear_k(s)
            v = self.linear_v(s)
            k = k.view(k.shape[:-1] + (self.num_heads, -1))
            v = v.view(v.shape[:-1] + (self.num_heads, -1))
        else:
            kv = self.linear_kv(s)
            kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))
            k, v = torch.split(kv, self.d_hid, dim=-1)

        q_pts = self.linear_q_points(s)

        def process_points(pts, no_points, split=False):
            shape = pts.shape[:-1] + (pts.shape[-1] // 3, 3)
            if self.separate_kv:
                # alphafold-multimer uses different layout
                pts = pts.view(pts.shape[:-1] + (self.num_heads, no_points * 3))
            pts = torch.split(pts, pts.shape[-1] // 3, dim=-1)
            pts = torch.stack(pts, dim=-1).view(*shape)
            pts = f[..., None].apply(pts)

            pts = pts.view(pts.shape[:-2] + (self.num_heads, no_points, 3))
            return pts

        q_pts = process_points(q_pts, self.num_qk_points, split=True)

        if self.separate_kv:
            k_pts = self.linear_k_points(s)
            v_pts = self.linear_v_points(s)
            k_pts = process_points(k_pts, self.num_qk_points)
            v_pts = process_points(v_pts, self.num_v_points)
        else:
            kv_pts = self.linear_kv_points(s)

            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = f[..., None].apply(kv_pts)

            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.num_heads, -1, 3))

            k_pts, v_pts = torch.split(
                kv_pts, [self.num_qk_points, self.num_v_points], dim=-2
            )

        if torch.distributed.is_initialized():
            dap_size = torch.distributed.get_world_size()
        else:
            dap_size = 1

        seq_length = s.shape[-2]
        padding_size = get_padding_size(torch.tensor(seq_length), torch.tensor(dap_size))

        if self.scatter_input:
            if dap_size != 1:
                if padding_size != 0:
                    q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, padding_size))
                    q_pts = torch.nn.functional.pad(q_pts, (0, 0, 0, 0, 0, 0, 0, padding_size))
                    z = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, padding_size))
                    square_mask = torch.nn.functional.pad(square_mask, (0, 0, 0, padding_size))

                if s.shape[0] == 1:
                    q = scatter(q, dim=1)
                    q_pts = scatter(q_pts, dim=1)
                    z = scatter(z, dim=1)
                    square_mask = scatter(square_mask, dim=2)
                else:
                    q = scatter(q, dim=0)
                    q_pts = scatter(q_pts, dim=0)
                    z = scatter(z, dim=0)   
                    square_mask = scatter(square_mask, dim=1)

        bias = self.linear_b(z)

        attn = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )
        attn *= math.sqrt(1.0 / (3 * self.d_hid))
        attn += (math.sqrt(1.0 / 3) * permute_final_dims(bias, (2, 0, 1)))

        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att.float() ** 2

        pt_att = pt_att.sum(dim=-1)
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.num_qk_points * 9.0 / 2))
        )
        pt_att *= head_weights * (-0.5)

        pt_att = torch.sum(pt_att, dim=-1)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        attn += square_mask
        attn += pt_att.type(attn.dtype)
        attn = self.softmax(attn)

        o = torch.matmul(attn, v.transpose(-2, -3)).transpose(-2, -3)
        o = o.contiguous().view(*o.shape[:-2], -1)

        o_pts = torch.sum(
            (
                attn[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        o_pts = permute_final_dims(o_pts, (2, 0, 3, 1))

        o_pair = torch.matmul(attn.transpose(-2, -3), z)

        o_pair = o_pair.view(*o_pair.shape[:-2], -1)

        if self.gather_output:
            if dap_size != 1:
                if s.shape[0] == 1:
                    o_pair = gather(o_pair, dim=1, dtype=self.dtype)
                    o = gather(o, dim=1, dtype=self.dtype)
                    o_pts = gather(o_pts.contiguous(), dim=1, dtype=self.dtype)
                else:
                    o_pair = gather(o_pair, dim=0, dtype=self.dtype)
                    o = gather(o, dim=0, dtype=self.dtype)
                    o_pts = gather(o_pts.contiguous(), dim=0, dtype=self.dtype)
                
                if padding_size != 0:
                    if s.shape[0] == 1:
                        o_pair = o_pair[:, :-padding_size]
                        o = o[:, :-padding_size]
                        o_pts = o_pts[:, :-padding_size]
                    else:
                        o_pair = o_pair[:-padding_size]
                        o = o[:-padding_size]
                        o_pts = o_pts[:-padding_size]

        o_pts = f[..., None, None].invert_apply(o_pts)

        o_pts_norm = torch.sqrt(torch.sum(o_pts.float() ** 2, dim=-1) + self.eps).type(
            o_pts.dtype
        )

        o_pts_norm = o_pts_norm.view(*o_pts_norm.shape[:-2], -1)

        o_pts = o_pts.view(*o_pts.shape[:-3], -1, 3)

        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pts, dim=-1), o_pts_norm, o_pair), dim=-1)
        )
    
        return s