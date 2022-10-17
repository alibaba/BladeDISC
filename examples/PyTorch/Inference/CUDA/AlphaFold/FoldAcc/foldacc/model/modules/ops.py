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
import numpy as np
from functools import partialmethod
from typing import Optional, Callable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

from foldacc.model.modules.utils import flatten_final_dims, permute_final_dims, chunk_layer

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

class Linear(nn.Linear):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class LayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out

class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = torch.ones(shape).to(dtype=x.dtype).cuda()
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


class Transition(nn.Module):
    def __init__(self, c_m, n, act=torch.nn.ReLU):
        super(Transition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.act = act()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.linear_1(m)
        m = self.act(m)
        m = self.linear_2(m) * mask
        return m

    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        m = self.layer_norm(m)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    s = torch.nn.functional.softmax(t, dim=dim)

    return s

def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, Q, C_hidden]
    query = permute_final_dims(query, (1, 0, 2))

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 2, 0))

    # [*, H, V, C_hidden]
    value = permute_final_dims(value, (1, 0, 2))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b
            
    a = softmax(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a

class Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        o_bias: bool = True,
        inf: float = 1e9,
    ):
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.o_bias = o_bias
        self.inf = inf

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init="final")

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: torch.Tensor,
                  kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        if self.o_bias:
            o = self.linear_o(o)
        else:
            o = F.linear(o, self.linear_o.weight)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: List[torch.Tensor]=[]
    ) -> torch.Tensor:

        q, k, v = self._prep_qkv(q_x, kv_x)

        o = _attention(q, k, v, biases)

        o = self._wrap_up(o, q_x)
        return o

class GlobalAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(c_in, c_hidden * no_heads, bias=False, init="glorot")

        self.linear_k = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_v = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1),
                      dim=-2) / (torch.sum(mask, dim=-1)[..., None] + self.eps)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q = q * (self.c_hidden**(-0.5))

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, N_res, H, N_seq]
        a = torch.matmul(
            q,
            k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
        )
        bias = (self.inf * (mask - 1))[..., :, None, :]
        a = a + bias
        a = softmax(a)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a,
            v,
        )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m