# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_blade


def patch_conv2d(mod):
    from torch.fx import symbolic_trace

    def _collect_conv2d(mod):
        def fn_recursive(module):
            if (
                isinstance(module, torch.nn.Conv2d)
            ):
                return [module]
            else:
                ret = []
                if not isinstance(module, torch.nn.Module):
                    return ret
                for child in module.children():
                    ret += fn_recursive(child)
                return ret

        return fn_recursive(mod)

    conv2ds = _collect_conv2d(mod)
    old_conv2d = torch.nn.functional.conv2d
    for conv in conv2ds:
        conv.weight_nhwc = torch.nn.Parameter(conv.weight.data.permute([0, 2, 3, 1]).contiguous())
        conv.weight = torch.nn.Parameter(conv.weight_nhwc.permute([0, 3, 1, 2]))

        def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            # convert input dtype for some autocast context
            return torch.ops.torch_blade.conv2d_weight_nhwc(input.type(conv.weight_nhwc.dtype), conv.weight_nhwc, bias, stride, padding, dilation, groups)

        torch.nn.functional.conv2d = _patched_conv2d
        fx_conv = symbolic_trace(conv)
        conv.forward = fx_conv.forward

    torch.nn.functional.conv2d = old_conv2d


def patch_diffusers_unet_qkv_proj(unet):
    from diffusers.models.attention import CrossAttention

    class CrossAttnProcessor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            attn: CrossAttention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
        ):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            qkv_same_input = False
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                qkv_same_input = True
            elif attn.cross_attention_norm:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
                qkv_same_input = False

            if qkv_same_input:
                qkv = torch.nn.functional.linear(encoder_hidden_states, attn.qkv_weight, attn.qkv_bias)
                query, key, value = qkv.chunk(3, -1)
            else:
                query = attn.to_q(hidden_states)
                kv = torch.nn.functional.linear(encoder_hidden_states, attn.kv_weight, attn.kv_bias)
                key, value = kv.chunk(2, -1)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states

    def _collect_attention(unet):

        def fn_recursive(module):
            if (
                isinstance(module, CrossAttention)
            ):
                return [module]
            else:
                ret = []
                if not isinstance(module, torch.nn.Module):
                    return ret
                for child in module.children():
                    ret += fn_recursive(child)
                return ret
        return fn_recursive(unet)


    attns = _collect_attention(unet)

    # merge qkv weights, and chunk merged weight into views
    for att in attns:
        att.processor = CrossAttnProcessor()
        to_q = att.to_q
        to_k = att.to_k
        to_v = att.to_v
        _, q_embd_dim = to_q.weight.shape
        _, k_embd_dim = to_k.weight.shape
        _, v_embd_dim = to_v.weight.shape
        assert k_embd_dim == v_embd_dim
        if q_embd_dim == k_embd_dim:
            qkv_w = torch.cat([to_q.weight, to_k.weight, to_v.weight], dim=0)
            to_q.weight.data, to_k.weight.data, to_v.weight.data = qkv_w.chunk(3, 0)
            att.qkv_weight = torch.nn.Parameter(qkv_w)
            att.qkv_bias = None
            if to_q.bias and to_k.bias and to_v.bias:
                qkv_b = torch.cat([to_q.bias, to_k.bias, to_v.bias], dim=0)
                to_q.bias.data, to_k.bias.data, to_v.bias.data = qkv_b.chunk(3, 0)
                att.qkv_bias = torch.nn.Parameter(qkv_b)
        else:
            kv_w = torch.cat([to_k.weight, to_v.weight], dim=0)
            to_k.weight.data, to_v.weight.data = kv_w.chunk(2, 0)
            att.kv_weight = torch.nn.Parameter(kv_w)
            att.kv_bias = None
            if to_k.bias and to_v.bias:
                kv_b = torch.cat([to_k.bias, to_v.bias], dim=0)
                to_k.bias.data, to_v.bias.data = kv_b.chunk(2, 0)
                att.kv_bias = torch.nn.Parameter(kv_b)
