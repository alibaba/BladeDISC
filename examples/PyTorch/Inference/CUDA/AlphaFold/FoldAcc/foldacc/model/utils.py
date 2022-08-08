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
from typing import Optional, List

import torch

def attn_trace_module(module, m, biases):
    return module.mha(
        q_x=m, 
        kv_x=m, 
        biases=biases
    )

class attn_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(attn_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, m:torch.Tensor, biases:List[torch.Tensor], chunk_size: int):
        orig_batch_dims = m.shape[:-2]
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= d
        chunk_num = math.ceil(flat_batch_dim/chunk_size)
        outputs = []
        for i in range(int(chunk_num)):
            mi = m[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else m[i*chunk_size:]
            bi = []
            for b in biases:
                if b.shape[0] != 1:
                    bi.append(b[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else b[i*chunk_size:])
                else:
                    bi.append(b)

            out = self.chunk_traced_module(mi, bi)

            outputs.append(out)
        
        output = torch.cat(outputs, 0)
        return output

def global_attn_trace_module(module, m, mask):
    return module.global_attention(
        m=m, 
        mask=mask
    )

class global_attn_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(global_attn_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, m:torch.Tensor, mask:torch.Tensor, chunk_size: int):
        orig_batch_dims = m.shape[:-2]
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= d
        chunk_num = math.ceil(flat_batch_dim/chunk_size)
        outputs = []
        for i in range(int(chunk_num)):
            mi = m[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else m[i*chunk_size:]
            mki = mask[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else mask[i*chunk_size:]

            out = self.chunk_traced_module(mi, mki)

            outputs.append(out)
        
        output = torch.cat(outputs, 0)
        return output

def triangle_attn_trace_module(module, x, biases):
    return module.mha(q_x=x, kv_x=x, biases=biases)

class triangle_attn_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(triangle_attn_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, x:torch.Tensor, biases:List[torch.Tensor], chunk_size: int):
        orig_batch_dims = x.shape[:-2]
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= d
        chunk_num = math.ceil(flat_batch_dim/chunk_size)
        outputs = []
        for i in range(int(chunk_num)):
            xi = x[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else x[i*chunk_size:]
            bi = []
            for b in biases:
                if b.shape[0] != 1:
                    bi.append(b[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else b[i*chunk_size:])
                else:
                    bi.append(b)

            out = self.chunk_traced_module(xi, bi)

            outputs.append(out)
        
        output = torch.cat(outputs, 0)
        return output

class triangle_mul_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(triangle_mul_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, z, mask, weight_ag, bias_ag, weight_ap, bias_ap, weight_bg, bias_bg, weight_bp, bias_bp, chunk_size:int):
        chunk_num = math.ceil(weight_ag.shape[0]/chunk_size)
        outputs = []
        for i in range(int(chunk_num)):
            w_ag = weight_ag[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else weight_ag[i*chunk_size:]
            b_ag = bias_ag[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else bias_ag[i*chunk_size:]
            w_ap = weight_ap[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else weight_ap[i*chunk_size:]
            b_ap = bias_ap[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else bias_ap[i*chunk_size:]
            w_bg = weight_bg[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else weight_bg[i*chunk_size:]
            b_bg = bias_bg[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else bias_bg[i*chunk_size:]
            w_bp = weight_bp[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else weight_bp[i*chunk_size:]
            b_bp = bias_bp[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else bias_bp[i*chunk_size:]

            o = self.chunk_traced_module(z, mask, w_ag, b_ag, w_ap, b_ap, w_bg, b_bg, w_bp, b_bp)
            
            outputs.append(o)
        
        p = torch.cat(outputs, 0)

        return p

class opm_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(opm_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, a:torch.Tensor, b:torch.Tensor, norm:torch.Tensor, chunk_size: int):
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        norm_reshape = norm.reshape((-1,) + norm.shape[-3:])
        out = []
        for a_prime, b_prime, norm_prime in zip(a_reshape, b_reshape, norm_reshape):
            orig_batch_dims = a_prime.shape[:-2]
            flat_batch_dim = 1
            for d in orig_batch_dims:
                flat_batch_dim *= d
            chunk_num = math.ceil(flat_batch_dim/chunk_size)
            outers = []
            for i in range(int(chunk_num)):
                ai = a_prime[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else a_prime[i*chunk_size:]
                ni = norm_prime[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else norm_prime[i*chunk_size:]
                outer = self.chunk_traced_module(ai, b_prime, ni)

                outers.append(outer)

            out.append(torch.cat(outers, 0))

        if(len(out) == 1):
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

def trans_trace_module(module, m, mask):
    m = module.linear_1(m)
    m = module.act(m)
    m = module.linear_2(m) * mask
    return m

class trans_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(trans_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, m:torch.Tensor, mask:torch.Tensor, chunk_size: int):
        orig_batch_dims = m.shape[:-2]
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= d
        chunk_num = math.ceil(flat_batch_dim/chunk_size)
        outputs = []
        for i in range(int(chunk_num)):
            mi = m[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else m[i*chunk_size:]
            mki = mask[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else mask[i*chunk_size:]

            out = self.chunk_traced_module(mi, mki)

            outputs.append(out)
        
        output = torch.cat(outputs, 0)
        return output

class template_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(template_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: torch.Tensor, 
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

            single = self.chunk_traced_module(single, single_mask, chunk_size)

            single_templates[i] = single.unsqueeze(-4)

        z = torch.cat(single_templates, dim=-4)

        return z

def template_attn_trace_module(module, z, t, biases):
    return module.mha(
        q_x=z, 
        kv_x=t, 
        biases=biases
    )

class template_attn_script_module(torch.nn.Module):
    def __init__(self, chunk_traced_module):
        super(template_attn_script_module, self).__init__()
        self.chunk_traced_module = chunk_traced_module
    
    def forward(self, z:torch.Tensor, t:torch.Tensor, biases:List[torch.Tensor], chunk_size: int):
        orig_batch_dims = z.shape[:-2]
        flat_batch_dim = 1
        for d in orig_batch_dims:
            flat_batch_dim *= d
        chunk_num = math.ceil(flat_batch_dim/chunk_size)
        outputs = []
        for i in range(int(chunk_num)):
            zi = z[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else z[i*chunk_size:]
            ti = t[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else t[i*chunk_size:]
            bi = []
            for b in biases:
                if b.shape[0] != 1:
                    bi.append(b[i*chunk_size:(i+1)*chunk_size] if i != chunk_num - 1 else b[i*chunk_size:])
                else:
                    bi.append(b)

            out = self.chunk_traced_module(zi, ti, bi)

            outputs.append(out)
        
        output = torch.cat(outputs, 0)
        return output