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

import torch

from foldacc.model.optimize import optimize_module
from foldacc.model.modules.evoformer import Evoformer, ExtraMSA
from foldacc.model.modules.template import TemplatePairStack

def optimize_openfold(
    model, 
    model_config, 
    enable_disc=True, 
    enable_low_mem=False,
    enable_amp=True,
    enable_trace=True,
    dtype=torch.half,
    trace_check_tolerance=1e-5,
    save_dir=None,
    load_dir=None,
    device="0"
):
    torch.cuda.set_device(device)
    torch._C._jit_set_profiling_executor(False)

    dummy_seq_len = 128
    dummy_msa_size = 128
    dummy_chunk_size = 64
    
    model_dtype = torch.float if enable_amp else dtype

    # ---------------------------- optimize evoformer ------------------------------
    evoformer_config = model_config["model"]["evoformer_stack"]
    evoformer = Evoformer(
        **evoformer_config,
        comm_dtype = dtype,
        mask_trans = False,
        low_mem = enable_low_mem,
        dtype = model_dtype
    )

    evoformer.eval()
    evoformer_params = model.evoformer.state_dict()
    evoformer.load_state_dict(evoformer_params)
    evoformer = evoformer.to(device=device, dtype=model_dtype)

    dummy_inputs = (
        torch.randn(dummy_msa_size, dummy_seq_len, evoformer.c_m).to(device=device, dtype=model_dtype), # m
        torch.randn(dummy_seq_len, dummy_seq_len, evoformer.c_z).to(device=device, dtype=model_dtype), # z
        torch.ones(dummy_msa_size, dummy_seq_len).to(device=device, dtype=model_dtype), # msa_mask
        torch.ones(dummy_seq_len, dummy_seq_len).to(device=device, dtype=model_dtype), # pair_mask
        dummy_chunk_size # chunk_size
    )

    evoformer = optimize_module(evoformer, dummy_inputs=dummy_inputs, 
        enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
        dtype=dtype, trace_check_tolerance=trace_check_tolerance,
        save_dir=save_dir, load_dir=load_dir, name="evoformer", device=device)

    model.evoformer = evoformer

    # ---------------------------- optimize extra msa ------------------------------
    if model_config["model"]["extra_msa"]["enabled"]:
        extra_msa_config = model_config["model"]["extra_msa"]["extra_msa_stack"]
        extra_msa = ExtraMSA(
            **extra_msa_config,
            comm_dtype = dtype,
            mask_trans = False,
            low_mem = enable_low_mem,
            dtype = model_dtype
        )

        extra_msa.eval()
        extra_params = model.extra_msa_stack.state_dict()
        extra_msa.load_state_dict(extra_params)
        extra_msa = extra_msa.to(device=device, dtype=model_dtype)

        dummy_inputs = (
            torch.randn(dummy_msa_size, dummy_seq_len, extra_msa.c_m).to(device=device, dtype=model_dtype), # m
            torch.randn(dummy_seq_len, dummy_seq_len, extra_msa.c_z).to(device=device, dtype=model_dtype), # z
            torch.ones(dummy_msa_size, dummy_seq_len).to(device=device, dtype=model_dtype), # msa_mask
            torch.ones(dummy_seq_len, dummy_seq_len).to(device=device, dtype=model_dtype), # pair_mask
            dummy_chunk_size # chunk_size
        )

        extra_msa = optimize_module(extra_msa, dummy_inputs=dummy_inputs, 
            enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
            dtype=dtype, trace_check_tolerance=trace_check_tolerance,
            save_dir=save_dir, load_dir=load_dir, name="extra_msa", device=device)

        model.extra_msa_stack = extra_msa

    # ---------------------------- optimize template ------------------------------
    if model_config["model"]["template"]["enabled"]:
        template_config = model_config["model"]["template"]
        template_ps_config = template_config["template_pair_stack"]
        template_pair_stack = TemplatePairStack(
            **template_ps_config,
            comm_dtype = dtype,
            low_mem = enable_low_mem,
            dtype = model_dtype
        )

        template_pair_stack.eval()
        template_pair_stack_params = model.template_pair_stack.state_dict()
        template_pair_stack_params[f"blocks.{len(template_pair_stack.blocks)-1}.final_layer_norm.weight"] = template_pair_stack_params["layer_norm.weight"]
        template_pair_stack_params[f"blocks.{len(template_pair_stack.blocks)-1}.final_layer_norm.bias"] = template_pair_stack_params["layer_norm.bias"]
        template_pair_stack.load_state_dict(template_pair_stack_params)
        template_pair_stack = template_pair_stack.to(device=device, dtype=model_dtype)

        dummy_inputs = (
            torch.randn(4, dummy_msa_size, dummy_seq_len, template_pair_stack.c_t).to(device=device, dtype=model_dtype), # t
            torch.ones(1, dummy_msa_size, dummy_seq_len).to(device=device, dtype=model_dtype), # mask
            dummy_chunk_size # chunk_size
        )

        template_pair_stack = optimize_module(template_pair_stack, dummy_inputs=dummy_inputs,
            enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
            dtype=dtype, trace_check_tolerance=trace_check_tolerance,
            save_dir=save_dir, load_dir=load_dir,name="template_pair_stack", device=device)

        model.template_pair_stack = template_pair_stack

    return model