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
from foldacc.model.modules.template import TemplatePairStack, TemplatePointwiseAttention
from foldacc.model.modules.embedder import InputEmbedder, TemplatePairEmbedder
from foldacc.model.modules.structure_module import InvariantPointAttention

def unifold2foldacc(state_dict, load_dict):
    """ convert unifold weight to foldacc """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "msa_att_col" in key:
            new_key = key.replace("msa_att_col", "msa_att_col._msa_att")
        if "extra_msa_stack" in key:
            new_key = key.replace("extra_msa_stack", "extra_msa_stack.stack")
        if "tri_mul" in key:
            if "linear_ab_p" in key:
                weights = torch.split(value, value.shape[0]//2)
                new_key = key.replace("tri_mul_out", "core.tri_mul_out").replace("tri_mul_in", "core.tri_mul_in")
                if new_key.replace("linear_ab_p", "linear_a_p") not in load_dict:
                    new_key = key
                new_state_dict[new_key.replace("linear_ab_p", "linear_a_p")] = weights[0]
                new_state_dict[new_key.replace("linear_ab_p", "linear_b_p")] = weights[1]
                continue
            if "linear_ab_g" in key:
                weights = torch.split(value, value.shape[0]//2)
                new_key = key.replace("tri_mul_out", "core.tri_mul_out").replace("tri_mul_in", "core.tri_mul_in")
                if new_key.replace("linear_ab_g", "linear_b_g") not in load_dict:
                    new_key = key
                new_state_dict[new_key.replace("linear_ab_g", "linear_a_g")] = weights[0]
                new_state_dict[new_key.replace("linear_ab_g", "linear_b_g")] = weights[1]
                continue
        if ".pae." in key:
            new_key = key.replace(".pae.", ".tm.")
        if "msa_transition" in key:
            new_key = key.replace("msa_transition", "core.msa_transition")
        if "outer_product_mean" in key:
            new_key = key.replace("outer_product_mean", "core.outer_product_mean")
        if "tri_mul_out" in key:
            new_key = key.replace("tri_mul_out", "core.tri_mul_out")
        if "tri_mul_in" in key:
            new_key = key.replace("tri_mul_in", "core.tri_mul_in")
        if "tri_att_start" in key:
            new_key = key.replace("tri_att_start", "core.tri_att_start")
        if "tri_att_end" in key:
            new_key = key.replace("tri_att_end", "core.tri_att_end")
        if "pair_transition" in key:
            new_key = key.replace("pair_transition", "core.pair_transition")

        if new_key not in load_dict:
            new_key = key

        new_state_dict[new_key] = value
    return new_state_dict

def optimize_unifold(
    model, 
    model_config, 
    enable_disc=True, 
    enable_low_mem=False,
    enable_amp=True,
    enable_trace=True,
    dtype=torch.half,
    save_dir=None,
    load_dir=None,
    device="0"
):
    torch.cuda.set_device(device)
    torch._C._jit_set_profiling_executor(True)

    dummy_seq_len = 128
    dummy_msa_size = 128
    dummy_chunk_size = 64

    module_act = torch.nn.ReLU if model_config["globals"]["alphafold_original_mode"] else torch.nn.GELU
    
    # ---------------------------- optimize evoformer ------------------------------
    evoformer_config = model_config["model"]["evoformer_stack"]
    evoformer = Evoformer(
        c_m = evoformer_config["d_msa"],
        c_z = evoformer_config["d_pair"],
        c_hidden_msa_att = evoformer_config["d_hid_msa_att"],
        c_hidden_opm = evoformer_config["d_hid_opm"],
        c_hidden_mul = evoformer_config["d_hid_mul"],
        c_hidden_pair_att = evoformer_config["d_hid_pair_att"],
        c_s = evoformer_config["d_single"],
        no_heads_msa = evoformer_config["num_heads_msa"],
        no_heads_pair = evoformer_config["num_heads_pair"],
        no_blocks = evoformer_config["num_blocks"],
        transition_n = evoformer_config["transition_n"],
        msa_dropout = evoformer_config["msa_dropout"],
        pair_dropout = evoformer_config["pair_dropout"],
        inf = evoformer_config["inf"],
        eps = evoformer_config["eps"],
        outer_product_mean_first = evoformer_config["outer_product_mean_first"],
        comm_dtype = dtype,
        mask_trans = False,
        act = module_act,
        low_mem = enable_low_mem,
        dtype = model.dtype
    )

    if not model_config["globals"]["alphafold_original_mode"]:
        for name, module in evoformer.named_modules():
            if hasattr(module, "apply_unifold_original_mode"):
                module.apply_unifold_original_mode()

    evoformer.eval()
    evoformer_params = model.evoformer.state_dict()
    evoformer_params = unifold2foldacc(evoformer_params, evoformer.state_dict())
    evoformer.load_state_dict(evoformer_params)
    evoformer = evoformer.to(device=device, dtype=model.dtype)

    dummy_inputs = (
        torch.randn(dummy_msa_size, dummy_seq_len, evoformer.c_m).to(device=device, dtype=model.dtype), # m
        torch.randn(dummy_seq_len, dummy_seq_len, evoformer.c_z).to(device=device, dtype=model.dtype), # z
        torch.ones(dummy_msa_size, dummy_seq_len).to(device=device, dtype=model.dtype), # msa_mask
        torch.ones(dummy_seq_len, dummy_seq_len).to(device=device, dtype=model.dtype), # pair_mask
        dummy_chunk_size # chunk_size
    )

    evoformer = optimize_module(evoformer, dummy_inputs=dummy_inputs, 
        enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
        dtype=dtype, save_dir=save_dir, load_dir=load_dir, name="evoformer", device=device)

    model.evoformer = evoformer

    # ---------------------------- optimize extra msa ------------------------------
    if model_config["model"]["extra_msa"]["enabled"]:
        extra_msa_config = model_config["model"]["extra_msa"]["extra_msa_stack"]
        extra_msa = ExtraMSA(
            c_m = extra_msa_config["d_msa"],
            c_z = extra_msa_config["d_pair"],
            c_hidden_msa_att = extra_msa_config["d_hid_msa_att"],
            c_hidden_opm = extra_msa_config["d_hid_opm"],
            c_hidden_mul = extra_msa_config["d_hid_mul"],
            c_hidden_pair_att = extra_msa_config["d_hid_pair_att"],
            no_heads_msa = extra_msa_config["num_heads_msa"],
            no_heads_pair = extra_msa_config["num_heads_pair"],
            no_blocks = extra_msa_config["num_blocks"],
            transition_n = extra_msa_config["transition_n"],
            msa_dropout = extra_msa_config["msa_dropout"],
            pair_dropout = extra_msa_config["pair_dropout"],
            inf = extra_msa_config["inf"],
            eps = extra_msa_config["eps"],
            outer_product_mean_first = extra_msa_config["outer_product_mean_first"],
            comm_dtype = dtype,
            mask_trans = False,
            act = module_act,
            low_mem = enable_low_mem,
            dtype = model.dtype
        )

        if not model_config["globals"]["alphafold_original_mode"]:
            for name, module in extra_msa.named_modules():
                if hasattr(module, "apply_unifold_original_mode"):
                    module.apply_unifold_original_mode()

        extra_msa.eval()
        extra_params = model.extra_msa_stack.state_dict()
        extra_params = unifold2foldacc(extra_params, extra_msa.state_dict())
        extra_msa.load_state_dict(extra_params)
        extra_msa = extra_msa.to(device=device, dtype=model.dtype)

        dummy_inputs = (
            torch.randn(dummy_msa_size, dummy_seq_len, extra_msa.c_m).to(device=device, dtype=model.dtype), # m
            torch.randn(dummy_seq_len, dummy_seq_len, extra_msa.c_z).to(device=device, dtype=model.dtype), # z
            torch.ones(dummy_msa_size, dummy_seq_len).to(device=device, dtype=model.dtype), # msa_mask
            torch.ones(dummy_seq_len, dummy_seq_len).to(device=device, dtype=model.dtype), # pair_mask
            dummy_chunk_size # chunk_size
        )

        extra_msa = optimize_module(extra_msa, dummy_inputs=dummy_inputs, 
            enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
            dtype=dtype, save_dir=save_dir, load_dir=load_dir, name="extra_msa", device=device)

        model.extra_msa_stack = extra_msa

    # ---------------------------- optimize template ------------------------------
    if model_config["model"]["template"]["enabled"]:
        template_config = model_config["model"]["template"]
        template_ps_config = template_config["template_pair_stack"]
        template_pair_stack = TemplatePairStack(
            c_t = template_ps_config["d_template"],
            c_hidden_tri_att = template_ps_config["d_hid_tri_att"],
            c_hidden_tri_mul = template_ps_config["d_hid_tri_mul"],
            no_blocks = template_ps_config["num_blocks"],
            no_heads = template_ps_config["num_heads"],
            pair_transition_n = template_ps_config["pair_transition_n"],
            dropout_rate = template_ps_config["dropout_rate"],
            inf = model_config["globals"]["inf"],
            tri_attn_first = template_ps_config["tri_attn_first"],
            comm_dtype = dtype,
            return_mean = not template_config["template_pointwise_attention"]["enabled"],
            act = module_act,
            low_mem = enable_low_mem,
            dtype = model.dtype
        )

        if not model_config["globals"]["alphafold_original_mode"]:
            for name, module in template_pair_stack.named_modules():
                if hasattr(module, "apply_unifold_original_mode"):
                    module.apply_unifold_original_mode()

        template_pair_stack.eval()
        template_pair_stack_params = model.template_pair_stack.state_dict()
        template_pair_stack_params = unifold2foldacc(template_pair_stack_params, template_pair_stack.state_dict())
        template_pair_stack_params[f"blocks.{len(template_pair_stack.blocks)-1}.final_layer_norm.weight"] = template_pair_stack_params["layer_norm.weight"]
        template_pair_stack_params[f"blocks.{len(template_pair_stack.blocks)-1}.final_layer_norm.bias"] = template_pair_stack_params["layer_norm.bias"]
        template_pair_stack.load_state_dict(template_pair_stack_params)
        template_pair_stack = template_pair_stack.to(device=device, dtype=model.dtype)

        dummy_inputs = (
            torch.randn(4, dummy_msa_size, dummy_seq_len, template_pair_stack.c_t).to(device=device, dtype=model.dtype), # t
            torch.ones(1, dummy_msa_size, dummy_seq_len).to(device=device, dtype=model.dtype), # mask
            dummy_chunk_size # chunk_size
        )

        template_pair_stack = optimize_module(template_pair_stack, dummy_inputs=dummy_inputs,
            enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
            dtype=dtype, save_dir=save_dir, load_dir=load_dir,name="template_pair_stack", device=device)

        model.template_pair_stack = template_pair_stack

        if template_config["template_pointwise_attention"]["enabled"]:
            template_pa_config = template_config["template_pointwise_attention"]
            template_pointwise_att = TemplatePointwiseAttention(
                c_t = template_pa_config["d_template"],
                c_z = template_pa_config["d_pair"],
                c_hidden = template_pa_config["d_hid"],
                no_heads = template_pa_config["num_heads"],
                inf = template_pa_config["inf"],
                comm_dtype = dtype,
                attn_o_bias = False,
                dtype = model.dtype
            )

            template_pointwise_att.eval()
            template_pointwise_att_params = model.template_pointwise_att.state_dict()
            template_pointwise_att_params = unifold2foldacc(template_pointwise_att_params, template_pointwise_att.state_dict())
            template_pointwise_att.load_state_dict(template_pointwise_att_params)
            template_pointwise_att = template_pointwise_att.to(device=device, dtype=model.dtype)

            dummy_inputs = (
                torch.randn(1, 4, dummy_msa_size, dummy_seq_len, template_pointwise_att.c_t).to(device=device, dtype=model.dtype), # t
                torch.randn(1, dummy_seq_len, dummy_seq_len, template_pointwise_att.c_z).to(device=device, dtype=model.dtype), # z
                torch.ones(1, 4).to(device=device, dtype=model.dtype), # mask
                dummy_chunk_size # chunk_size
            )

            template_pointwise_att = optimize_module(template_pointwise_att, dummy_inputs=dummy_inputs, 
                enable_disc=enable_disc, enable_amp=enable_amp, enable_trace=enable_trace, 
                dtype=dtype, save_dir=save_dir, load_dir=load_dir, name="template_pointwise_att", device=device)

            model.template_pointwise_att = template_pointwise_att
    
    if enable_low_mem:
        # ---------------------------- optimize embedder ------------------------------
        input_embedder_config = model_config["model"]["input_embedder"]
        input_embedder = InputEmbedder(
            **input_embedder_config,
            use_chain_relative=model_config["model"]["is_multimer"],
            low_mem=enable_low_mem,
            chunk_size=model_config["globals"]["chunk_size"],
            dtype=model.input_embedder.linear_relpos.weight.dtype
        )

        input_embedder.eval()
        input_embedder_params = model.input_embedder.state_dict()
        input_embedder_params = unifold2foldacc(input_embedder_params, input_embedder.state_dict())
        input_embedder.load_state_dict(input_embedder_params)
        input_embedder = input_embedder.to(device=device, dtype=model.input_embedder.linear_relpos.weight.dtype)

        model.input_embedder = input_embedder

        pair_embedder_config = model_config["model"]["template"]["template_pair_embedder"]
        pair_embedder = TemplatePairEmbedder(
            **pair_embedder_config,
            low_mem=enable_low_mem,
            chunk_size=model_config["globals"]["chunk_size"],
        )

        pair_embedder.eval()
        pair_embedder_params = model.template_pair_embedder.state_dict()
        pair_embedder.load_state_dict(pair_embedder_params)
        pair_embedder = pair_embedder.to(device=device, dtype=[p for p in model.template_pair_embedder.parameters()][0].dtype)

        model.template_pair_embedder = pair_embedder

        # ---------------------------- optimize structure module ------------------------------
        structure_module_config = model_config["model"]["structure_module"]
        inv_point_attn = InvariantPointAttention(
            d_single=structure_module_config["d_single"],
            d_pair=structure_module_config["d_pair"],
            d_hid=structure_module_config["d_ipa"],
            num_heads=structure_module_config["num_heads_ipa"],
            num_qk_points=structure_module_config["num_qk_points"],
            num_v_points=structure_module_config["num_v_points"],
            separate_kv=structure_module_config["separate_kv"],
            bias=structure_module_config["ipa_bias"],
            eps=structure_module_config["epsilon"],
            scatter_input = True,
            gather_output = True,
            dtype = model.structure_module.ipa.linear_q.weight.dtype
        )

        inv_point_attn.eval()
        inv_point_attn_params = model.structure_module.ipa.state_dict()
        inv_point_attn.load_state_dict(inv_point_attn_params)
        inv_point_attn = inv_point_attn.to(device=device, dtype=model.structure_module.ipa.linear_q.weight.dtype)

        model.structure_module.ipa = inv_point_attn

    return model