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
import copy
import logging
import os
from shutil import copyfile
from functools import partial

from foldacc.foldacc import Config, optimize
from foldacc.optimization.utils import print_logger, wrap_model_inputs
from foldacc.optimization.distributed.convert import convert_save_model, convert_load_model

from foldacc.model.modules.ops import (
    Linear,
    Transition,
    Attention,
)
from foldacc.model.modules.msa import (
    MSARowAttentionWithPairBias,
    MSAAttention,
    MSAColumnAttention,
    MSAColumnGlobalAttention,

)
from foldacc.model.modules.outer_product_mean import (
    OuterProductMean
)
from foldacc.model.modules.triangle import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttention,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from foldacc.model.modules.template import (
    TemplatePairStackBlock,
    TemplatePointwiseAttention
)
from foldacc.model.utils import(
    attn_trace_module, attn_script_module,
    global_attn_trace_module, global_attn_script_module,
    triangle_attn_trace_module, triangle_attn_script_module,
    opm_script_module,
    trans_trace_module, trans_script_module,
    triangle_mul_script_module,
    template_script_module,
    template_attn_trace_module, template_attn_script_module
)

logger = logging.getLogger("foldacc")

def generate_config(enable_disc=True, enable_amp=True, enable_trace=True, dtype=torch.float, device="cuda:0"):
    optimize_config = Config()

    optimize_config.precision = dtype
    optimize_config.enable_auto_mix_precision = enable_amp
    optimize_config.check_tolerance = 1e3
    optimize_config.add_low_precision_module([Attention, Linear])

    optimize_config.add_chunking_module(
        MSAAttention, '_chunk', attn_trace_module, attn_script_module)
    optimize_config.add_chunking_module(
        MSARowAttentionWithPairBias, '_chunk', attn_trace_module, attn_script_module)
    optimize_config.add_chunking_module(
        MSAColumnGlobalAttention, '_chunk', global_attn_trace_module, global_attn_script_module)
    optimize_config.add_chunking_module(
        TriangleAttentionStartingNode, '_chunk', triangle_attn_trace_module, triangle_attn_script_module)
    optimize_config.add_chunking_module(
        TriangleAttentionEndingNode, '_chunk', triangle_attn_trace_module, triangle_attn_script_module)
    optimize_config.add_chunking_module(
        TriangleMultiplicationOutgoing, '_chunk', TriangleMultiplicationOutgoing._low_mem_mul, triangle_mul_script_module)
    optimize_config.add_chunking_module(
        TriangleMultiplicationIncoming, '_chunk', TriangleMultiplicationIncoming._low_mem_mul, triangle_mul_script_module)
    optimize_config.add_chunking_module(
        OuterProductMean, '_chunk', OuterProductMean._opm, opm_script_module)
    optimize_config.add_chunking_module(
        Transition, '_chunk', trans_trace_module, trans_script_module)

    optimize_config.add_chunking_module(
        TemplatePointwiseAttention, '_chunk', template_attn_trace_module, template_attn_script_module)
    optimize_config.add_chunking_module(
        TemplatePairStackBlock, 'forward_loop', TemplatePairStackBlock.forward_single, template_script_module)

    optimize_config.enable_bladedisc = enable_disc
    optimize_config.enable_trace = enable_trace

    optimize_config.device = device

    return optimize_config

def optimize_module(
    model, 
    dummy_inputs, 
    optimize_config=None, 
    enable_disc=True, 
    enable_amp=True,
    enable_trace=True,
    dtype=torch.float,
    trace_check_tolerance=1e-5,
    save_dir="tmp",
    load_dir="tmp",
    name="evoformer",
    device="cuda:0"
):
    with torch.no_grad():
        if optimize_config is None:
            optimize_config = generate_config(enable_disc=enable_disc, enable_amp=enable_amp, 
                enable_trace=enable_trace, dtype=dtype, device=device)
        optimize_config.trace_check_tolerance = trace_check_tolerance

        if hasattr(model, "blocks"):
            org_blocks = model.blocks
        else:
            org_blocks = [model]

        other_tag = ""
        if torch.distributed.is_initialized():
            other_tag = f"_{torch.distributed.get_rank()}"

        for i, block in enumerate(org_blocks):
            is_load = False
            # load model
            if load_dir is not None:
                load_path = f'{load_dir}/{name}_{i}{other_tag}.pt'
                if os.path.exists(load_path):
                    optimized_block = torch.jit.load(load_path)
                    optimized_block = convert_load_model(optimized_block)
                    optimized_block = wrap_model_inputs(optimized_block)
                    print_logger(logger.info, f'load model from {load_path}')
                    is_load = True
            
            if not is_load:
                optimize_inputs = copy.deepcopy(dummy_inputs)
                optimized_block = optimize(block, optimize_inputs, optimize_config)

            optimize_inputs = copy.deepcopy(dummy_inputs)
            block_output = optimized_block(*optimize_inputs)
            if isinstance(block_output, torch.Tensor):
                dummy_inputs = tuple([block_output] + list(dummy_inputs[1:]))
            else:
                dummy_inputs = tuple(list(block_output) + list(dummy_inputs[len(block_output):]))

            if hasattr(model, "blocks"):
                model.blocks[i] = optimized_block
            else:
                model = optimized_block

            # save model
            if save_dir is not None and not is_load and enable_trace:
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                    
                    torch.distributed.barrier()
                else:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                save_path = f'{save_dir}/{name}_{i}{other_tag}.pt'
                if os.path.exists(f'{optimize_config.temp_dir}/load_model{other_tag}.pt'):
                    copyfile(f'{optimize_config.temp_dir}/load_model{other_tag}.pt', save_path)
                elif os.path.exists(f'{optimize_config.temp_dir}/save_model{other_tag}.pt'):
                    copyfile(f'{optimize_config.temp_dir}/save_model{other_tag}.pt', save_path)
                else:
                    optimized_block = convert_save_model(optimized_block)
                    torch.jit.save(optimized_block, save_path)
                    optimized_block = convert_load_model(optimized_block)
                print_logger(logger.info, f'save model to {save_path}')
            
            print_logger(logger.info, f"optimize {name} block {i} finish.")

            optimize_config.clean_tmp()

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    return model