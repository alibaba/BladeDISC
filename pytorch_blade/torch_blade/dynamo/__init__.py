# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch_blade.dynamo.patch_user_defined
import torch_blade.dynamo.monkey_patch

from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.utils import torchscript

from torch._functorch import compilers
from functorch.compile import min_cut_rematerialization_partition

import torch
import torch_blade
import torch.fx as fx
from typing import Callable

def _disc_compile(fx_g: fx.GraphModule, inps, use_ts=False, is_training=True) -> Callable:
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """
    with compilers._disable_jit_autocast():
        compilers.strip_overloads(fx_g)
        for node in fx_g.graph.nodes:
            if (
                node.target == torch.ops.aten._to_copy
                and len(node.args) == 1
                and len(node.kwargs) == 1
                and "dtype" in node.kwargs
            ):
                node.target = torch.ops.aten.to
            if node.target == torch.ops.prims.div:
                node.target = torch.ops.aten.div
            if node.target == torch.ops.aten.alias:
                node.target = torch.ops.aten.clone
            if node.target == torch.ops.prims.var:
                node.target = torch.ops.aten.var
            if node.target == torch.ops.prims.sum:
                node.target = torch.ops.aten.sum
            if node.target == torch.ops.prims.convert_element_type:
                node.target = torch.ops.aten.to
            if node.target == torch.ops.aten.view:
                node.target = torch.ops.aten.reshape

        for node in fx_g.graph.nodes:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type
                new_kwargs[k] = v
            node.kwargs = new_kwargs

        fx_g.graph.lint()
        fx_g.recompile()
        f = torch.jit.script(fx_g)
        torch._C._jit_pass_remove_mutation(f.graph)
        if not is_training:
            # evaluation will const propergate some ops into constant tensors,
            # the const tensor cased some unepxected error while executing with fake Tnesor
            f = torch.jit.freeze(f.eval())
        if use_ts:
            return f
        cfg = torch_blade.Config()
        cfg.disable_optimization_for_inference = is_training
        with cfg:
            f = torch_blade.optimize(f, True, tuple(inps))
        with open('aot_disc.py', 'a') as writer:
            writer.write(str(f.forward.code) + '\n')

    return f

@compilers.make_boxed_compiler
def disc_compile(fx_g: fx.GraphModule, inps, use_ts=False) -> Callable:
    return _disc_compile(fx_g, inps, use_ts=False)

def disc(fx_g: fx.GraphModule, inps) -> Callable:
    scripted = torchscript(fx_g, inps)
    torch._C._jit_pass_remove_mutation(scripted.graph)
    f = torch.jit.freeze(scripted.eval())
    cfg = torch_blade.Config()
    cfg.disable_optimization_for_inference = False
    with cfg:
        f = torch_blade.optimize(f, True, tuple(inps))
    return f

@compilers.make_boxed_compiler
def disc_compile_ts(fx_g: fx.GraphModule, inps, use_ts=False) -> Callable:
    return _disc_compile(fx_g, inps, use_ts=True)

def _get_disc_decomp():
    from torch._decomp import get_decompositions
    aten = torch.ops.aten
    decompositions_dict = get_decompositions(
        [
            aten.var_mean,
            aten._adaptive_avg_pool2d_backward,
            aten.addcmul,
            aten.avg_pool2d_backward,
            aten.binary_cross_entropy_with_logits,
            aten.clamp_max,
            aten.clamp_min,
            aten.col2im,
            aten.cudnn_batch_norm,
            aten.cudnn_batch_norm_backward,
            aten.detach,
            aten.dot,
            aten.elu,
            aten.elu_backward,
            aten._embedding_bag,
            aten.embedding_dense_backward,
            aten.expand_as,
            aten.eye,
            aten.flip,
            aten._fused_moving_avg_obs_fq_helper,
            aten.gelu,
            aten.gelu_backward,
            aten.glu_backward,
            aten.grid_sampler_2d,
            aten.hardsigmoid,
            aten.hardsigmoid_backward,
            aten.hardswish,
            aten.hardswish_backward,
            aten.hardtanh,
            aten.hardtanh_backward,
            aten.im2col,
            aten.index_add,
            aten.index_add_,
            aten.index_select,
            aten.l1_loss,
            aten.leaky_relu,
            aten.leaky_relu_backward,
            aten.linalg_vector_norm,
            aten.logit,
            aten.logit_backward,
            aten._log_softmax,
            aten._log_softmax_backward_data,
            aten.logsumexp.default,
            aten.max_pool2d_with_indices_backward,
            aten.mse_loss,
            aten.mse_loss_backward,
            aten.mv,
            aten.narrow,
            aten.native_batch_norm,
            aten.native_batch_norm_backward,
            aten.native_dropout_backward,
            aten.native_group_norm,
            aten.native_group_norm_backward,
            aten.native_layer_norm,
            aten.native_layer_norm_backward,
            aten.new_empty,
            aten.new_full,
            aten.new_ones,
            aten.nll_loss_backward,
            aten.nll_loss_forward,
            aten.norm,
            aten.reflection_pad2d_backward,
            aten._reshape_alias,
            aten.select_backward,
            aten.select_scatter,
            aten.sigmoid_backward,
            aten.silu_backward,
            aten.slice_backward,
            aten.sgn,
            aten.std_mean.correction,
            aten._softmax,
            aten._softmax_backward_data,
            aten.stack,
            aten.t,
            aten.tanh_backward,
            aten.threshold_backward,
            aten.transpose.int,
            aten.tril.default,
            aten.upsample_bilinear2d.vec,
            aten.upsample_nearest2d_backward,
            aten._unsafe_view,
        ]
    )
    return decompositions_dict

aot_disc = aot_autograd(
    # these are taken from memory_efficient_fusion()
    fw_compiler=disc_compile,
    bw_compiler=disc_compile,
    # NB: lambda here is to delay import of inductor
    decompositions=_get_disc_decomp(),
    partition_fn=min_cut_rematerialization_partition)


aot_disc_debug = aot_autograd(
    # these are taken from memory_efficient_fusion()
    fw_compiler=disc_compile_ts,
    bw_compiler=disc_compile_ts,
    # NB: lambda here is to delay import of inductor
    decompositions=_get_disc_decomp(),
    partition_fn=min_cut_rematerialization_partition)


register_backend(name="disc", compiler_fn=disc)
register_backend(name="aot_disc", compiler_fn=aot_disc)
register_backend(name="aot_disc_debug", compiler_fn=aot_disc_debug)
