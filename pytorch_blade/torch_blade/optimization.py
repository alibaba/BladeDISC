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

import torch
from torch_blade.config import Config, OptPipelines
from torch_blade.exporter import export
from torch_blade.pass_manager import _optimize_common

__all__ = ['optimize']


def _recursive_optimize(model):
    cfg = Config.get_current_context_or_new()
    if isinstance(model, torch.jit.ScriptModule):
        optimized_c_module = _optimize_common(model._c)
        model._reconstruct(optimized_c_module)
        optimizaiton_func = OptPipelines.pipelines[cfg.optimization_pipeline]
        optimizaiton_func(model)
    else:
        assert not cfg.dynamic_tuning_shapes, \
            "TensorRT Dynamic shape is currently not supported for models" \
            "that can not been exported to one torchscript."
        for _, m in model.named_children():
            _recursive_optimize(m)

@torch.no_grad()
def _optimize(model, allow_tracing, model_inputs):
    optimized_model = export(model, allow_tracing, model_inputs)

    assert isinstance(optimized_model, torch.nn.Module),\
        'Currently, input module of optimization process must be in type of torch.nn.Module.'

    # todo(bohua.cbh): Support more kinds of optimization algorithms.
    _recursive_optimize(optimized_model)

    return optimized_model


@torch.no_grad()
def _static_optimize(model, allow_tracing=None, model_inputs=None):
    """
    Optimize the inference process of a PyTorch model with static shapes.
    The whole model (or part of its submodules) will be exported to
    TorchScript to obtain its inference graph firstly.
    Then the specified optimization measure will be applied to it.

    This function is experimental, it would do more optimization,
    such as static shape const-folding.
    We would evaluate in a while whether to retain it.
    That depends on whether it's more efficient than
    the common optimization without static shape const-folding.

    Args:
        model (torch.nn.Module): PyTorch model to be optimized.

        allow_tracing (Optional[List[str]] or bool):
            Contains names of submodules that will be replaced with
            corresponding tracing module. Tracing is always more friendly to TorchScript export as it has less grammar
            limitations. However it can only correctly records functions and modules which are not data dependent.
            (More details can be seen in https://pytorch.org/docs/stable/jit.html#torch.jit.trace) Be sure that the
            specified submodules DO NOT contain any such codes. If it is set to True, the whole model will be exported
            through `torch.jit.trace`.

        model_inputs (Optional[Tuple[Any]] or torch.Tensor):
            Inputs of the model, this variable must be provided.

    return:
        optimized_model(torch.nn.Module):
            An optimized PyTorch model that works only on the static shapes of the given model inputs.

    """
    assert model_inputs is not None, "Static shape specific optimization need model inputs"
    cfg = Config.get_current_context_or_new().clone()
    cfg.enable_static_shape = True
    with cfg:
        return _optimize(model, allow_tracing, model_inputs)


@torch.no_grad()
def optimize(model, allow_tracing=None, model_inputs=None):
    """
    Optimize the inference process of a PyTorch model. The whole model (or part of its submodules) will be exported to
    TorchScript to obtain its inference graph firstly. Then the specified optimization measure will be applied to it.

    Args:
        model (torch.nn.Module): PyTorch model to be optimized.

        allow_tracing (Optional[List[str]] or bool):
            Contains names of submodules that will be replaced with
            corresponding tracing module. Tracing is always more friendly to TorchScript export as it has less grammar
            limitations. However it can only correctly records functions and modules which are not data dependent.
            (More details can be seen in https://pytorch.org/docs/stable/jit.html#torch.jit.trace) Be sure that the
            specified submodules DO NOT contain any such codes. If it is set to True, the whole model will be exported
            through `torch.jit.trace`.

        model_inputs (Optional[Tuple[Any]] or torch.Tensor):
            Inputs of the model, if allow_tracing is not None,
            this variable must be provided.

    return:
        optimized_model(torch.nn.Module): Optimized PyTorch model.
    """
    return _optimize(model, allow_tracing, model_inputs)
