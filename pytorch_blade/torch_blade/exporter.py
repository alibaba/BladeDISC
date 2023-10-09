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

import copy
import types
import io
import functools
from typing import Tuple, List

import torch
import torch.nn as nn

import torch_blade.pass_manager as pm
from torch_blade import jit_pass_propagate_input_shapes, jit_pass_constant_propagation
from torch_blade.config import Config
from torch_blade.logging import logger
from torch_blade.tools import set_tensor_shape
from torch_blade.tools.shape_inference import record_shape_by_tracing

__all__ = ['export', 'match_submodules']

def _record_shape_information(s_module, inputs):
    if inputs is None:
        return
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs, )
    # TODO:
    # handle the exception raised by record_shape_by_tracing,
    # such as inputs device mismatch
    record_shape_by_tracing(s_module._c, inputs)

def _script_module_preprocess(s_module, inputs, input_dims=[]):
    graph = s_module._c.forward.graph
    torch._C._jit_pass_inline(graph)
    pm._jit_pass_hack_cpu_device(graph)
    pm._jit_pass_hack_gpu_device(graph)

    cfg = Config.get_current_context_or_new()
    for jit_pass in cfg.customize_jit_passes:
        jit_pass(graph)

    cfg = Config.get_current_context_or_new()
    if not cfg.disable_optimization_for_inference:
        # TODO(tanyo):
        # It will record tensor information, such as ranks, data types,
        # and devices, that are useful for analysis and optimizations
        # by tracing with auxiliary inputs.
        # Should be deprecated once shape analysis is complete and robust
        _record_shape_information(s_module, inputs)
        return

    for idx, input in enumerate(graph.inputs()):
        # skip the 1th self input value
        is_tensor = input.type().isSubtypeOf(torch._C.TensorType.get())
        if not is_tensor: continue
        inp = inputs[idx-1]
        dim = len(inp.size())
        if isinstance(dim, int):
            set_tensor_shape(input, list(inp.size()))
        inp_typ = input.type()
        inp_typ = inp_typ.with_dtype(inp.dtype)
        input.setType(inp_typ.with_device(inp.device))
    if cfg.disable_optimization_for_inference:
        # TODO(note): using a simple constant propagation pass on training
        jit_pass_constant_propagation(graph)
    else:
        torch._C._jit_pass_constant_propagation(graph)
    jit_pass_propagate_input_shapes(graph)

def _deep_copy_script_module(model):

    # store the namelist of modules that do not have __deepcopy__ function
    _no_deepcopy_namelist = set()

    def _delete_deepcopy_func(mod):
        for name, m in mod.named_modules():
            if isinstance(m, torch.jit.ScriptModule):
                if name in _no_deepcopy_namelist:
                    del m.__dict__['__deepcopy__']
                    _no_deepcopy_namelist.remove(name)

    def _add_deepcopy_func(mod):
        for name, m in mod.named_modules():
            if isinstance(m, torch.jit.ScriptModule):
                if '__deepcopy__' not in m.__dict__:
                    m.__dict__['__deepcopy__'] = types.MethodType(_deepcopy_replacement, m)
                    _no_deepcopy_namelist.add(name)

    def _deepcopy_replacement(self, memo):
        try:
            buffer = io.BytesIO()
            torch.jit.save(self, buffer)
            buffer.seek(0)
            new_instance = torch.jit.load(buffer)
        except Exception as e:
            logger.info(str(e) + '\n' +
                        'Fail to deepcopy ScriptModule {}'.format(self))
            raise
        else:
            return new_instance

    _add_deepcopy_func(model)
    _model = copy.deepcopy(model)
    _delete_deepcopy_func(model)

    return _model


def _deepcopy(model):
    _model = _deep_copy_script_module(model)
    return _model


def match_submodules(model, types=()):
    """
    Iterate over the submodules and return name list of those whose type is matched in type_list.

    Args:
        model (nn.Module): Model to be matched.
        types (iterable): Specify the types of the submodule to match.
    return:
        name_list (List[str]): Name list of the target submodules. Empty List will be returned if nothing is matched.
    """
    # todo(bohua.cbh): add match through string.
    def helper(mod, prefix=''):
        for name, sub_mod in mod.named_children():
            curr_name = prefix + '.' + name if prefix != '' else name
            if any(isinstance(sub_mod, t) for t in types):
                name_list.append(curr_name)
            helper(sub_mod, curr_name)

    types = [t for t in types]
    name_list = []
    helper(model)
    return name_list


@torch.no_grad()
def export(model, allow_tracing=None, model_inputs=None):
    """
    Given a PyTorch model, we first replace submodules (specified in allow_tracing) with a tracing torchscript.
    Then we export torchscript through torch.jit.script on the top level of the model.

    Args:
        model (nn.Module): Model to be optimized.

        allow_tracing (Optional[List[str]] or bool): 
            Contains submodule names that will be replaced with tracing module.
            If it is set to True, the whole model will be exported through `torch.jit.trace`.

        model_inputs (Optional[Tuple[Any]] or torch.Tensor): Inputs of the model.
    return:
        scripted_module: 
            if torchscript is exported successfully, then it will be returned.
            else return the original model.
    """
    def wrap_forward(name, func):
        """
        Wrap forward function of nn.Module in order to record its input
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                # todo(bohua.cbh): we may flatten kwargs using inspect.getfullargspec
                # https://docs.python.org/3.8/library/inspect.html#inspect.getfullargspec
                submodule_inputs[name] = (args, kwargs, True)
            else:
                submodule_inputs[name] = args
            return func(*args, **kwargs)

        return wrapper

    def wrap_model(mod, prefix=''):
        """
        Iterate through all the submodules of the input model and wrap all forward functions.
        """
        for name, sub_mod in mod.named_children():
            curr_name = prefix + '.' + name if prefix != '' else name
            sub_mod.forward = wrap_forward(curr_name, sub_mod.forward)
            wrap_model(sub_mod, curr_name)

    def replace_module_with_script(model, prefix=''):
        for name, sub_mod in model.named_children():
            curr_name = prefix + '.' + name if prefix != '' else name
            try:
                scripted_mod = torch.jit.script(sub_mod)
                x = submodule_inputs.get(curr_name, None)
                if ((x is not None) and (x[-1] is not True)):
                    # **kwargs in inputs is currently not supported
                    _script_module_preprocess(scripted_mod, x)
            except Exception as e:
                logger.warning(str(e) + '\n' +
                               'Submodule {} is unsuccessfully exported by torch.jit.script'.format(curr_name))
                replace_module_with_script(sub_mod, curr_name)
            else:
                logger.info('Submodule {} is successfully exported by torch.jit.script'.format(curr_name))
                setattr(model, name, scripted_mod)


    def replace_module_with_trace(mod, tracing_list, prefix=''):
        """
        Replace a submodule with tracing torchscript when its name is matched in allow_tracing.
        """
        for name, sub_mod in mod.named_children():
            curr_name = prefix + '.' + name if prefix != '' else name
            if curr_name in tracing_list:
                assert curr_name in submodule_inputs, \
                    '{} is not matched in the name list of submodules'.format(curr_name)
                x = submodule_inputs[curr_name]
                assert x[-1] is not True, \
                    '**kwargs is not currently supported as inputs to modules that will be exported by torch.jit.trace'

                try:
                    traced_module = torch.jit.trace(sub_mod, x, check_trace=False, strict=False)
                    _script_module_preprocess(traced_module, x)
                except Exception as e:
                    logger.warning(
                        str(e) + '\n' + 'Fail to convert module {} with tracing module.\n'
                                        'We will leave it unchanged and export it through torch.jit.script '
                                        'on top level of the model.'
                        .format(curr_name)
                    )
                else:
                    logger.info(
                        'Replacing module {} with traced module'.format(curr_name)
                    )
                    setattr(mod, name, traced_module)
            else:
                replace_module_with_trace(sub_mod, tracing_list, curr_name)

    # Records all submodule names and corresponding input
    # top level module is not recorded here
    submodule_inputs = {}

    assert isinstance(model, nn.Module), \
        'Model must be nn.Module, nn.DataParallel or nn.parallel.DistributedDataParallel'
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    with Config.get_current_context_or_new() as cfg:
        # If disable_optimization_for_inference is True, we will not need deep copy of the model.
        if not cfg.disable_optimization_for_inference:
            _model = _deepcopy(model)
        else:
            _model = model

    if allow_tracing:
        assert model_inputs is not None, 'model_inputs can not be None when use torch.jit.trace'
        assert isinstance(model_inputs, (Tuple, torch.Tensor)), 'Model_inputs must be Tuple or torch.Tensor'

        if isinstance(model_inputs, torch.Tensor):
            model_inputs = (model_inputs,)

        if allow_tracing is True:
            logger.info('Export the whole model through torch.jit.trace')
            try:
                traced_model = torch.jit.trace(_model, model_inputs, check_trace=False, strict=False)
            except Exception as e:
                logger.warning(str(e) + '\n' + 'Failed! Try to export it through torch.jit.script:\n')
            else:
                logger.info('Done!')
                _script_module_preprocess(traced_model, model_inputs)
                return traced_model

        elif isinstance(allow_tracing, List):
            assert all(isinstance(s, str) for s in allow_tracing)
            logger.info(
                'Submodules listed below will be transformed by torch.jit.trace:\n'
                '{}'.format(allow_tracing)
            )
            wrapped_model = _deepcopy(_model)
            wrap_model(wrapped_model)
            wrapped_model(*model_inputs)
            del wrapped_model
            replace_module_with_trace(_model, allow_tracing, '')
        else:
            raise TypeError('Only List[str] and bool is supported for allow_tracing')
    else:
        logger.info('Export the whole model through torch.jit.script')

    try:
        scripted_model = torch.jit.script(_model)
        _script_module_preprocess(scripted_model, model_inputs)
    except Exception as e:
        logger.warning(str(e) + '\n' + 'Fail to export torchscript on the top level of the model, We will iterate over '
                                       'the submodules and replace those that can be successfully exported by the '
                                       'torch.jit.script')
        replace_module_with_script(_model)
        return _model
    else:
        logger.info('Done!')
        return scripted_model
