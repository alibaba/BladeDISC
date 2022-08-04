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
import torch_blade

from torch_blade import tools
from torch_blade.logging import logger
from typing import List, Tuple, Optional


class RecordingObserver(torch.nn.Module):
    r"""
    The module is mainly for records the shape types during runtime
    """
    sizes_t = List[int]
    strides_t = List[int]
    device_t = str
    dtype_t = int
    is_contiguous_t = bool
    requires_grad_t = bool
    type_info_t = Tuple[sizes_t, strides_t, device_t, dtype_t, requires_grad_t, is_contiguous_t]
    type_info: Optional[type_info_t]

    def __init__(self):
        super().__init__()
        self.type_info = None

    @torch.jit.export
    def record(self, x):
        sizes = x.size()
        strides = [x.stride(d) for d in range(x.dim())]
        self.type_info = (sizes, strides, str(x.device), x.dtype, x.requires_grad, x.is_contiguous())
        return x


def jit_add_observer(graph, jit_obs, value, method_name='record'):
    obs_module, attr_name, obs_type = jit_obs
    get_attr = graph.createGetAttr(obs_module, attr_name)
    graph.appendNode(get_attr)
    get_attr.output().setType(obs_type)

    call_method = graph.create('prim::CallMethod')
    # record tensor each time prim::CallMethod was called
    #
    # TODO: support tensorrt dynamic shape,
    # which need min/max/opt profiler shape.
    call_method.s_('name', method_name)
    value.replaceAllUsesWith(call_method.output())
    call_method.addInput(get_attr.output())
    call_method.addInput(value)
    call_method.output().setType(value.type())
    graph.appendNode(call_method)

    get_attr.moveAfter(value.node())
    call_method.moveAfter(get_attr)
    return get_attr, call_method

def _collect_all_tensors(block):
    all_tensors = [out for node in block.nodes() for out in node.outputs()
                   if isinstance(out.type(), torch._C.TensorType)]
    for node in block.nodes():
        for inner_blk in node.blocks():
            all_tensors += _collect_all_tensors(inner_blk)
    return all_tensors

def _add_tensor_observers(graph):
    # Create a new ScriptModule to hold all new temporary observers
    observer_owner = torch.jit.ScriptModule()
    jit_obs_m = graph.addInput()
    jit_obs_m.setType(observer_owner._c._type())

    all_tensors = _collect_all_tensors(graph)

    def _is_normal_tensor(tensor):
        if 'prim::Uninitialized' in tensor.node().kind():
            return False
        if 'prim::Loop' in tensor.node().kind():
            return False
        if 'prim::If' in tensor.node().kind():
            return False
        if len(list(tensor.uses())) == 0:
            return False
        return True

    all_tensors = [t for t in all_tensors if _is_normal_tensor(t)]
    observers = []
    for idx, tensor in enumerate(all_tensors):
        attr_name = "observer_{}".format(idx)
        # RecordingObserver is creating from python could be derivated easily
        obs = torch.jit.script(RecordingObserver())
        observer_owner._c._register_attribute(attr_name, obs._c._type(), obs)
        obs_type = obs._c._type()
        get_attr, call_method = jit_add_observer(graph, (jit_obs_m, attr_name, obs_type), tensor)
        observers.append((attr_name, tensor, get_attr, call_method))

    return observer_owner, observers

def _infer_type_from_observer(graph, observer_owner, observers):
    for attr_name, tensor, get_attr, call_method in observers:
        obs = observer_owner.__getattr__(attr_name)
        type_info = obs.type_info
        # if type_info is None,
        # there are none shapes recorded for this tensor
        if type_info is not None:
            tools.set_value_type(tensor, *type_info)
        else:
            pass
        call_method.output().replaceAllUsesWith(tensor)
        call_method.destroy()
        get_attr.destroy()
    graph.eraseInput(len(graph.input_list()) - 1)

def record_shape_by_tracing(module, inputs, graph=None):
    # Note: the input module forward graph must be inlined
    # TODO: only the top level block nodes was record
    graph = graph or module.forward.graph
    torch_blade.pass_manager._jit_pass_remove_nograd(graph)
    g_inps = graph.input_list()
    g_inps = g_inps[1:]
    for val, ival in zip(g_inps, inputs):
        if (isinstance(ival, torch.Tensor)):
            assert (isinstance(val.type(), torch._C.TensorType))
            val.inferTypeFrom(ival)

    observer_owner, observers = _add_tensor_observers(graph)
    module.create_method_from_graph('_torch_blade_record_tensor', graph)
    try:
        _ = module._torch_blade_record_tensor(*inputs, observer_owner)
    except Exception as e:
        logger.error(
            str(e) + '\n' + 'Some Tensors\' shapes are not recorded.'
        )
    _infer_type_from_observer(graph, observer_owner, observers)
    module.unsafe_remove_method('_torch_blade_record_tensor')
    torch_blade.pass_manager._jit_pass_dce_during_lower_to_trt(graph)
    torch_blade.pass_manager._jit_pass_lint(graph)

    def gather_loops(outer_block):
        loops = [n for n in outer_block.nodes() if 'prim::Loop' in n.kind()]
        for node in outer_block.nodes():
            for blk in node.blocks():
                loops += gather_loops(blk)

        return loops

    loops = gather_loops(graph)
    # The following codes were used to set tracing shape/type to block inputs
    for loop in loops:
        loop_blk = next(loop.blocks())
        blk_inps = [inp for inp in loop_blk.inputs()][1:]
        loop_inps = loop.input_list()[2:]
        for blk_inp, loop_inp in zip(blk_inps, loop_inps):
            blk_inp.setTypeAs(loop_inp)
