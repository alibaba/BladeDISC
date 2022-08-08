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
import torch.nn as nn

import logging

from foldacc.optimization.distributed.utils import (
    Gather_save, Scatter_save, AlltoAll_save,
    Gather_load, Scatter_load, AlltoAll_load,
)
from foldacc.optimization.utils import print_logger

logger = logging.getLogger("foldacc")

save_map = {
    "Gather": Gather_save,
    "Scatter": Scatter_save,
    "AlltoAll": AlltoAll_save
}

load_map = {
    "Gather": Gather_load,
    "Scatter": Scatter_load,
    "AlltoAll": AlltoAll_load,
    "FoldAccGather": Gather_load,
    "FoldAccScatter": Scatter_load,
    "FoldAccAlltoAll": AlltoAll_load
}


def _replace_pythonop(graph, node, saved_module, name):
    ginputs = [inp for inp in graph.inputs()]
    m_self = ginputs[0]

    attr = graph.create('prim::GetAttr')
    attr.addInput(m_self)
    attr.s_('name', name)
    attr.output().setType(saved_module._c._type())
    graph.appendNode(attr)
    attr.moveBefore(node)

    noutputs = [out for out in node.outputs()]
    ninputs = [inp for inp in node.inputs()]

    call_method = graph.create('prim::CallMethod')
    call_method.s_('name', 'forward')
    call_method.addInput(attr.output())
    call_method.addInput(ninputs[0])
    call_method.output().setType(noutputs[0].type())
    graph.appendNode(call_method)
    call_method.moveAfter(attr)

    cout = [out for out in call_method.outputs()][0]

    noutputs[0].replaceAllUsesWith(cout)

    node.destroy()

def _replace_saveop(graph, pair_node, load_module, name):
    if len(pair_node) == 3:
        save_attr, save_op, save_method = pair_node
    elif len(pair_node) == 2:
        save_attr, save_method = pair_node

    ginputs = [inp for inp in graph.inputs()]
    m_self = ginputs[0]

    attr = graph.create('prim::GetAttr')
    attr.addInput(m_self)
    attr.s_('name', name)
    attr.output().setType(load_module._c._type())
    graph.appendNode(attr)
    attr.moveBefore(save_attr)

    noutputs = [out for out in save_method.outputs()]
    ninputs = [inp for inp in save_method.inputs()]

    call_method = graph.create('prim::CallMethod')
    call_method.s_('name', 'forward')
    graph.appendNode(call_method)
    call_method.addInput(attr.output())
    call_method.addInput(ninputs[1])
    call_method.output().setType(noutputs[0].type())
    call_method.moveAfter(save_method)

    cout = [out for out in call_method.outputs()][0]

    noutputs[0].replaceAllUsesWith(cout)

    save_method.destroy()
    if len(pair_node) == 3:
        save_op.destroy()

    try:
        save_attr.destroy()
    except:
        return

def _get_submodules(model):
    sub_modules = []
    for name, module in model.named_modules():
        if (not module._c._has_method("forward")):
            continue

        if module == model:
            sub_modules.append(module)
        else:
            sub_modules.extend(_get_submodules(module))
    
    return sub_modules

def _get_nodes(blocks):
    all_nodes = [node for node in blocks.nodes()]
    for node in blocks.nodes():
        for inner_blk in node.blocks():
            all_nodes += _get_nodes(inner_blk)

    return all_nodes

def convert_save_model(model):
    sub_modules = _get_submodules(model)

    count = 0
    for module in sub_modules:
        nodes = _get_nodes(module.graph)
        pynodes = []
        for node in nodes:
            if node.kind() != "prim::PythonOp":
                continue
            if node.pyname() in ["Gather", "Scatter", "AlltoAll"]:
                pynodes.append(node)

        for i, node in enumerate(pynodes):
            opname = node.pyname()
            scalar_args = node.scalar_args()
            
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1
            
            saved_module = torch.jit.script(save_map[node.pyname()](*scalar_args, world_size))
            save_name = f"foldacc_save_{node.pyname()}_{count}"
            module._c._register_attribute(save_name, saved_module._c._type(), saved_module)

            _replace_pythonop(module.graph, node, saved_module, save_name)

            print_logger(logger.debug, f"Convert {opname} to saveable op[{save_name}] with args={scalar_args}")

            count += 1

    return model

def convert_load_model(model):
    sub_modules = _get_submodules(model)
    # load
    count = 0
    for module in sub_modules:
        nodes = _get_nodes(module.graph)
        pair_nodes = []

        visited_nodes = []

        for node in nodes:
            if node in visited_nodes:
                continue
            node_outputs = [o for o in node.outputs()]
            if node.kind() == "prim::GetAttr" and "foldacc_save" in node.s("name"):
                pair_node = [node]

                is_inline = True
                for nn in nodes:
                    # getattr op
                    inputs = [inp for inp in nn.inputs()]
                    if len(inputs) == 1 and node_outputs[0] == inputs[0]:
                        pair_node.append(nn)
                        is_inline = True
                        break
                    
                    if len(inputs) == 2 and node_outputs[0] == inputs[0]:
                        pair_node.append(nn)
                        is_inline = False
                        break
                
                if len(pair_node) != 2:
                    continue
                
                if not is_inline:
                    pair_nodes.append(pair_node)
                    continue

                outputs = [out for out in pair_node[1].outputs()]
                for nn in nodes:
                    # callmethod op
                    inputs = [inp for inp in nn.inputs()]
                    if len(inputs) == 2 and outputs[0] == inputs[0]:
                        pair_node.append(nn)
                        break
                
                if len(pair_node) != 3:
                    continue

                pair_nodes.append(pair_node)
                visited_nodes.extend(pair_node)

            elif node.kind() == "prim::Constant" and len(node_outputs) == 1 and hasattr(node_outputs[0].type(), "name") and node_outputs[0].type().name() in [
                'FoldAccGather', 'FoldAccScatter', 'FoldAccAlltoAll'
            ]:
                pair_node = [node]
                outputs = [out for out in node.outputs()]

                for nn in nodes:
                    # forward op
                    inputs = [inp for inp in nn.inputs()]
                    if len(inputs) == 2 and outputs[0] == inputs[0]:
                        pair_node.append(nn)
                        break
                
                if len(pair_node) != 2:
                    continue
                
                pair_nodes.append(pair_node)
                visited_nodes.extend(pair_node)

        for i, node in enumerate(pair_nodes):
            if len(node) == 3:
                save_attr, save_op, save_method = node
            elif len(node) == 2:
                save_attr, save_method = node
            
            random_input = torch.ones(128, 128).cuda()
            
            if save_attr.kind() == "prim::GetAttr":
                save_name = save_attr.s('name')
                op_name = "_".join(save_attr.s("name").replace("foldacc_save_", "").split("_")[:-1])
            else:
                save_name = list(save_attr.output_list())[0].type().name() + "_" + list(save_attr.output_list())[0].debugName()
                op_name = list(save_attr.output_list())[0].type().name()
    
            if "Gather" in save_name or "Scatter" in save_name:
                if save_attr.kind() == "prim::GetAttr":
                    comm_cls = None
                    for n, m in module.named_modules():
                        if n.split(".")[-1] == save_attr.s("name"):
                            comm_cls = m
                    if len(node) == 3:
                        dim = comm_cls.op._get_method("__getstate__")()[0]
                    else:
                        dim = comm_cls.dim
                else:
                    dim = list(save_attr.output_list())[0].toIValue()._get_method("__getstate__")()[0]
                load_module = torch.jit.trace(load_map[op_name](dim), (random_input,))
                scalar_args = [dim]
            else:
                if save_attr.kind() == "prim::GetAttr":
                    comm_cls = None
                    for n, m in module.named_modules():
                        if n.split(".")[-1] == save_attr.s("name"):
                            comm_cls = m
                    if len(node) == 3:
                        dim = comm_cls.op._get_method("__getstate__")()[:-1]
                        in_dim, out_dim = dim
                    else:
                        in_dim = comm_cls.in_dim
                        out_dim = comm_cls.out_dim                        
                else:
                    dim = list(save_attr.output_list())[0].toIValue()._get_method("__getstate__")()[:-1]
                    in_dim, out_dim = dim
                load_module = torch.jit.trace(load_map[op_name](in_dim, out_dim), (random_input,))
                scalar_args = [in_dim, out_dim]
        
            load_name = f"foldacc_load_{op_name}_{count}"
            module._c._register_attribute(load_name, load_module._c._type(), load_module)

            _replace_saveop(module.graph, node, load_module, load_name)

            print_logger(logger.debug, f"Convert {save_name} to runable op[{load_name}] with args={scalar_args}")

            count += 1
    return model