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

import itertools
import torch


def _check_list_in_node(node):
    # todo: may move these logic to c++ when necessary.
    for val in itertools.chain(node.input_list(), node.output_list()):
        if type(val.type()) == torch._C.ListType:
            return True
    return False


def _aten_add(node):
    # two kinds of schema
    # 1. aten::add(a, b)
    # 2. aten::add(a, b, alpha)
    if _check_list_in_node(node):
        return False
    return True


def _aten_mul(node):
    if _check_list_in_node(node):
        return False
    return True


def _aten_eq(node):
    if _check_list_in_node(node):
        return False

    inps = node.input_list()
    if not type(inps[0].type()) == type(inps[1].type()):
        return False

    return True


_handler_mapping = {
    "aten::add": _aten_add,
    "aten::eq" : _aten_eq,
    "aten::mul": _aten_mul
}


def check_graph_with_rules(graph):
    """
        Check whether a TorchScript graph should be transformed to onnx.
        True for should and False for should not.
    """
    for node in graph.node_list():
        node_kind = node.kind()
        if node_kind in _handler_mapping:
            if not _handler_mapping[node_kind](node):
                return False
    return True
