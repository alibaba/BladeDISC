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

import functools
from collections import defaultdict

import onnx
import torch
from torch_blade import pass_manager, tools
from torch_blade.config import Config
from torch_blade.logging import logger
from torch_blade.quantization import is_fake_quant_op
from torch_blade.tools import onnx_lower_guard


class OnnxBackendChecker:
    def __init__(self, subgraph, onnx_backend_test_func, backend_name):
        self._graph = subgraph
        self._onnx_backend_test_func = onnx_backend_test_func
        self._backend_name = backend_name

    def _check_concrete_shape(self, graph):
        def is_tensor(val):
            return isinstance(val.type(), torch._C.TensorType)

        all_concrete_inputs = all(
            tools.is_concrete_shape_tensor_type(inp) for inp in graph.inputs()
        )
        all_tensor_outputs = all(
            is_tensor(out) for node in graph.nodes() for out in node.outputs()
        )
        return all_concrete_inputs and all_tensor_outputs

    def _patch_inputs_scalar_types(self, graph, scalar_types):
        """Add scalar type and dim to plain TensorType."""
        assert len(list(graph.inputs())) == len(scalar_types)
        for inp, scalar_type in zip(graph.inputs(), scalar_types):
            inp_type = inp.type()
            if scalar_type and inp_type.isSubtypeOf(torch._C.TensorType.get()):
                new_type = tools.create_tensor_type_from_scalar_type(scalar_type)
                inp.setType(new_type)

    def _record_inputs_scalar_types(self, graph):
        scalar_inputs = []
        for inp in graph.inputs():
            if inp.type().isSubtypeOf(torch._C.NumberType.get()):
                scalar_inputs.append(inp.type())
            else:
                scalar_inputs.append(None)
        return scalar_inputs

    def __call__(self):
        try:
            graph = self._graph
            # use mannual rules to filter the graph
            if not onnx_lower_guard.check_graph_with_rules(graph):
                return False

            # Even though direct non-tensor inputs are forbidden in `appendNode`
            # method, indirect non-tensor may still be introduced by recursive
            # call of `_appendNode` and `_add_graph_input_if_need` method. To be
            # robust here we record these scalar inputs and convert them into
            # tensor types.
            scalar_types = self._record_inputs_scalar_types(graph)
            graph, _ = pass_manager._jit_pass_lower_to_onnx(graph)
            self._patch_inputs_scalar_types(graph, scalar_types)
            if not self._check_concrete_shape(graph):
                return False

            proto = pass_manager._export_onnx(graph, {}, fold_constants=True)
            onnx_model = onnx.load_from_string(proto)
            # pylint: disable=maybe-no-member
            if len(onnx_model.graph.node) == 0:
                node_kinds = [n.kind() for n in self._graph.nodes()]
                logger.warning(f"The subgraph exported from {str(node_kinds)} is empty")
                # TODO: Currently we believe that empty onnx export
                # means the corresponding TorchScript graph has no
                # meanings in a typical inference backend
                # (e.g. contiguous, dropout, detach). If we see
                # counterexamples in the future, we will switch the
                # default to execute such graphs with fallback and use
                # a whitelist to let ops like dropout, contiguous and
                # detach through.
                return True

            supported = self._onnx_backend_test_func(proto)
            if not supported:
                node_kinds = [n.kind() for n in graph.nodes()]
                logger.warning(
                    f"{str(node_kinds)} export to onnx success, but is not supported by "
                    f"the backend: {self._backend_name}"
                )
            return supported
        except Exception as error:
            logger.debug(error)
            return False


class OnnxBackendTestBed:
    """
    Try to test whether it can be converted to the backend for each node in the original graph.
    """

    def __init__(
        self, graph, ignore_device, onnx_backend_test_func, backend_name
    ):
        # the original graph must not be modified
        self._orig_graph = graph
        self._onnx_backend_test_func = onnx_backend_test_func
        self._backend_name = backend_name

        self._ignore_device = ignore_device
        self._unsupported_node_from_orig_graph = set()

        self._current_segment = torch._C.Graph()
        self._current_segment_size = 0
        # TODO: Add some comments
        self._max_segment_size = 1
        self._segment_list = []
        # store whether a view-kind node should be considered as an inplace node.
        self._seen_view_kinds_node = dict()

        # was used to create clone node from original graph -> segment
        self._orig2segment_value_map = dict()
        self._segment2orig_value_map = dict()

        cfg = Config.get_current_context_or_new()
        self._black_list = set(
            [
                "prim::If",
                "prim::Loop",  # add here currently not supported
                "prim::TupleConstruct",
                "prim::DictConstruct",  # onnx not support
                "prim::CallMethod",  # onnx & hie not support
                "aten::empty_like",  # hie not supported
                "aten::sort",  # raise exception when converting to TopK
            ]
            + [op for op in cfg.customize_op_black_list if "@" not in op]
        )

        node_kind_list = defaultdict(list)
        for node in graph.node_list():
            node_kind_list[node.kind()].append(node)

        customize_node_black_list = [
            op for op in cfg.customize_op_black_list if "@" in op
        ]
        self._black_node_list = []
        for node_name in customize_node_black_list:
            try:
                op_kind, node_idx = node_name.split("@")
                node_idx = int(node_idx)
                node = node_kind_list[op_kind][node_idx]
                self._black_node_list.append(node)
            except Exception as error:
                logger.warning(error)

        self._shape_white_list = []
        if cfg.enable_onnx_shape_white_list:
            # Because many shape operations on scalar, which would be placed on CPU device,
            # so the _ignore_device is also enable if enable_onnx_shape_white_list
            self._ignore_device = True
            # TODO(gty): try to find shape computation subgraph automatic
            self._shape_white_list = [
                "aten::view",
                "aten::size",
                "aten::reshape",
                "aten::mul",
                "aten::floor_divide",
                "aten::floordiv",
                "aten::Int",
                "prim::NumToTensor",
            ]
            logger.warning(
                "Enable _ignore_device because of enable_onnx_shape_white_list"
            )

        self._white_list = set(self._shape_white_list + cfg.customize_op_white_list)
        self._view_kinds = set(
            [
                "aten::select",
                "aten::view",
                "aten::slice",
                "aten::expand",
                "aten::expand_as",
            ]
        )

        # todo(bohua.cbh): enable this
        self._fp16_excluded_list = []

    def _is_inplace_kinds(self, node):
        """
        Check whether a view-kind node should be considered as an inplace node.

        let A defines a node whose kind is defined in self._view_kinds, we have:
            A -> A -> aten::add    |    A should not be considered as a inplace node
            A -> A -> aten::add_   |    A should be considered as a inplace node

        Given a view-kind node, we 'build' a tree with the following steps:
        1. mark its inputs as the root_values
        2. iterate through all child nodes who are the consumers of the root_values
        3. For each child node:
            a. If it is not of view-kind or its outputs have no consumers, add it to the tree
               as a leaf.
            b. If it is of view-kind, add it to the tree as a tree node and mark its outputs
               as the root_values, then go back to step 2.

        If there is no inplace node in this tree, then all the view-kind nodes in this tree
        should not be considered as inplace nodes, otherwise they should.
        """
        if node.kind() not in self._view_kinds:
            return False
        if node in self._seen_view_kinds_node:
            return self._seen_view_kinds_node[node]

        def _check_tensor(val):
            is_tensor = val.type().isSubtypeOf(torch._C.TensorType.get())
            is_non_const = val.node().kind() != "prim::Constant"
            return is_tensor and is_non_const

        def _check_all_user(val):
            for u in val.uses():
                new_node = u.user
                if new_node.kind() in self._view_kinds:
                    seen_node.append(new_node)
                    for oup in new_node.output_list():
                        if _check_tensor(oup):
                            _check_all_user(oup)
                else:
                    if new_node.kind().endswith("_"):
                        nonlocal result
                        result = True

        result = False
        seen_node = []
        # start from the input tensor to build the tree
        for inp in node.input_list():
            if _check_tensor(inp):
                _check_all_user(inp)
        # one node may be added twice (e.g. aten::expand_as)
        seen_node = set(seen_node)
        for n in seen_node:
            assert n not in self._seen_view_kinds_node
            self._seen_view_kinds_node[n] = result
        return result

    def _is_inplace(self, node):
        return node.kind().endswith("_") or self._is_inplace_kinds(node)

    @functools.lru_cache(maxsize=None)
    def _is_inplace_safe(self, node):
        """This method is stateful, it cache safe inplace ops flags"""
        # In many models inplace ops formed up a pipeline on the
        # features map.  We would like to find out the simplest case,
        # that an inplace op is the last consumer of its input values.
        # In this case, we could replace the inplace ops with it's
        # corresponding out-of-place version safely.
        #
        # See the following graph:
        #           o ---.               .-> w_
        #                 \             /
        #  o --> o_ --> o_ --> x_ --> y_ ------> v_ --> z_
        #        t      t      t      t      f    f      f   | inplace_safe?
        #
        # Some inplace ops that can be replace with it's correspoding out-of-place version:
        #           o ---.               .-> w_
        #                 \             /
        #  o  --> o  --> o --> x ---> y -------> v_ --> z_
        #         t      t     t      t      f    f      f   | inplace_safe?
        #
        # We cache these inplace ops that is safe during the OnnxBackendTestBed is building,
        # so that it is able to escape the verification of black list

        if node.kind() == "prim::Param":
            # We should never change inplace modifications on graph inputs,
            # because this side-effect may be used outside the graph scope.
            return False
        if not self._is_inplace(node):
            return True

        non_const_inputs = set(
            inp for inp in node.inputs() if inp.node().kind() != "prim::Constant"
        )

        def is_last_user(val):
            for u in val.uses():
                if u.user is node:
                    continue
                if u.user.isAfter(node):
                    return False
            return True

        all_is_last_user = all(is_last_user(inp) for inp in non_const_inputs)
        all_inplace_inps_safe = all(
            self._is_inplace_safe(inp.node()) for inp in non_const_inputs
        )

        is_safe = all_is_last_user and all_inplace_inps_safe
        return is_safe

    def _hit_black_list(self, node):
        is_inplace_safe = self._is_inplace_safe(node)
        is_hit = (node.kind() in self._black_list) or not is_inplace_safe
        is_hit = (node in self._black_node_list) or is_hit
        return is_hit

    def get_unsupported(self):
        return self._unsupported_node_from_orig_graph

    def _clear(self):
        self._segment_list.append(self._current_segment)
        self._current_segment = torch._C.Graph()
        self._current_segment_size = 0
        self._orig2segment_value_map.clear()
        self._segment2orig_value_map.clear()

    def _add_graph_input_if_need(self, old_node):
        # all prim::Constant inputs of node is fused into subgraph if need
        for inp in old_node.inputs():
            if inp in self._orig2segment_value_map:
                continue

            inp_node_kind = inp.node().kind()
            is_const = inp_node_kind == "prim::Constant"
            is_listconstruct = inp_node_kind == "prim::ListConstruct"

            if is_const or is_listconstruct:
                # prim::Constant, prim::ListConstruct
                self._appendNode(inp.node())
                continue

            if is_fake_quant_op(inp_node_kind):
                self._appendNode(inp.node())
                continue

            inp_ = self._current_segment.addInput()
            self._orig2segment_value_map[inp] = inp_
            self._segment2orig_value_map[inp_] = inp
            inp_.copyMetadata(inp)

    def _add_unsupported(self, node):
        logger.debug("Found unsupported: %s" % (node.kind()))
        self._unsupported_node_from_orig_graph.add(node)
        self._clear()

    def appendNode(self, node):
        if self._max_segment_size <= self._current_segment_size:
            # to save conversion time, create a new segment & store the old one
            self._clear()
        if self._fp16_excluded_list:
            for oup in node.output_list():
                if oup.debugName() in self._fp16_excluded_list:
                    self._add_unsupported(node)
                    return False

        if self._hit_black_list(node):
            self._add_unsupported(node)
            return False

        # There are some white_list operations regarded as supported, which were
        # given according to prior knowledge. But they might cause failure when doing
        # onnx2hie exporting. To give users options to use these white_list, we have
        # defined some configuration flags, such as Config.enable_onnx_shape_white_list.
        if node.kind() in self._white_list:
            return True

        # This rule was added because it will introduce non-Tensor outputs of prim::Param
        # as the inputs of the subgraph. But we only accept Tensor as subgraph inputs.
        for inp in node.inputs():
            is_tensor = inp.type().isSubtypeOf(torch._C.TensorType.get())
            if not is_tensor and inp.node().kind() == "prim::Param":
                self._add_unsupported(node)
                return False

        if not self._ignore_device:
            for val in node.input_list() + node.output_list():
                if val.node().kind() == "prim::Constant":
                    continue
                # NB: only guarantee cuda tensor was used for tensor producer/consumer nodes
                # since other nodes would not appear at the boundary of a graph.
                # Because only tensor usages of boundary nodes would lead to
                # runtime interpreter input device mismatch exception.
                if isinstance(val.type(), torch._C.TensorType):
                    if val.uses() and not tools.is_gpu_tensor_type(val):
                        self._add_unsupported(node)
                        return False

        if node.kind().startswith("prim"):
            # prim Ops will be lazy added, when used by a trival aten Op
            return True

        self._current_segment_size += 1
        try:
            self._appendNode(node)
            subgraph = self._segment_to_subgraph()
            checker = OnnxBackendChecker(
                subgraph, self._onnx_backend_test_func, self._backend_name
            )
            supported = checker()
            if supported:
                logger.debug("Found supported: %s" % (node.kind()))
                return True
        except Exception as error:
            logger.debug(error)

        self._add_unsupported(node)
        return False

    def _appendNode(self, old_node):
        self._add_graph_input_if_need(old_node)
        # old_node maybe from another graph
        node = self._current_segment.createClone(
            old_node, lambda x: self._orig2segment_value_map[x]
        )
        self._current_segment.appendNode(node)
        for src_out, dst_out in zip(old_node.outputs(), node.outputs()):
            self._orig2segment_value_map[src_out] = dst_out
            self._segment2orig_value_map[dst_out] = src_out

    def _segment_to_subgraph(self):
        # graph.copy() would change the value debugName
        subgraph = self._current_segment.copy()
        # swap, need original values to look up self._segment2orig_value_map
        subgraph, self._current_segment = self._current_segment, subgraph

        all_used_values = {
            val for node in subgraph.nodes() for val in node.inputs()
        }
        all_inter_outputs = {
            val for node in subgraph.nodes() for val in node.outputs()
        }

        def _orig_node_used(n):
            return (
                n in self._segment2orig_value_map
                and self._segment2orig_value_map[n].uses()
            )

        for out in all_inter_outputs:
            if out not in all_used_values and _orig_node_used(out):
                subgraph.registerOutput(out)

        return subgraph


def get_unsupported_nodes(
    graph, onnx_backend_test_func, backend_name, ignore_device=False
):
    builder = OnnxBackendTestBed(
        graph, ignore_device, onnx_backend_test_func, backend_name
    )
    for node in graph.node_list():
        builder.appendNode(node)
    unsupported_set = builder.get_unsupported()
    return unsupported_set
