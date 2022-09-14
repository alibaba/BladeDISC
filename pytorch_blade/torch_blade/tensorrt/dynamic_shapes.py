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

import contextlib

import torch
import torch_blade._torch_blade._backends as _backends

from torch_blade import version
from torch_blade.clustering.support_fusion_group import supported_node_fusion
from torch_blade.clustering.support_group_conversion import group_nodes
from torch_blade.config import Config
from torch_blade.logging import logger
from torch_blade.tools.shape_inference import record_shape_by_tracing

_type_map = {
    "Bool": torch.bool,
    "Byte": torch.uint8,
    "Char": torch.int8,
    "Short": torch.int16,
    "Int": torch.int,
    "Long": torch.int64,
    "Half": torch.half,
    "Float": torch.float,
    "Double": torch.double,
}


def get_shape_for_one_input(c_module, input_setting, trt_unsupported):
    # This should not change the topology and the order of the graph.
    # Also, since clustering is unstable, we do not execute clustering here.
    # Instead we use topology order to determing the unsupported nodes.

    # graph.copy() create a new graph by clone original graph.nodes() in order, according to pytorch codes:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/csrc/jit/python/python_ir.cpp#L349
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/csrc/jit/ir/ir.cpp#L688
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/csrc/jit/ir/ir.cpp#L658
    def _is_list_of_int(shape):
        if not isinstance(shape, list):
            return False
        return all(isinstance(x, int) for x in shape)

    all_shapes = all(_is_list_of_int(shape) for shape in input_setting)

    if all_shapes:
        graph = c_module.forward.graph
        input_types = [_type_map[inp.type().scalarType()] for inp in graph.input_list()[1:]]
        if len(input_types) != len(input_setting):
            raise Exception(
                "Input shapes number({}) not match with model input number({}), input shapes: {}".format(
                    len(input_setting), len(input_types), input_setting
                )
            )

        inputs = [
            torch.ones(inp, device="cuda" if version.cuda_available else "cpu").to(typ)
            for inp, typ in zip(input_setting, input_types)
        ]
    else:
        inputs = input_setting
    return _get_shape_for_one_input(c_module, inputs, trt_unsupported)

def _get_shape_for_one_input(c_module, inputs, trt_unsupported):
    graph = c_module.forward.graph.copy()
    record_shape_by_tracing(c_module, inputs, graph)
    unsupported_indices = [
        idx
        for idx, n in enumerate(c_module.forward.graph.nodes())
        if n in trt_unsupported
    ]
    new_trt_unsupported = set(
        n for i, n in enumerate(graph.nodes()) if i in unsupported_indices
    )

    # TODO: Since clustering is unstable and time-consuming, it is better
    # to execute clustering only once. So we do not execute new clustering
    # process during getting dynamic settings.
    supported_node_fusion(graph, graph, new_trt_unsupported)
    all_subgraph_inputs = []
    for node in group_nodes(graph):
        subgraph = node.g("Subgraph")
        single_subgraph_inputs = [inp.type().sizes() for inp in subgraph.inputs()]
        all_subgraph_inputs.append(single_subgraph_inputs)

    return all_subgraph_inputs


def check_two_shape(shape1, shape2):
    """
    shape1: [shapes_for_subgraph1, shapes_for_subgraph2, ...]
    shape2: [shapes_for_subgraph1, shapes_for_subgraph2, ...]
    three conditions must be satisfied, described below
    """
    # 1. for the same module, the results of clustering must consts of the
    # same number of subgraphs
    assert len(shape1) == len(shape2)

    for s1, s2 in zip(shape1, shape2):
        # 2. for each subgraph, the numbers of input will not change
        assert len(s1) == len(s2)

        # 3. for each input, they should have the same dimension
        for inp1, inp2 in zip(s1, s2):
            assert len(inp1) == len(inp2)


def get_dynamic_shape(min_shape, max_shape):
    """
    Determine which dimension should be set to -1 according to min/max shape
    """
    assert len(min_shape) == len(max_shape)
    dynamic_shape = []
    for min_inp, max_inp in zip(min_shape, max_shape):
        # If one dimension is same between min_shape and max_shape
        # we consider it as a static dimension. Or it is treated as
        # dynamic dimension which will be set to -1
        dyn = [i if i == j else -1 for i, j in zip(min_inp, max_inp)]
        dynamic_shape.append(dyn)
    return dynamic_shape


@contextlib.contextmanager
def set_extra_inputs(c_module, extra_inputs, extra_dynamic_shapes):
    def setattr_leaf(obj, fields, val):
        if len(fields) == 1:
            # return old value and set new
            old_val = obj.getattr(fields[0])
            obj.setattr(fields[0], val)
            return old_val
        return setattr_leaf(obj.getattr(fields[0]), fields[1:], val)

    extra_inputs = extra_inputs or []
    extra_dynamic_shapes = extra_dynamic_shapes or []
    assert len(extra_inputs) == len(
        extra_dynamic_shapes
    ), "the number of extra_inputs and shape are not equal"

    try:
        old_values = []
        for inp, shape in zip(extra_inputs, extra_dynamic_shapes):
            fields = inp.split(".")
            old_val = setattr_leaf(c_module, fields, torch.randn(shape, device="cuda"))
            old_values.append(old_val)
        yield
    finally:
        for inp, old_val in zip(extra_inputs, old_values):
            fields = inp.split(".")
            setattr_leaf(c_module, fields, old_val)


def _get_dynamic_settings(c_module, trt_unsupported):
    trt_dynamic_shapes = Config.get_current_context_or_new().dynamic_tuning_shapes
    trt_dynamic_inputs = Config.get_current_context_or_new().dynamic_tuning_inputs
    all_shapes = []
    if (not trt_dynamic_shapes) and (not trt_dynamic_inputs):
        return all_shapes

    if trt_dynamic_shapes and trt_dynamic_inputs:
        logger.warn("dynamic_tuning_shapes and dynamic_tuning_inputs are set at the same time, "
                    "torch-blade will only use dynamic_tuning_shapes")

    trt_dynamic_settings = trt_dynamic_shapes if trt_dynamic_shapes else trt_dynamic_inputs

    trt_extra_dynamic_shapes = (
        Config.get_current_context_or_new().extra_dynamic_tuning_shapes
    )

    if trt_extra_dynamic_shapes and len(trt_dynamic_settings) != len(
        trt_extra_dynamic_shapes
    ):
        raise Exception(
            "If trt_extra_dynamic_shapes is used, it should have the same length as trt_dynamic_settings"
        )

    for idx, each_shape_setting in enumerate(trt_dynamic_settings):
        each_shape = []
        each_trt_extra_dynamic_shapes = (
            trt_extra_dynamic_shapes[idx] if trt_extra_dynamic_shapes else {}
        )
        extra_inputs = each_trt_extra_dynamic_shapes.get("extra_inputs", None)

        extra_min_shape = each_trt_extra_dynamic_shapes.get("min", None)
        if "min" not in each_shape_setting:
            raise Exception("The ranges of min/max/opts needs to be all set")
        min_setting = each_shape_setting["min"]
        with set_extra_inputs(c_module, extra_inputs, extra_min_shape):
            min_shape_for_subgraphs = get_shape_for_one_input(
                c_module, min_setting, trt_unsupported
            )

        extra_max_shape = each_trt_extra_dynamic_shapes.get("max", None)
        if "max" not in each_shape_setting:
            raise Exception("The ranges of min/max/opts needs to be all set")
        max_shape = each_shape_setting["max"]
        with set_extra_inputs(c_module, extra_inputs, extra_max_shape):
            max_shape_for_subgraphs = get_shape_for_one_input(
                c_module, max_shape, trt_unsupported
            )

        check_two_shape(min_shape_for_subgraphs, max_shape_for_subgraphs)

        if "opts" not in each_shape_setting:
            raise Exception("The ranges of min/max/opts needs to be all set")
        opt_shapes = each_shape_setting["opts"]
        if extra_inputs is None:
            opt_shapes_for_subgraphs = [
                get_shape_for_one_input(c_module, ops_shape, trt_unsupported)
                for ops_shape in opt_shapes
            ]
        else:
            extra_opt_shapes = each_trt_extra_dynamic_shapes.get("opts", None)
            assert len(opt_shapes) == len(extra_opt_shapes)
            opt_shapes_for_subgraphs = []
            for opt_shape, extra_opt_shape in zip(opt_shapes, extra_opt_shapes):
                with set_extra_inputs(c_module, extra_inputs, extra_opt_shape):
                    opt_shapes_for_subgraphs.append(
                        get_shape_for_one_input(c_module, opt_shape, trt_unsupported)
                    )

        for opt in opt_shapes_for_subgraphs:
            check_two_shape(opt, min_shape_for_subgraphs)
        opt_shapes_for_subgraphs = list(map(list, list(zip(*opt_shapes_for_subgraphs))))

        for subgraph_min, subgraph_max, subgraph_opts in zip(
            min_shape_for_subgraphs, max_shape_for_subgraphs, opt_shapes_for_subgraphs
        ):
            dynamic_ranges = _backends.DynamicRanges()
            dynamic_ranges.min_shape = subgraph_min
            dynamic_ranges.max_shape = subgraph_max
            dynamic_ranges.dynamic_setting = get_dynamic_shape(
                subgraph_min, subgraph_max
            )
            dynamic_ranges.opt_shapes = subgraph_opts
            each_shape.append(dynamic_ranges)

        all_shapes.append(each_shape)
    return all_shapes


def get_dynamic_settings(c_module, trt_unsupported):
    try:
        return _get_dynamic_settings(c_module, trt_unsupported)
    except Exception as error:
        logger.warning(error)
        return None
