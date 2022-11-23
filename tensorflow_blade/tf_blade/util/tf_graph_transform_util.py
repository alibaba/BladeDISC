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

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tensorflow.core.framework import attr_value_pb2, tensor_shape_pb2
from tensorflow.python.framework import constant_op, tensor_util

from tf_blade.util.graph_transform import TransformGraph
from tf_blade.util.simple_graph import SimpleGraph, SimpleNode
from tf_blade.util.tf_import_helper import tf
from tf_blade.util.tf_util import tensor_name_to_node_name


class OpType(Enum):
    ADD = "Add"
    ADD_N = "AddN"
    ADD_V2 = "AddV2"
    AVG_POOL = "AvgPool"
    BATCH_MAT_MUL = "BatchMatMul"
    BATCH_MAT_MUL_V2 = "BatchMatMulV2"
    BATCH_NORM_WITH_GOLBAL_NORMALIZATION = "BatchNormWithGlobalNormalization"
    BIAS_ADD = "BiasAdd"
    CAST = "Cast"
    CONCAT_V2 = "ConcatV2"
    CONST = "Const"
    CONV2D = "Conv2D"
    CUDNN_RNN = "CudnnRNN"
    ENTER = "Enter"
    EXIT = "Exit"
    EXPAND_DIMS = "ExpandDims"
    FUSED_BATCH_NORM = "FusedBatchNorm"
    FUSED_BATCH_NORM_V2 = "FusedBatchNormV2"
    FUSED_BATCH_NORM_V3 = "FusedBatchNormV3"
    GATHERV2 = "GatherV2"
    GREATER = "Greater"
    IDENTITY = "Identity"
    IDENTITY_N = "IdentityN"
    LESS = "Less"
    LOOP_COND = "LoopCond"
    MAT_MUL = "MatMul"
    MAX = "Max"
    MAX_POOL = "MaxPool"
    MAXIMUM = "Maximum"
    MEAN = "Mean"
    MERGE = "Merge"
    MINIMUM = "Minimum"
    MUL = "Mul"
    NEXT_ITERATION = "NextIteration"
    PACK = "Pack"
    PAD = "Pad"
    PLACEHOLDER = "Placeholder"
    RELU = "Relu"
    RESHAPE = "Reshape"
    RSQRT = "Rsqrt"
    SHAPE = "Shape"
    SLICE = "Slice"
    SQUARE = "Square"
    STRIDED_SLICE = "StridedSlice"
    SUB = "Sub"
    SUM = "Sum"
    SWITCH = "Switch"
    TRANSPOSE = "Transpose"
    UNPACK = "Unpack"
    SPARSE_SEGMENT_MEAN = "SparseSegmentMean"
    SPARSE_SEGMENT_SUM = "SparseSegmentSum"
    # custom ops
    BLADE_BATCH_MAT_MUL = "BladeOptBatchMatMul"
    BLADE_BILSTM = "BladeBilstm"
    BLADE_FUSED_LSTM_ELEMENT_WISE = "BladeFusedLSTMElementWise"
    FUSED_NMT_LAYER_NORM = "BladeFusedTransformerLayerNorm"
    BLADE_FUSED_EMBEDDING_SPARSE = "BladeFusedEmbeddingSparse"


# [Set node attribute]
def copy_node_attr(
    src_node: tf.NodeDef, src_key: str, dst_key: str, dst_node: tf.NodeDef
) -> None:
    try:
        dst_node.attr[dst_key].CopyFrom(src_node.attr[src_key])
    except Exception as err:
        logging.warning("Failed to copy attribute: {}".format(err))


def set_attr_bool(node: tf.NodeDef, key: str, value: bool) -> None:
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_dtype(node: tf.NodeDef, key: str, value: tf.DType) -> None:
    try:
        enum = value.as_datatype_enum
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(type=enum))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_dtype_list(node: tf.NodeDef, key: str, value: List[tf.DType]) -> None:
    type_value = [v.as_datatype_enum for v in value]
    list_value = attr_value_pb2.AttrValue.ListValue(type=type_value)
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_float(node: tf.NodeDef, key: str, value: float) -> None:
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_int(node: tf.NodeDef, key: str, value: int) -> None:
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_int_list(node: tf.NodeDef, key: str, value: List[int]) -> None:
    list_value = attr_value_pb2.AttrValue.ListValue(i=value)
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_shape(
    node: tf.NodeDef, key: str, value: Union[List[int], Tuple[int]]
) -> None:
    try:
        dim = [tensor_shape_pb2.TensorShapeProto.Dim(size=d) for d in value]
        shape_prt = tensor_shape_pb2.TensorShapeProto(dim=dim)
        shape_attr = attr_value_pb2.AttrValue(shape=shape_prt)
        node.attr[key].CopyFrom(shape_attr)
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_string(node: tf.NodeDef, key: str, value: str) -> None:
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value.encode("utf-8")))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


def set_attr_byte(node: tf.NodeDef, key: str, value: bytes) -> None:
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))
    except Exception as err:
        logging.warning("Failed to set attribute: {}".format(err))


# [Add node]
def _add_op_common(
    graph_def: tf.GraphDef, op_type: OpType, input_name_list: List[str], name: str
) -> tf.NodeDef:
    node = graph_def.node.add()
    node.name = name
    node.op = op_type.value
    node.input.extend(input_name_list)
    return node


def add_binary_op(
    graph_def: tf.GraphDef,
    op_type: OpType,
    input_name_list: List[str],
    name: str,
    data_type: tf.DType,
) -> None:
    node = _add_op_common(graph_def, op_type, input_name_list, name)
    set_attr_dtype(node, "T", data_type)


def add_bias_add(
    graph_def: tf.GraphDef,
    input_name: str,
    bias_name: str,
    name: str,
    data_type: tf.DType,
    data_format: str = "NHWC",
) -> None:
    input_name_list = [input_name, bias_name]
    node = _add_op_common(graph_def, OpType.BIAS_ADD, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_string(node, "data_format", data_format)


def add_cast(
    graph_def: tf.GraphDef,
    input_name: str,
    name: str,
    src_type: tf.DType,
    dst_type: tf.DType,
) -> None:
    node = _add_op_common(graph_def, OpType.CAST, [input_name], name)
    set_attr_dtype(node, "SrcT", src_type)
    set_attr_dtype(node, "DstT", dst_type)


def add_concat_v2(
    graph_def: tf.GraphDef,
    input_name_list: List[str],
    name: str,
    data_type: tf.DType,
    idx_type: tf.DType = tf.int32,
    input_axis_name: Optional[str] = None,
    axis: int = 0,
) -> None:
    if input_axis_name is None:
        input_axis_name = "{}/axis".format(name)
        add_const(graph_def, np.array(axis), input_axis_name, idx_type)
    input_name_list.append(input_axis_name)
    node = _add_op_common(graph_def, OpType.CONCAT_V2, input_name_list, name)
    set_attr_int(node, "N", len(input_name_list) - 1)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tidx", idx_type)


def add_const(
    graph_def: tf.GraphDef, array: np.ndarray, name: str, data_type: tf.DType
) -> None:
    node = graph_def.node.add()
    constant = constant_op.constant(array, dtype=data_type, name=name)
    node.MergeFrom(constant.op.node_def)


def add_cudnn_rnn_lstm(
    graph_def: tf.GraphDef,
    input_name: str,
    input_h_name: str,
    input_c_name: str,
    param_name: str,
    name: str,
    data_type: tf.DType,
) -> None:
    input_name_list = [input_name, input_h_name, input_c_name, param_name]
    node = _add_op_common(graph_def, OpType.CUDNN_RNN, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_string(node, "direction", "unidirectional")
    set_attr_float(node, "dropout", 0.0)
    set_attr_string(node, "input_mode", "linear_input")
    set_attr_bool(node, "is_training", False)
    set_attr_string(node, "rnn_mode", "lstm")
    set_attr_int(node, "seed", 0)
    set_attr_int(node, "seed2", 0)


def add_expand_dims(
    graph_def: tf.GraphDef,
    input_name: str,
    name: str,
    data_type: tf.DType,
    dim_type: tf.DType = tf.int32,
    input_dim_name: Optional[str] = None,
    dim: int = 0,
) -> None:
    if input_dim_name is None:
        input_dim_name = "{}/dim".format(name)
        add_const(graph_def, np.array(dim), input_dim_name, dim_type)
    input_name_list = [input_name, input_dim_name]
    node = _add_op_common(graph_def, OpType.EXPAND_DIMS, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tdim", dim_type)


def add_fused_batch_norm(
    graph_def: tf.GraphDef,
    input_name_list: List[str],
    name: str,
    data_type: tf.DType,
    epsilon: float,
    is_training: bool = False,
) -> None:
    op_type = OpType.FUSED_BATCH_NORM
    node = _add_op_common(graph_def, op_type, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_float(node, "epsilon", epsilon)
    set_attr_bool(node, "is_training", is_training)


def add_identity(
    graph_def: tf.GraphDef, input_name: str, name: str, data_type: tf.DType
) -> tf.NodeDef:
    node = _add_op_common(graph_def, OpType.IDENTITY, [input_name], name)
    set_attr_dtype(node, "T", data_type)
    return node


def add_identity_n(
    graph_def: tf.GraphDef,
    input_name_list: List[str],
    name: str,
    data_type_list: List[tf.DType],
) -> None:
    node = _add_op_common(graph_def, OpType.IDENTITY_N, input_name_list, name)
    set_attr_dtype_list(node, "T", data_type_list)


def add_max(
    graph_def: tf.GraphDef,
    input_name: str,
    axis_input_name: str,
    name: str,
    data_type: tf.DType,
    idx_type: tf.DType = tf.int32,
    keep_dims: bool = False,
) -> None:
    input_name_list = [input_name, axis_input_name]
    node = _add_op_common(graph_def, OpType.MAX, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tidx", idx_type)
    set_attr_bool(node, "keep_dims", keep_dims)


def add_maximum(
    graph_def: tf.GraphDef, x_name: str, y_name: str, name: str, data_type: tf.DType,
) -> None:
    input_name_list = [x_name, y_name]
    add_binary_op(graph_def, OpType.MAXIMUM, input_name_list, name, data_type)


def add_merge(
    graph_def: tf.GraphDef, input_name_list: List[str], name: str, data_type: tf.DType
) -> None:
    node = _add_op_common(graph_def, OpType.MERGE, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_int(node, "N", len(input_name_list))


def add_minimum(
    graph_def: tf.GraphDef, x_name: str, y_name: str, name: str, data_type: tf.DType,
) -> None:
    input_name_list = [x_name, y_name]
    add_binary_op(graph_def, OpType.MINIMUM, input_name_list, name, data_type)


def add_gather(
    graph_def: tf.GraphDef,
    weight_name: str,
    query_name: str,
    axis_name: str,
    name: str,
    params_type: tf.DType,
    axis_type: tf.DType = tf.int32,
    id_type: tf.DType = tf.int32,
    batch_dims: int = 0,
) -> None:
    input_name_list = [weight_name, query_name, axis_name]
    node = _add_op_common(graph_def, OpType.GATHERV2, input_name_list, name)
    set_attr_dtype(node, "Taxis", axis_type)
    set_attr_dtype(node, "Tindices", id_type)
    set_attr_dtype(node, "Tparams", params_type)
    set_attr_int(node, "batch_dims", batch_dims)


def add_next_iteration(
    graph_def: tf.GraphDef, input_name_list: List[str], name: str, data_type: tf.DType
) -> None:
    node = _add_op_common(graph_def, OpType.NEXT_ITERATION, input_name_list, name)
    set_attr_dtype(node, "T", data_type)


def add_exit(
    graph_def: tf.GraphDef, input_name_list: List[str], name: str, data_type: tf.DType
) -> None:
    node = _add_op_common(graph_def, OpType.EXIT, input_name_list, name)
    set_attr_dtype(node, "T", data_type)


def add_pack(
    graph_def: tf.GraphDef,
    input_name_list: List[str],
    name: str,
    data_type: tf.DType,
    axis: int = 0,
) -> None:
    node = _add_op_common(graph_def, OpType.PACK, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_int(node, "N", len(input_name_list))
    set_attr_int(node, "axis", axis)


def add_pad(
    graph_def: tf.GraphDef,
    input_name: str,
    paddings_name: str,
    name: str,
    data_type: tf.DType,
    paddings_type: tf.DType = tf.int32,
) -> None:
    input_name_list = [input_name, paddings_name]
    node = _add_op_common(graph_def, OpType.PAD, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tpaddings", paddings_type)


def add_placeholder(
    graph_def: tf.GraphDef, data_type: tf.DType, shape: List[Any], name: str
) -> tf.NodeDef:
    shape = [s if s is not None else -1 for s in shape]
    node = _add_op_common(graph_def, OpType.PLACEHOLDER, [], name)
    set_attr_dtype(node, "dtype", data_type)
    set_attr_shape(node, "shape", shape)
    set_attr_shape(node, "_output_shapes", shape)
    return node


def add_shape(
    graph_def: tf.GraphDef,
    input_name: str,
    name: str,
    data_type: tf.DType,
    out_type: tf.DType = tf.int32,
) -> None:
    node = _add_op_common(graph_def, OpType.SHAPE, [input_name], name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "out_type", out_type)


def add_reshape(
    graph_def: tf.GraphDef,
    input_name: str,
    shape_name: str,
    name: str,
    data_type: tf.DType,
    shape_type: tf.DType = tf.int32,
) -> None:
    node = _add_op_common(graph_def, OpType.RESHAPE, [input_name, shape_name], name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tshape", shape_type)


def add_slice(
    graph_def: tf.GraphDef,
    input_name: str,
    begin_name: str,
    size_name: str,
    name: str,
    data_type: tf.DType,
    index_type: tf.DType = tf.int32,
) -> None:
    input_name_list = [input_name, begin_name, size_name]
    node = _add_op_common(graph_def, OpType.SLICE, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Index", index_type)


def add_strided_slice(
    graph_def: tf.GraphDef,
    input_name_list: List[str],
    name: str,
    data_type: tf.DType,
    index_type: tf.DType = tf.int32,
    begin_mask: int = 0,
    ellipsis_mask: int = 0,
    end_mask: int = 0,
    new_axis_mask: int = 0,
    shrink_axis_mask: int = 0,
) -> None:
    op_type = OpType.STRIDED_SLICE
    node = _add_op_common(graph_def, op_type, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Index", index_type)
    set_attr_int(node, "begin_mask", begin_mask)
    set_attr_int(node, "ellipsis_mask", ellipsis_mask)
    set_attr_int(node, "end_mask", end_mask)
    set_attr_int(node, "new_axis_mask", new_axis_mask)
    set_attr_int(node, "shrink_axis_mask", shrink_axis_mask)


def add_subgraph(
    dst: tf.GraphDef,
    subgraph: tf.GraphDef,
    input_dict: Dict[str, str],
    scope: str,
    output_node_names: List[str] = [],
    exclude_node_names: List[str] = [],
) -> Dict[str, tf.NodeDef]:
    output_nodes = {name: None for name in output_node_names}
    for node in subgraph.node:
        if node.name in exclude_node_names:
            continue
        new_node = dst.node.add()
        new_node.MergeFrom(node)
        new_node.name = f"{scope}/{node.name}"
        new_inputs = [f"{scope}/{name}" for name in node.input]
        new_node.ClearField("input")
        new_node.input.extend(new_inputs)
        if node.name in output_nodes:
            output_nodes[node.name] = new_node
    rename_node_inputs(
        dst,
        {f"{scope}/{src_name}": dst_name for src_name, dst_name in input_dict.items()},
    )
    return output_nodes


def add_sum(
    graph_def: tf.GraphDef,
    input_name: str,
    axis_input_name: str,
    name: str,
    data_type: tf.DType,
    idx_type: tf.DType = tf.int32,
    keep_dims: bool = False,
) -> None:
    input_name_list = [input_name, axis_input_name]
    node = _add_op_common(graph_def, OpType.SUM, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tidx", idx_type)
    set_attr_bool(node, "keep_dims", keep_dims)


def add_switch(
    graph_def: tf.GraphDef, input_name_list: List[str], name: str, data_type: tf.DType
) -> None:
    add_binary_op(graph_def, OpType.SWITCH, input_name_list, name, data_type)


def add_transpose(
    graph_def: tf.GraphDef,
    input_name: str,
    perm_name: str,
    name: str,
    data_type: tf.DType,
    perm_type: tf.DType,
) -> None:
    input_name_list = [input_name, perm_name]
    node = _add_op_common(graph_def, OpType.TRANSPOSE, input_name_list, name)
    set_attr_dtype(node, "T", data_type)
    set_attr_dtype(node, "Tperm", perm_type)


def add_unpack(
    graph_def: tf.GraphDef,
    input_name: str,
    name: str,
    data_type: tf.DType,
    num: int,
    axis: int = 0,
) -> None:
    node = _add_op_common(graph_def, OpType.UNPACK, [input_name], name)
    set_attr_dtype(node, "T", data_type)
    set_attr_int(node, "num", num)
    set_attr_int(node, "axis", axis)


def make_const_node(
    graph_def: tf.GraphDef, name: str, data_type: int, data: np.ndarray,
) -> None:
    dtype = tf.DType(data_type)
    data = np.asarray(data, dtype=dtype.as_numpy_dtype)
    add_const(graph_def, data, name, dtype)


# [Basic graph utils]
def get_node_name_parts_from_input(input_name: str) -> Tuple[str, str, str]:
    input_parts = input_name.split(":")
    if len(input_parts) < 2:
        suffix = ""
    else:
        suffix = ":" + input_parts[1]
    node_name = input_parts[0]
    if node_name[0] == "^":
        prefix = "^"
        node_name = node_name[1:]
    else:
        prefix = ""
    return prefix, node_name, suffix


def get_node_name_from_input(input_name: Optional[str]) -> str:
    if input_name is None:
        return ""
    _, node_name, _ = get_node_name_parts_from_input(input_name)
    return node_name


def get_canonical_input_name(input_name: str) -> str:
    prefix, node_name, suffix = get_node_name_parts_from_input(input_name)
    suffix = ":0" if suffix == "" else suffix
    return prefix + node_name + suffix


def get_unique_name(unique_name: str, graph_def: tf.GraphDef) -> str:
    node_name_list = [node.name for node in graph_def.node]
    if unique_name in node_name_list:
        for i in range(len(node_name_list)):
            unique_name = "{}/{}".format(unique_name, i + 1)
            if unique_name not in node_name_list:
                break
    return unique_name


def get_const_value(node: tf.NodeDef) -> np.ndarray:
    # Alternatively
    # tf.contrib.util.constant_value(tensor) will get a tensor"s constant value
    return np.array(tensor_util.MakeNdarray(node.attr["value"].tensor))


def get_const_value_by_name(
    graph_def: tf.GraphDef, name: str, simple_graph: Optional[SimpleGraph] = None
) -> np.ndarray:
    if simple_graph:
        node = get_node_by_name(graph_def, simple_graph, name)
        return get_const_value(node)
    else:
        node_name = get_node_name_from_input(name)
        founds = [nd for nd in graph_def.node if nd.name == node_name]
        if len(founds) == 0:
            error_msg = "Unknown node name: {}".format(node_name)
            raise Exception(error_msg)
        return get_const_value(founds[0])


# deprecated: use SimpleGraph.get_node_by_name instead
def get_node_by_name(
    graph_def: Optional[tf.GraphDef], simple_graph: SimpleGraph, name: str
) -> tf.NodeDef:
    return simple_graph.get_node_by_name(name)


# deprecated: use simple_graph.get_simple_node_by_name instead
def get_simple_node_by_name(simple_graph: SimpleGraph, name: str) -> SimpleNode:
    return simple_graph.get_simple_node_by_name(name)


def get_node_attr_value(node: tf.NodeDef, attr: str, vname: str) -> Any:
    if attr not in node.attr:
        return None
    if hasattr(node.attr[attr], vname):
        ret = getattr(node.attr[attr], vname)
        if type(ret) == bytes:
            ret = ret.decode()
        return ret
    else:
        return None


def get_node_attr_value_ex(node: tf.NodeDef, attr: str, vname: str) -> Any:
    ret = get_node_attr_value(node, attr, vname)
    if ret is None:
        error_msg = "Unknown attribute {} in {}".format(attr, node.name)
        raise Exception(error_msg)
    return ret


def get_node_type(node: tf.NodeDef, attr: str) -> tf.DType:
    return tf.DType(get_node_attr_value_ex(node, attr, "type"))


def get_node_type_by_name(
    graph_def: tf.GraphDef, simple_graph: SimpleGraph, name: str, attr: str
) -> tf.DType:
    node = get_node_by_name(graph_def, simple_graph, name)
    return get_node_type(node, attr)


def get_tensor_type_by_name(graph: tf.Graph, name: str) -> tf.DType:
    input_name = get_canonical_input_name(name)
    return graph.get_tensor_by_name(input_name).dtype


def get_shape_by_name(graph: tf.Graph, name: str) -> Optional[List[int]]:
    input_name = get_canonical_input_name(name)
    shape = graph.get_tensor_by_name(input_name).shape
    if shape == tf.TensorShape(None):
        return None
    return list(shape.as_list())


def get_tensor_format(node: tf.NodeDef, attr: str) -> str:
    return str(node.attr[attr].s.decode("utf-8")) if attr in node.attr else "NHWC"


def get_filter_format(node: tf.NodeDef, attr: str) -> str:
    return str(node.attr[attr].s.decode("utf-8")) if attr in node.attr else "OHWI"


# [Graph transform utils]
def map_node_names_to_outputs(simple_graph: SimpleGraph) -> Dict[str, List[str]]:
    outputs_map: Dict[str, List[str]] = dict()
    for node in simple_graph.nodes:
        for input_name in node.inputs:
            input_name = get_canonical_input_name(input_name)
            if input_name not in outputs_map:
                outputs_map[input_name] = list()
            outputs_map[input_name].append(node.name)
    return outputs_map


def remove_node_by_index(
    graph_def: tf.GraphDef, remove_list: Union[Set[int], List[int]]
) -> None:
    if not isinstance(remove_list, set):
        remove_list = set(remove_list)
    remove_list = list(remove_list)
    remove_list.sort(reverse=True)
    for index in remove_list:
        graph_def.node.pop(index)


def rename_node_inputs(graph_def: tf.GraphDef, rename_dict: Dict[str, str]) -> None:
    rename_dict = {get_canonical_input_name(k): v for k, v in rename_dict.items()}
    src_names = list(rename_dict.keys())
    for i, node in enumerate(graph_def.node):
        if any([get_canonical_input_name(k) in src_names for k in node.input]):
            new_inputs = list()
            for input_name in node.input:
                canonical_input_name = get_canonical_input_name(input_name)
                if canonical_input_name in src_names:
                    new_inputs.append(rename_dict[canonical_input_name])
                else:
                    new_inputs.append(input_name)
            graph_def.node[i].ClearField("input")
            graph_def.node[i].input.extend(new_inputs)


# [Pattern matching]
def check_inputs(  # noqa: C901
    simple_graph: SimpleGraph,
    current_node: SimpleNode,
    pattern_nodes: Dict[str, SimpleNode],
    first_node: SimpleNode,
    matched_name_map: Dict[str, str],
) -> bool:
    # check op type
    if first_node.op != "*":
        matched_ops = [op.strip() for op in first_node.op.split("|")]
        if current_node.op not in matched_ops:
            return False
    # check node name
    if first_node.name in matched_name_map:
        if matched_name_map[first_node.name] != current_node.name:
            return False
    # check inputs
    if (len(first_node.inputs) == 1) and (first_node.inputs[0] == "*"):
        matched_name_map[first_node.name] = current_node.name
        return True
    # if inputs contains both unknown inputs and known inputs
    if (len(first_node.inputs) > 1) and ("*" in first_node.inputs):
        known_inputs = [name for name in first_node.inputs if name != "*"]
        start_idx = 0
        for key_name in known_inputs:
            matched = False
            if key_name.isdigit():
                matched = True
                continue
            for i in range(start_idx, len(current_node.inputs)):
                input_name = current_node.inputs[i]
                cur_input_node = simple_graph.get_simple_node_by_name(input_name)
                expected_input_op_str = (pattern_nodes[key_name].op).strip()
                if "|" in expected_input_op_str:
                    expected_input_ops = expected_input_op_str.split("|")
                else:
                    expected_input_ops = list([expected_input_op_str])
                if (cur_input_node.op in expected_input_ops) and (
                    check_inputs(
                        simple_graph,
                        cur_input_node,
                        pattern_nodes,
                        pattern_nodes[key_name],
                        matched_name_map,
                    )
                ):
                    matched = True
                    start_idx = i
            if not matched:
                return False
    # if all listed inputs are known inputs
    else:
        if len(current_node.inputs) != len(first_node.inputs):
            return False
        for i, input_name in enumerate(current_node.inputs):
            cur_input_node = simple_graph.get_simple_node_by_name(input_name)
            if first_node.inputs[i].isdigit():
                continue
            tmp_input_node = pattern_nodes[first_node.inputs[i]]
            if not check_inputs(
                simple_graph,
                cur_input_node,
                pattern_nodes,
                tmp_input_node,
                matched_name_map,
            ):
                return False
    matched_name_map[first_node.name] = current_node.name
    return True


def get_matched_pattern(
    simple_graph: SimpleGraph,
    pattern_nodes: Dict[str, SimpleNode],
    first_node_key: str,
    init_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    matched_name_maps = list()
    for i, node in enumerate(simple_graph.nodes):
        simple_node = get_simple_node_by_name(simple_graph, node.name)
        tmp_name_map = init_name_map.copy() if init_name_map else dict()
        if check_inputs(
            simple_graph,
            simple_node,
            pattern_nodes,
            pattern_nodes[first_node_key],
            tmp_name_map,
        ):
            matched_name_maps.append(tmp_name_map)
    return matched_name_maps


# generate graphviz codes for given pattern
# also check if input & output consistent if validate is True
def gen_pattern_graphviz(
    pl: Union[List[SimpleNode], Dict[str, SimpleNode]], validate: bool = False
) -> str:
    if isinstance(pl, list):
        pattern_nodes = {n.name: n for n in pl}
    elif isinstance(pl, dict):
        pattern_nodes = pl
    else:
        raise Exception(f"unknown input pl type: {type(pl)}")

    vizcode = "digraph {\n"
    for name, node in pattern_nodes.items():
        for input in node.inputs:
            if input in pattern_nodes:
                inop = pattern_nodes[input].op
                if validate:
                    inode = pattern_nodes[input]
                    if name not in inode.output_nodes:
                        raise Exception(
                            f"node {input} in node {name} input list, but node {name} not in node {input} output list"
                        )
            else:
                inop = "-"
            vizcode += f"    {input}@{inop} -> {node.name}@{node.op};\n"
    vizcode += "}\n"

    return vizcode


def get_input_target_op_name(
    simple_graph: SimpleGraph,
    node_name: str,
    input_index: int,
    target_op: str,
    op_map: Dict[str, List[int]],
) -> Optional[str]:
    node = get_simple_node_by_name(simple_graph, node_name)
    input_node_name = get_node_name_from_input(node.inputs[input_index])
    input_node = get_simple_node_by_name(simple_graph, input_node_name)
    if input_node.op == target_op:
        return input_node_name
    if input_node.op in op_map:
        for tmp_index in op_map[input_node.op]:
            target_name = get_input_target_op_name(
                simple_graph, input_node_name, tmp_index, target_op, op_map
            )
            if target_name is not None:
                break
        return target_name
    else:
        return None


# [Pattern utils]
# Check if the node results are used by other sub-graphs or graph outputs
def need_preserve(
    simple_graph: SimpleGraph,
    node_names: Union[str, List[str]],
    pattern_map: Optional[Dict[str, str]],
    graph_outputs: Optional[List[str]],
) -> bool:
    if not isinstance(node_names, list):
        node_names = [node_names]
    pattern_map = pattern_map or dict()
    graph_outputs = graph_outputs or list()
    for node_name in node_names:
        node = get_simple_node_by_name(simple_graph, node_name)
        is_inner = all([k in pattern_map.values() for k in node.output_nodes])
        if (not is_inner) or (node_name in graph_outputs):
            return True
    return False


def add_remove_list(
    remove_node_list: Set[int],
    simple_graph: SimpleGraph,
    node_names: Union[str, List[str]],
    check_preserve: bool = False,
    pattern_map: Optional[Dict[str, str]] = None,
    graph_outputs: Optional[List[str]] = None,
) -> bool:
    if not isinstance(node_names, list):
        node_names = [node_names]
    removed = False
    for name in node_names:
        if (not check_preserve) or (
            not need_preserve(simple_graph, name, pattern_map, graph_outputs)
        ):
            remove_node_list.add(simple_graph.name2index(name))
            removed = True
    return removed


def add_condition_pattern(
    graph_def: tf.GraphDef,
    true_node: tf.NodeDef,
    false_node: tf.NodeDef,
    condition_name: str,
    data_type: tf.DType,
    output_name: str,
    output_num: int,
) -> None:
    # Add Switch ops for inputs
    updated_inputs: Dict[str, str] = dict()
    for node, condition in [(true_node, True), (false_node, False)]:
        for i, input_name in enumerate(node.input):
            if input_name in updated_inputs:
                switch_name = updated_inputs[input_name]
            else:
                switch_name = "{}/cond/Switch/input_{}".format(node.name, i)
                updated_inputs[input_name] = switch_name
                input_name_list = [input_name, condition_name]
                add_switch(graph_def, input_name_list, switch_name, data_type)
            name = "{}:1".format(switch_name) if condition else switch_name
            node.input[i] = name
    # Add Merge ops for outputs
    output_names = list()
    for i in range(output_num):
        true_name = true_node.name
        false_name = false_node.name
        true_out = "{}:{}".format(true_name, i) if i > 0 else true_name
        false_out = "{}:{}".format(false_name, i) if i > 0 else false_name
        output_names.append("{}/cond/Merge/output_{}".format(output_name, i))
        tmp_inputs = [true_out, false_out]
        add_merge(graph_def, tmp_inputs, output_names[-1], data_type)
    data_type_list = [data_type] * output_num
    add_identity_n(graph_def, output_names, output_name, data_type_list)


# [Graph transform tools]
def transform_graph(
    graph_def: tf.GraphDef, inputs: List[str], outputs: List[str], transforms: List[str]
) -> Optional[tf.GraphDef]:
    logging.info(("Graph transformations:\n{}").format(transforms))
    try:
        opt_graph_def = TransformGraph(graph_def, inputs, outputs, transforms)
        tf.reset_default_graph()
        tf.import_graph_def(opt_graph_def, name="")
    except Exception as err:
        info = "Error encountered while transformation: {}".format(err)
        if len(info) > 1024:
            info = info[:1024] + "..."
        logging.warning(info)
        return None
    return opt_graph_def


def remove_identity_nodes(
    graph_def: tf.GraphDef, graph_inputs: List[str], graph_outputs: List[str]
) -> tf.GraphDef:
    opt_graph_def = transform_graph(
        graph_def, graph_inputs, graph_outputs, ["remove_nodes(op=Identity)"]
    )
    return opt_graph_def or graph_def


def sort_by_execution_order(
    graph_def: tf.GraphDef, graph_inputs: List[str], graph_outputs: List[str]
) -> tf.GraphDef:
    transforms = [
        "merge_duplicate_nodes",
        "strip_unused_nodes",
        "sort_by_execution_order",
    ]
    for transform in transforms:
        opt_graph_def = transform_graph(
            graph_def, graph_inputs, graph_outputs, [transform]
        )
        graph_def = opt_graph_def or graph_def
    return graph_def


def is_const_or_enter_from_const(
    graph_def: tf.GraphDef, simple_graph: SimpleGraph, name: str
) -> bool:
    name = tensor_name_to_node_name(name)
    node = get_node_by_name(graph_def, simple_graph, name)
    if str(node.op) == OpType.CONST.value:
        return True
    elif str(node.op) == OpType.ENTER.value:
        return is_const_or_enter_from_const(graph_def, simple_graph, str(node.input[0]))
    return False


def try_get_const_or_enter_node_value(
    graph_def: tf.GraphDef, simple_graph: SimpleGraph, name: str
) -> Optional[np.ndarray]:
    name = tensor_name_to_node_name(name)
    node = get_node_by_name(graph_def, simple_graph, name)
    if str(node.op) == OpType.CONST.value:
        return get_const_value(node)
    elif str(node.op) == OpType.ENTER.value:
        return try_get_const_or_enter_node_value(
            graph_def, simple_graph, str(node.input[0])
        )
    logging.warning("Failed to get Const value from Node named: {}.".format(name))
    return None


def try_get_2d_weight_tensor_shape(
    graph_def: tf.GraphDef, simple_graph: SimpleGraph, name: str
) -> Optional[List[int]]:
    tensor_weight = try_get_const_or_enter_node_value(graph_def, simple_graph, name)
    if tensor_weight is None:
        logging.error(
            "Fail to get const or enter-const node {} weight shape".format(name)
        )
        return None
    weight_shape_list = list(tensor_weight.shape)
    if len(weight_shape_list) != 2:
        logging.error("Invalid MatMul weight tensor, with shape dimensions not equal 2")
        return None
    return weight_shape_list


def try_get_1d_tensor_strided_slice_info(
    graph_def: tf.GraphDef, simple_graph: SimpleGraph, node: tf.NodeDef
) -> Optional[List[int]]:
    if node.op != OpType.STRIDED_SLICE.value:
        return None
    if not (node.attr["begin_mask"].i == int(0) or node.attr["begin_mask"].i == int(1)):
        return None
    if node.attr["end_mask"].i != int(0):
        return None
    if node.attr["ellipsis_mask"].i != int(0):
        return None
    beg = try_get_const_or_enter_node_value(graph_def, simple_graph, str(node.input[1]))
    if beg is None:
        logging.warning(
            "Fail to get beg info for strided_slice node {}".format(node.name)
        )
        return None
    end = try_get_const_or_enter_node_value(graph_def, simple_graph, str(node.input[2]))
    if end is None:
        logging.warning(
            "Fail to get end info for strided_slice node {}".format(node.name)
        )
        return None
    stride = try_get_const_or_enter_node_value(
        graph_def, simple_graph, str(node.input[3])
    )
    if stride is None:
        logging.warning(
            "Fail to get stride info for strided_slice node {}".format(node.name)
        )
        return None
    out = list()
    for x in [beg, end, stride]:
        if (x.dtype != np.int32) or (x.size != 1):
            logging.warning("Got invalid 1D Tensor StridedSlice stack info.")
            return None
        else:
            out.append(int(x))
    return out
