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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tensorflow.compat.v1.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.training import saver

from tf_blade.util.tf_import_helper import tf


class TensorInfo:
    def __init__(self, name: str, shape: List[Any], dtype: tf.DType) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype


def get_tf_version() -> str:
    return str(tf.__version__)


def get_tf_major_version() -> int:
    return int(get_tf_version().split(".")[0])


def is_tf2() -> bool:
    return get_tf_major_version() == 2


def is_pai_tf() -> bool:
    return "PAI" in tf.__version__


def get_node_name(full_name: str) -> Tuple[str, bool, int]:
    """Get node actual name from full name in graph_def connection"""
    is_ctrl = False
    port = -1
    node_name = full_name
    if full_name.startswith("^"):
        node_name = full_name[1:]
        is_ctrl = True

    node_name_split = node_name.split(":")
    node_name = node_name_split[0]
    if len(node_name_split) > 1:
        port = int(node_name_split[1])
    return node_name, is_ctrl, port


def get_canonical_tensor_name(name: str) -> str:
    """
    Legal tensor names are like: name, ^name, or name:digits. Please refert to:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/graph/tensor_id.cc#L35
    """
    parts = name.split(":")
    is_control_input = name.startswith("^")
    if len(parts) == 1:
        suffix = "" if is_control_input else ":0"
        return name + suffix
    elif len(parts) == 2 and parts[1].isdecimal() and not is_control_input:
        return name
    else:
        raise Exception(f"Invalid tensor name: {name}")


def tensor_name_to_node_name(tensor_name: str) -> str:
    return tensor_name.strip().strip("^").split(":")[0]


def get_tensor_output_idx(tensor_name: str) -> int:
    return get_node_name(get_canonical_tensor_name(tensor_name))[2]


def get_output_shape(
    node: tf.NodeDef, name_or_idx: Union[str, int]
) -> Union[np.ndarray, None]:
    output_idx: int = (
        name_or_idx
        if isinstance(name_or_idx, int)
        else get_tensor_output_idx(name_or_idx)
    )
    if ("_output_shapes" not in node.attr) or (
        len(node.attr["_output_shapes"].list.shape) <= output_idx
    ):
        return None
    output_shape_attr = node.attr["_output_shapes"].list.shape[output_idx]
    output_shape = [d.size for d in output_shape_attr.dim]
    return np.array(output_shape)


def get_value_from_const(node: tf.NodeDef) -> np.ndarray:
    return np.array(tf.make_ndarray(node.attr["value"].tensor))


def make_const_node(
    name: str, dtype: Union[str, tf.DType], data: Union[np.ndarray, list]
) -> tf.NodeDef:
    if not isinstance(dtype, tf.DType):
        dtype = tf.DType(dtype)
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=dtype.as_numpy_dtype)
    node = tf.NodeDef()
    node.op = "Const"
    node.name = name
    node.attr["dtype"].type = dtype.as_datatype_enum
    node.attr["value"].CopyFrom(
        tf.AttrValue(
            tensor=tf.make_tensor_proto(
                data, dtype=dtype.as_datatype_enum, shape=data.shape
            )
        )
    )
    return node


def add_shape(graph_def: tf.GraphDef) -> tf.GraphDef:
    """Add fixed shape to GraphDef"""
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph.as_graph_def(add_shapes=True)


def set_attr_i(node: tf.NodeDef, key: str, value: Any) -> None:
    """Set int attribute"""
    try:
        node.attr[key].CopyFrom(tf.AttrValue(i=value))
    except KeyError:
        pass


def generate_node(
    n: tf.NodeDef, new_inputs: List[str], identity_map: Dict[str, str]
) -> tf.NodeDef:
    new_n = tf.NodeDef()
    if n.name in identity_map:
        new_n.name = n.name
        new_n.op = "Identity"
        if n.attr["T"]:
            new_n.attr["T"].CopyFrom(n.attr["T"])
        elif n.attr["dtype"]:
            new_n.attr["T"].CopyFrom(n.attr["dtype"])
    else:
        new_n.CopyFrom(n)
        del new_n.input[:]

    for ni in new_inputs:
        new_n.input.append(ni)
    return new_n


def modify_graph(
    graph_def: tf.GraphDef, dead_nodes: Set[tf.NodeDef], identity_map: Dict[str, str]
) -> tf.GraphDef:

    new_nodes = []
    new_gd = tf.GraphDef()
    for n in graph_def.node:
        if n.name in dead_nodes:
            continue

        if n.name in identity_map:
            valid_iname = get_node_name(identity_map[n.name])[0]

        new_inputs = []
        df_input = 0
        for iedge in n.input:
            iname, is_ctrl, _ = get_node_name(iedge)
            if iname in identity_map:
                iedge = iedge.split(":")[0]
            if n.name in identity_map and not is_ctrl and iname != valid_iname:
                continue
            if iname in dead_nodes:
                continue
            if n.name in identity_map and df_input > 0:
                continue
            if not is_ctrl:
                df_input += 1
            new_inputs.append(iedge)

        new_n = generate_node(n, new_inputs, identity_map)
        new_nodes.append(new_n)
    new_gd.node.extend(new_nodes)
    return new_gd


def get_tensor_info_mapping(graph_def: tf.GraphDef) -> Dict[str, TensorInfo]:
    tensor_info_map = {}
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
        for node in graph_def.node:
            operation = graph.get_operation_by_name(node.name)
            output_tensor = operation.outputs
            for tensor in output_tensor:
                try:
                    if tensor.shape == tf.TensorShape(None):
                        tensor_shape = []
                    else:
                        tensor_shape = tensor.shape.as_list()
                    info = TensorInfo(tensor.name, tensor_shape, tensor.dtype)
                except Exception as err:
                    raise Exception(f"Unknown error when getting tensor info: {err}")
                tensor_info_map[tensor.name] = info
    return tensor_info_map


def add_shapes(graph_def: tf.GraphDef) -> tf.GraphDef:
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None,
        )
        return graph.as_graph_def(add_shapes=True)


def check_node_rank(node: tf.NodeDef, rank: int) -> bool:
    # get shapes
    if "_output_shapes" not in node.attr:
        return False
    if not node.attr["_output_shapes"].list.shape:
        return False
    if len(node.attr["_output_shapes"].list.shape[0].dim) != rank:
        return False
    return True


"""
go through all nodes in graph_def and replace node ops defined in op_map
skip one node if filter callback is given and return True
return the number of nodes replaced
"""


def replace_node_ops(
    graph_def: tf.GraphDef,
    op_map: Dict[str, str],
    filter: Optional[Callable[[tf.NodeDef], bool]] = None,
) -> int:
    count = 0
    for node in graph_def.node:
        mapped_op = op_map.get(node.op, None)
        if mapped_op:
            if filter and filter(node):
                logging.debug(f"skip node replace by filter callback:\n{node}")
                continue
            node.op = mapped_op
            count += 1
    return count


def replace_node_ops_filter_dtype(
    graph_def: tf.GraphDef, op_map: Dict[str, str], dtype: tf.DType
) -> int:
    return replace_node_ops(
        graph_def,
        op_map,
        lambda n: "T" in n.attr
        and int(n.attr["T"].type) != int(dtype.as_datatype_enum),
    )


def graph_def_to_meta_graph(
    graph_def: tf.GraphDef,
    input_nodes: List[str],
    output_nodes: List[str],
    extra_reserved_nodes: List[str] = [],
    method_name: str = "blade_infer",
) -> tf.MetaGraphDef:
    """Convert GraphDef to MetaGraphDef.
    This function will try it"s best to build a sinature for the result MetaGraphDef. It
    will be built when all input nodes has attribute dtype and shape. Otherwise it falls
    back to add all input/out/extra_reserved nodes to train_op set.
    The difference between the two ways is that in Grappler"s constant fold pass, node
    identified as feed neither won"t be involved in shape tracking nor get it"s shape
    freezed as constant, which may lead to problematic constant folding of subsequent
    Shape node.

    Parameters
    ----------
    graph_def : tf.GraphDef
        Input GraphDef object.
    input_nodes : List[str]
        Input nodes of graph.
    output_nodes : List[str]
        Output nodes of graph.
    extra_reserved_nodes : List[str], optional
        Other nodes need to be reserved, by default []
    method_name : str, optional
        Method name of signature, by default "blade_infer"

    Returns
    -------
    tf.MetaGraphDef
        [description]
    """
    metagraph = saver.export_meta_graph(graph_def=graph_def)

    # If all input nodes has info about dtype and shape, we attach signature to metagraph.
    # Otherwise, fall back to legacy method: add all nodes to train_op set.
    name_to_node = {n.name: n for n in graph_def.node}
    if all([{"dtype", "shape"} <= name_to_node[n].attr.keys() for n in input_nodes]):

        def _as_tensor(node_or_tensor: str) -> str:
            return node_or_tensor if ":" in node_or_tensor else node_or_tensor + ":0"

        sig_inputs = {
            i: meta_graph_pb2.TensorInfo(
                name=_as_tensor(i),
                dtype=name_to_node[i].attr["dtype"].type,
                tensor_shape=name_to_node[i].attr["shape"].shape,
            )
            for i in input_nodes
        }
        # Grappler only need output names.
        sig_outputs = {
            o: meta_graph_pb2.TensorInfo(name=_as_tensor(o)) for o in output_nodes
        }
        signature_def = signature_def_utils.build_signature_def(
            inputs=sig_inputs, outputs=sig_outputs, method_name=method_name
        )
        metagraph.signature_def[DEFAULT_SERVING_SIGNATURE_DEF_KEY].CopyFrom(
            signature_def
        )
        # Just no need to add input and output nodes.
        train_op_collection = meta_graph_pb2.CollectionDef(
            node_list=meta_graph_pb2.CollectionDef.NodeList(value=extra_reserved_nodes)
        )
    else:
        train_op_collection = meta_graph_pb2.CollectionDef(
            node_list=meta_graph_pb2.CollectionDef.NodeList(
                value=input_nodes + output_nodes + extra_reserved_nodes
            )
        )
    # Add a collection "train_op" so that Grappler knows the inputs and outputs.
    metagraph.collection_def["train_op"].CopyFrom(train_op_collection)

    return metagraph
