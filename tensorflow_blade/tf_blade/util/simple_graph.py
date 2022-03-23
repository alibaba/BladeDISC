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
from typing import Dict, List, Optional, Set, Tuple

from tensorflow.core.framework import attr_value_pb2, function_pb2
from tensorflow.python.framework import function, graph_to_function_def

from tf_blade.util import tf_util
from tf_blade.util.tf_import_helper import tf


class SimpleNode:
    def __init__(
        self,
        name: str = "",
        op: str = "",
        inputs: List[str] = [],
        output_nodes: List[str] = [],
        tensors: Dict[str, List[str]] = {},
    ):
        self.name = name
        self.op = op
        self.inputs = inputs
        # Input tensors.
        self.inputs_tensors = [tf_util.get_canonical_tensor_name(n) for n in inputs]
        # Output nodes.
        self.output_nodes = output_nodes.copy()
        # Mapping from output tensor name to list of nodes that consume this tensor.
        self.tensors = tensors.copy()

    @property
    def num_inputs(self) -> int:
        return len(self.inputs_tensors)

    @property
    def num_outputs(self) -> int:
        return len(self.output_nodes)

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    @property
    def input_nodes(self) -> List[str]:
        return [tf_util.tensor_name_to_node_name(inp) for inp in self.inputs_tensors]

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SimpleNode):
            return False
        return (
            self.name == o.name
            and self.op == o.op
            and self.inputs_tensors == o.inputs_tensors
            and self.output_nodes == o.output_nodes
            and self.tensors == o.tensors
        )

    def __str__(self) -> str:
        s = ""
        s += "name          : {}\n".format(self.name)
        s += "op            : {}\n".format(self.op)
        s += "inputs_tensors: {}\n".format(self.inputs_tensors)
        s += "ouput_nodes   : {}\n".format(self.output_nodes)
        s += "tensors       : {}\n".format(self.tensors)
        return s


class SimpleGraph:
    def __init__(self, graph_def: tf.GraphDef):
        self._nodes = [
            SimpleNode(name=n.name, op=n.op, inputs=list(n.input))
            for n in graph_def.node
        ]
        self._name2index = {n.name: i for i, n in enumerate(graph_def.node)}
        self._graph_def = graph_def
        self._default_tensor_info_map: Dict[str, tf_util.TensorInfo] = {}

        for node in graph_def.node:
            for inp in node.input:
                inp_node_name = tf_util.tensor_name_to_node_name(inp)
                inp_tensor_name = tf_util.get_canonical_tensor_name(inp)
                if inp_node_name not in self._name2index:
                    raise Exception(
                        f"SimpleNode {node.name}: Unknown input node {inp}"
                    )
                input_node = self._nodes[self._name2index[inp_node_name]]
                # update input node"s [output_node, ..] list
                input_node.output_nodes.append(node.name)
                # update input node"s {tensor: output_node, ..} dictionary
                #   TODO: we are missing Graph final output node"s tensors,
                #   but it is not possible to inspect how many tensors inside
                #   it, therefore we currently ignore it.
                if inp_tensor_name not in input_node.tensors:
                    input_node.tensors[inp_tensor_name] = [node.name]
                else:
                    input_node.tensors[inp_tensor_name].append(node.name)

    @property
    def num_nodes(self) -> int:
        """Get total number of nodes in graph."""
        return len(self._nodes)

    @property
    def nodes(self) -> List[SimpleNode]:
        """Get all nodes in graph."""
        return self._nodes

    def name2index(self, name: str) -> int:
        """Get index of node."""
        if name not in self._name2index:
            error_msg = "Node {} not exists".format(name)
            logging.error(error_msg)
            raise Exception(error_msg)
        return self._name2index[name]

    def node(self, idx: int) -> SimpleNode:
        """Get node with given index."""
        if idx >= len(self._nodes):
            error_msg = "Node index {} out of range".format(idx)
            logging.error(error_msg)
            raise Exception(error_msg)
        return self._nodes[idx]

    def name2node(self, name: str) -> SimpleNode:
        """Get node by name."""
        return self.node(self._name2index[name])

    def input_nodes(self, blacklist: List = ["Const"]) -> List[str]:
        """Get names of input nodes"""
        return [
            n.name for n in self._nodes if n.num_inputs == 0 and n.op not in blacklist
        ]

    def output_nodes(self) -> List[str]:
        """Get names of output nodes, which are those without downstream."""
        return [n.name for n in self._nodes if n.num_outputs == 0]

    def input_nodes_index(self, node_idx: int) -> List[int]:
        """Get indexes of input nodes for node of given index."""
        return [self._name2index[n] for n in self._nodes[node_idx].input_nodes]

    def get_simple_node_by_name(self, name: str) -> SimpleNode:
        node_name = tf_util.get_node_name(name)[0]
        if node_name not in self._name2index:
            raise Exception(f"Unknown node name: {node_name}")
        return self.node(self.name2index(node_name))

    def get_node_by_name(self, name: str) -> tf.NodeDef:
        node_name = tf_util.get_node_name(name)[0]
        if node_name not in self._name2index:
            raise Exception(f"Unknown node name: {node_name}")
        idx = self._name2index[node_name]
        if idx >= len(self._graph_def.node):
            raise Exception(f"Unknown node name: {node_name}")
        return self._graph_def.node[idx]

    def topological_sort(self, reverse: bool = False) -> List[int]:
        """Sort given SimpleGraph in topological order.

        Parameters
        ----------
        reverse : bool = False
            Set True to list op from output to input.

        Returns
        -------
        List[int]
            Index of graph node in topological order.
        """
        ready = []
        pending_count = []
        ordered = []
        # Parse the inputs for each node
        for i, node in enumerate(self._nodes):
            if node.op == "Merge" or node.op == "RefMerge":
                num_control_edges = sum(
                    1 for inp in node.inputs_tensors if inp.startswith("^")
                )
                pending_count.append(num_control_edges + 1)
            else:
                pending_count.append(len(node.inputs_tensors))
            if len(node.inputs_tensors) == 0:
                ready.append(i)
        processed = 0
        # Process the NodeDefs in topological order
        # Code above sets this up by filling in ready_ with nodes that have no
        # inputs, pending_counts_ with the number of inputs for each node and
        # outputs_ with the outputs of each node
        while len(ready) != 0:
            o = ready.pop(-1)
            ordered.append(o)
            processed += 1
            # Update pending_count for outputs.
            for out in self._nodes[o].output_nodes:
                pending_count[self._name2index[out]] -= 1
                if pending_count[self._name2index[out]] == 0:
                    ready.append(self._name2index[out])
        if processed < self.num_nodes:
            raise Exception(f"{self.num_nodes-processed} nodes in a cycle.")
        if reverse:
            ordered.reverse()
        return ordered

    def is_reachable(self, src_idx: int, target_idx: Set[int]) -> bool:
        """Check if any nodes with index in `target_idx` can be reached from `src_idx` node.

        Parameters
        ----------
        src_idx : int
            Index of source node.
        target_idx : Set[int]
            Indexes of target nodes.

        Returns
        -------
        bool
            True if any node in `target_idx` is reachable from source node.
        """
        if self.get_reachable(target_idx, src_idx):
            return True
        return False

    def get_reachable(
        self, target_idx: Set[int], start_at: Optional[int] = None
    ) -> Set[int]:
        """Get all nodes that can reach to the target_idx.

        Parameters
        ----------
        target_idx : Set[int]
            The index of target node.
        start_at : Optional[int] = None
            This option is to check if a start point can reach any nodes in the
            target_idx set.

        Returns
        -------
        Set[int]
            A set of the index of the nodes that can reach to any of the target nodes.
        """
        stack = list(target_idx)
        reachable = set()
        while len(stack) > 0:
            idx = stack.pop(-1)
            if idx in reachable:
                continue
            reachable.add(idx)
            if start_at is not None and idx == start_at:
                return reachable
            stack.extend(
                [self._name2index[inp] for inp in self._nodes[idx].input_nodes]
            )
        if start_at is not None:
            return set()
        return reachable

    def get_tensor_info(self) -> Dict[str, tf_util.TensorInfo]:
        if not self._default_tensor_info_map:
            self._default_tensor_info_map = tf_util.get_tensor_info_mapping(
                self._graph_def
            )
        return self._default_tensor_info_map


# tf contrib tensorrt supported list
# Mean is converted into a UFF Reduce layer
# Reduce layer will be supported in our future release but we can"t disclsoure schedule information.
# Although TensorRT has a unary layer, UFF parser doesn"t support it.
# https://devtalk.nvidia.com/default/topic/1028954/dimension-error-in-uff-parser/
_SEGMENT_SUPPORTED_OP: Set[str] = {
    "Identity",
    "Snapshot",
    "Const",
    "Conv2D",
    "MaxPool",
    "BiasAdd",
    "Relu",
    "Add",
    "AddV2",
    "Mul",
    "Sub",
    "Rsqrt",
    "Pad",
    "Mean",
    "AvgPool",
    "ConcatV2_buggy",
    "DepthwiseConv2dNative",
    "FusedBatchNorm",
    "FusedBatchNormV2",
}


class GraphSegment:
    """
    Parameters
    ----------
    graph : SimpleGraph
        The original main graph.

    nodes_idx : Set[int]
        Index of nodes that belong to this segment.

    required_outputs : List[str] = []
        Manually specified outputs. There"re 3 kinds of outputs:
        1. If a node"s output is outside this segment(A cross boundary edge)
        2. If a node has no output nodes(Maybe the node itself is the output of the main graph)
        3. If a node is in required_outputs(It is manually set to be the output node)
    """

    def __init__(
        self, graph: SimpleGraph, nodes_idx: Set[int], required_outputs: List[str] = [],
    ) -> None:
        self._graph = graph
        self._nodes_idx = nodes_idx
        self._required_outputs = required_outputs

        # The generated graphdef of this GraphSegment
        self._subgraph_graphdef: Optional[tf.GraphDef] = None
        self._subgraph_input_names: List[str] = []
        self._subgraph_ctl_input_names: Set[str] = set()
        self._subgraph_output_nodes: List[tf.NodeDef] = []
        # The original input names of this subgraph
        self._updated_ori_input_names: List[str] = []
        self._subgraph_functiondef: Optional[function_pb2.FunctionDef] = None
        self.node_names = [self._graph.node(i).name for i in self._nodes_idx]

    # @property
    # def node_names(self) -> List[str]:
    #     return [self._graph.node(i).name for i in self._nodes_idx]

    def output_nodes(self) -> List[str]:
        """Find nodes in graph segment that output to outside of segment.
        See the explaination of GraphSegment"s `required_outputs`.
        """
        segment_node_names = self.node_names

        out_node_names = []
        for idx in self._nodes_idx:
            node = self._graph.node(idx)
            if (
                any(
                    [
                        (outname not in segment_node_names)
                        for outname in node.output_nodes
                    ]
                )
                or len(node.output_nodes) == 0
                or node.name in self._required_outputs
            ):
                out_node_names.append(node.name)

        return out_node_names

    def input_tensors(self) -> List[str]:
        """Find tensors that come into graph segment as input."""
        segment_op_names = self.node_names
        inp_tensor_names = []
        # Use this to filter out duplicate inputs
        visited = set()
        for idx in self._nodes_idx:
            for inp_name in self._graph.node(idx).inputs_tensors:
                inp_node = tf_util.tensor_name_to_node_name(inp_name)
                if inp_node not in segment_op_names and inp_name not in visited:
                    inp_tensor_names.append(inp_name)
                    visited.add(inp_name)
        return inp_tensor_names

    def output_offsets(self) -> List[int]:
        """
        Used to identity the indices of a certain segment"s outputs.
        For example, the out_node_names is ["a", "b"]
        "a" has 2 outputs that reach out of the segment,
        and "b" has 3 outputs that reach out of the segment.
        Then the return value is [0, 2, 5], meaning given the order of "a" and "b",
        the segment outputs 5 tensors, in which the 0th and 1st related to "a",
        and 2nd, 3rd, 4th related to "b"
        """
        out_node_names = self.output_nodes()
        out_node_outputs: Dict[str, Set[str]] = {x: set() for x in out_node_names}
        segment_node_names = self.node_names
        for node in self._graph.nodes:
            if node.name not in segment_node_names:
                for inp in node.inputs_tensors:
                    inp_node_name = tf_util.tensor_name_to_node_name(inp)
                    if inp_node_name in out_node_names:
                        out_node_outputs[inp_node_name].add(inp)
        offsets = [0]
        for i, out in enumerate(out_node_names):
            offset = len(out_node_outputs[out])
            offsets.append(offsets[i] + (offset if offset > 0 else 1))
        return offsets

    def to_graphdef(
        self, seg_id: int = 0, replicate_const_inputs: bool = False
    ) -> Tuple[tf.GraphDef, List[str], Set[str], List[tf.NodeDef], List[str]]:
        """Generate subgraph GraphDef from this segment.

        Parameters
        ----------
        seg_id : int = 0
            The id of this segment/subgraph.
        replicate_const_inputs : bool = False
            If an input tensor is a Const op, set True to make a replicate in this
            subgraph.

        Returns
        -------
        tf.GraphDef
            The generated subgraph GraphDef.
        List[str]
            The updated new data input names.
        Set[str]
            The updated new control input names.
        List[tf.NodeDef]
            The output identity node in the new GraphDef
        List[str]
            The the original input names of this subgraph.
        """
        if self._subgraph_graphdef:
            return (
                self._subgraph_graphdef,
                self._subgraph_input_names,
                self._subgraph_ctl_input_names,
                self._subgraph_output_nodes,
                self._updated_ori_input_names,
            )

        self._subgraph_graphdef = tf.GraphDef()

        # First: Add inputs to subgraph
        inp_tensor_replace_dict = self._process_subgraph_inputs(
            replicate_const_inputs, seg_id
        )

        # Second: Copy the other nodes to subgraph
        self._copy_graph_and_replace_input(inp_tensor_replace_dict)

        # Finally: Add Identity nodes as the output nodes of the subgraph
        self._process_subgraph_outputs(seg_id)

        return (
            self._subgraph_graphdef,
            self._subgraph_input_names,
            self._subgraph_ctl_input_names,
            self._subgraph_output_nodes,
            self._updated_ori_input_names,
        )

    def to_functiondef(
        self, seg_id: int = 0, replicate_const_inputs: bool = False
    ) -> function_pb2.FunctionDef:
        """Generate subgraph FunctionDef from this segment.

        Parameters
        ----------
        seg_id : int = 0
            The id of this segment/subgraph.
        replicate_const_inputs : bool = False
            If an input tensor is a Const op, set True to make a replicate in this
            subgraph.

        Returns
        -------
        tf.FunctionDef
            The generated subgraph FunctionDef.
        """

        if self._subgraph_functiondef:
            return self._subgraph_functiondef

        # Assert subgraph GraphDef has been generated
        self.to_graphdef(seg_id, replicate_const_inputs)

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(self._subgraph_graphdef, name="")
            self._subgraph_functiondef = graph_to_function_def.graph_to_function_def(
                graph,
                graph.get_operations(),
                [
                    graph.get_tensor_by_name(i + ":0")
                    for i in self._subgraph_input_names
                ],
                [
                    graph.get_tensor_by_name(i.name + ":0")
                    for i in self._subgraph_output_nodes
                ],
            )
        self._subgraph_functiondef.signature.name = "subgraph_{}".format(seg_id)
        attr = self._subgraph_functiondef.signature.attr.add()
        attr.name = "input_names"
        attr.type = "list(string)"
        attr.default_value.CopyFrom(
            attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue())
        )
        attr = self._subgraph_functiondef.signature.attr.add()
        attr.name = "input_shapes"
        attr.type = "list(shape)"
        attr.default_value.CopyFrom(
            attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue())
        )
        attr = self._subgraph_functiondef.signature.attr.add()
        attr.name = "Tin"
        attr.type = "list(type)"
        attr.default_value.CopyFrom(
            attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue())
        )
        attr = self._subgraph_functiondef.signature.attr.add()
        attr.name = "Tout"
        attr.type = "list(type)"
        attr.default_value.CopyFrom(
            attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue())
        )

        return self._subgraph_functiondef

    def _process_subgraph_inputs(
        self, replicate_const_inputs: bool, seg_id: int
    ) -> Dict[str, str]:
        """Add input tensors to the subgraph."""
        from tf_blade.util.tf_graph_transform_util import add_placeholder

        assert self._subgraph_graphdef is not None
        self._subgraph_input_names = []
        self._updated_ori_input_names = []
        self._subgraph_ctl_input_names = set()

        tensor_info_map = self._graph.get_tensor_info()
        inp_tensor_replace_dict = {}
        # control inputs to input nodes should be redirects to the whole segment.
        for i, inp in enumerate(self.input_tensors()):
            if inp.startswith("^"):
                self._subgraph_ctl_input_names.add(inp)
                continue
            inp_clean = inp.strip().strip("^")
            info = tensor_info_map.get(
                inp_clean if ":" in inp_clean else inp_clean + ":0", None
            )
            inode = self._graph.get_node_by_name(tf_util.tensor_name_to_node_name(inp))
            if replicate_const_inputs and inode.op == "Const":
                # Make a copy of the original const
                # This will not be as the subgraph inputs
                new_const = self._subgraph_graphdef.node.add()
                new_const.MergeFrom(inode)
            elif info is not None:
                placeholder_name = "subgraph_{}_placeholder_{}".format(seg_id, i)
                add_placeholder(
                    self._subgraph_graphdef, info.dtype, info.shape, placeholder_name,
                )
                inp_tensor_replace_dict[inp] = placeholder_name
                self._subgraph_input_names.append(placeholder_name)
                self._updated_ori_input_names.append(inp)

        return inp_tensor_replace_dict

    def _copy_graph_and_replace_input(
        self, inp_tensor_replace_dict: Dict[str, str]
    ) -> None:
        """Copy node from the original graph and replace the input names."""
        assert self._subgraph_graphdef is not None

        for node in self._graph._graph_def.node:
            if node.name in self.node_names:
                copy_node = self._subgraph_graphdef.node.add()
                copy_node.MergeFrom(node)
                new_inputs = []
                for i, inp in enumerate(copy_node.input):
                    if inp not in self._subgraph_ctl_input_names:
                        if inp in inp_tensor_replace_dict:
                            new_inputs.append(inp_tensor_replace_dict[inp])
                        elif (inp + ":0") in inp_tensor_replace_dict:
                            new_inputs.append(inp_tensor_replace_dict[inp + ":0"])
                        else:
                            new_inputs.append(inp)
                copy_node.input[:] = new_inputs

    def _process_subgraph_outputs(self, seg_id: int) -> None:
        """Add output nodes to this subgraph."""
        from tf_blade.util.tf_graph_transform_util import add_identity

        self._subgraph_output_nodes = []

        output_node_offsets = self.output_offsets()
        tensor_info_map = self._graph.get_tensor_info()

        for i, out_node_name in enumerate(self.output_nodes()):
            for j in range(output_node_offsets[i], output_node_offsets[i + 1]):
                id_input_name = (
                    (out_node_name + ":{}".format(j))
                    if output_node_offsets[i + 1] - output_node_offsets[i] > 1
                    else out_node_name
                )
                id_name = (
                    "subgraph_{}-".format(seg_id)
                    + out_node_name
                    + "-{}".format(j - output_node_offsets[i])
                )
                output_tensor_name = (
                    id_input_name if ":" in id_input_name else id_input_name + ":0"
                )
                id_node = add_identity(
                    self._subgraph_graphdef,
                    id_input_name,
                    id_name,
                    tensor_info_map[output_tensor_name].dtype,
                )
                self._subgraph_output_nodes.append(id_node)


class GraphDefPartitioner:
    """
    Parameters
    ----------
    graph_def : tf.GraphDef
        The target GraphDef to be partitioned.
    supported_list : Set[str] = _SEGMENT_SUPPORTED_OP
        A set contains all supported *op type*.
    black_list : Set[str] = set()
        A set contains the *name* of nodes which should not be included in any subgraph.
    minimum_segment_size : int = 2
        A subgraph segment should at least contains such number of nodes.
    skip_while_loop : bool = False
        Set True to skip the nodes which are inside a while loop.
    outputs : List[str] = []
        If not provid, all nodes without output nodes will be set as the output of this
        graph.
    """

    def __init__(
        self,
        graph_def: tf.GraphDef,
        supported_list: Set[str] = _SEGMENT_SUPPORTED_OP,
        black_list: Set[str] = set(),
        minimum_segment_size: int = 2,
        skip_while_loop: bool = False,
        outputs: List[str] = [],
    ) -> None:
        self.ori_graph = SimpleGraph(graph_def)
        self.main_graph_outputs = outputs if outputs else self.ori_graph.output_nodes()
        self.main_graph_reachable = self.ori_graph.get_reachable(
            {self.ori_graph.name2index(o) for o in self.main_graph_outputs}
        )
        self.graph_segment_list: List[GraphSegment] = self._segment_graph(
            supported_list, black_list, minimum_segment_size, skip_while_loop,
        )

        self._partitioned_main_graph: Optional[tf.GraphDef] = None
        self._partitioned_subgraphs: List[tf.GraphDef] = []
        self._partitioned_subgraphs_ori_inputs: List[List[str]] = []
        self._partitioned_subgraphs_new_inputs: List[List[str]] = []
        self._partitioned_subgraphs_outputs: List[List[str]] = []

    def _segment_graph(  # noqa: C901
        self,
        supported_list: Set[str],
        black_list: Set[str],
        minimum_segment_size: int,
        skip_while_loop: bool,
    ) -> List[GraphSegment]:
        """Generate graph segments.

        Parameters
        ----------
        supported_list: Set[str]
            A set contains all supported *op type*.
        black_list: Set[str]
            A set contains the *name* of nodes which should not be included in any subgraph.
        minimum_segment_size : int = 2
            A subgraph segment should at least contains such number of nodes.
        skip_while_loop : bool = False
            Set True to skip the nodes which are inside a while loop.

        Returns
        -------
        List[GraphSegment]
        """
        visited = [False] * self.ori_graph.num_nodes

        segments = []
        for o in self.ori_graph.topological_sort(reverse=True):
            if o not in self.main_graph_reachable:
                continue
            if visited[o]:
                continue
            seg_nodes: Set[int] = set()
            num_op_nodes = 0
            queue = [o]
            while len(queue) > 0:
                idx = queue.pop(0)
                if (
                    self.ori_graph.node(idx).op not in supported_list
                    or self.ori_graph.node(idx).name in black_list
                    or (
                        skip_while_loop
                        # Skip the node inside a while loop
                        and self.ori_graph.node(idx).name.count("while/")
                    )
                ):
                    visited[idx] = True
                    continue
                dfs_start_nodes = {
                    inp_idx
                    for nidx in seg_nodes
                    for inp_idx in self.ori_graph.input_nodes_index(nidx)
                    if inp_idx != idx and inp_idx not in seg_nodes
                }

                if len(dfs_start_nodes) == 0 or not self.ori_graph.is_reachable(
                    idx, dfs_start_nodes
                ):
                    seg_nodes.add(idx)
                    if self.ori_graph.node(idx).op not in ["Identity", "Placeholder"]:
                        num_op_nodes += 1
                    visited[idx] = True
                    for inp in self.ori_graph.input_nodes_index(idx):
                        if inp not in queue and not visited[inp]:
                            queue.append(inp)
            if num_op_nodes >= minimum_segment_size:
                segments.append(
                    GraphSegment(
                        self.ori_graph,
                        seg_nodes,
                        required_outputs=self.main_graph_outputs,
                    )
                )
        return segments

    def generate_subgraph_from_segment(  # noqa: C901
        self, add_function_def: bool = False, replicate_const_inputs: bool = False
    ) -> Tuple[
        tf.GraphDef,
        List[tf.GraphDef],
        List[List[str]],
        List[List[str]],
        List[List[str]],
    ]:
        """Split a GraphDef to a main GraphDef and several sub GraphDef according to segments.

        Parameters
        ----------
        add_function_def : bool = False
            Set True to pack the subgraphs as FunctionDef to the Main graph. This can be used
            as a fallback options.
        replicate_const_inputs : bool = False
            If an input tensor is a Const op, set True to make a replicate in this
            subgraph.

        Return
        ------
        main_graph : tf.GraphDef
            The updated main graph, which has the same input/output nodes as the original
            GraphDef with segments as subgraph nodes.
        subgraphs : List[tf.GraphDef]
            List of all subgraph GraphDefs. Each is a complete GraphDef.
        subgraph_ori_input_names : List[List[str]]
            The original input names of each subgraph. This can be used to generate the new
            tests data for those subgraphs.
        subgraph_new_input_names : List[List[str]]
            The new input names of each subgraph.
        subgraph_output_names : List[List[str]]
            The name of output nodes of subgraph.
        """
        from tf_blade.util.tf_graph_transform_util import add_identity

        if self._partitioned_main_graph:
            return (
                self._partitioned_main_graph,
                self._partitioned_subgraphs,
                self._partitioned_subgraphs_ori_inputs,
                self._partitioned_subgraphs_new_inputs,
                self._partitioned_subgraphs_outputs,
            )

        tensor_info_map = self.ori_graph.get_tensor_info()

        self._partitioned_main_graph = tf.GraphDef()
        self._partitioned_subgraphs = []
        self._partitioned_subgraphs_ori_inputs = []
        self._partitioned_subgraphs_new_inputs = []

        visited: Set[str] = set()
        subgraph_output_node_names: List[List[str]] = []
        subgraph_output_node_offsets: List[List[int]] = []

        subgraph_nodes_in_main_graph: List[tf.NodeDef] = []
        # Process each segment to a subgraph
        for seg_id, segment in enumerate(self.graph_segment_list):
            visited.update(segment.node_names)
            subgraph_output_node_offsets.append(segment.output_offsets())

            # Generate subgraph GraphDef from segment
            (
                subgraph,
                new_input_names,
                new_ctl_input_names,
                subgraph_outputs,
                updated_ori_input_names,
            ) = segment.to_graphdef(seg_id, replicate_const_inputs)
            self._partitioned_subgraphs.append(subgraph)
            self._partitioned_subgraphs_new_inputs.append(new_input_names)
            self._partitioned_subgraphs_ori_inputs.append(updated_ori_input_names)
            self._partitioned_subgraphs_outputs.append(
                [node.name for node in subgraph_outputs]
            )

            # Add "subgraph" node to the updated main graph
            subgraph_node = self._partitioned_main_graph.node.add()
            subgraph_node.name = "subgraph_{}".format(seg_id)
            if add_function_def:
                subgraph_node.op = "subgraph_{}".format(seg_id)
            # The original input tensor names, may be changed if the former node has
            # been segmented into another subgraph
            subgraph_node.input.extend(updated_ori_input_names)
            subgraph_node.input.extend(new_ctl_input_names)
            # control inputs are not added to attributes `input_names".
            subgraph_node.attr["input_names"].CopyFrom(
                attr_value_pb2.AttrValue(
                    list=attr_value_pb2.AttrValue.ListValue(
                        s=[i.encode("utf-8") for i in new_input_names]
                    )
                )
            )
            inp_tensor_type = []
            for inp in updated_ori_input_names:
                inp_node = self.ori_graph.get_node_by_name(
                    inp.strip().strip("^").split(":")[0]
                )
                if "dtype" in inp_node.attr:
                    inp_tensor_type.append(inp_node.attr["dtype"].type)
                elif "T" in inp_node.attr:
                    inp_tensor_type.append(inp_node.attr["T"].type)
                elif "DstT" in inp_node.attr:
                    inp_tensor_type.append(inp_node.attr["DstT"].type)  # Cast
                elif "Tparams" in inp_node.attr:
                    inp_tensor_type.append(inp_node.attr["Tparams"].type)  # Gather
                elif "Tidx" in inp_node.attr:
                    inp_tensor_type.append(inp_node.attr["Tidx"].type)  # Range
                else:
                    raise RuntimeError(
                        "Cannot acquire type of input tensor: {}.".format(inp)
                    )
            subgraph_node.attr["Tin"].CopyFrom(
                attr_value_pb2.AttrValue(
                    list=attr_value_pb2.AttrValue.ListValue(type=inp_tensor_type)
                )
            )
            subgraph_node.attr["Tout"].CopyFrom(
                attr_value_pb2.AttrValue(
                    list=attr_value_pb2.AttrValue.ListValue(
                        type=[i.attr["T"].type for i in subgraph_outputs]
                    )
                )
            )
            subgraph_nodes_in_main_graph.append(subgraph_node)

            # Check if there is any node result needed by main graph
            output_node_names = segment.output_nodes()
            subgraph_output_node_names.append(output_node_names)
            for node in subgraph.node:
                if node.name in self.main_graph_outputs:
                    # Assert all outputs have been added a identity
                    add_identity(
                        self._partitioned_main_graph,
                        "subgraph_{}:{}".format(
                            seg_id, output_node_names.index(node.name)
                        ),
                        node.name,
                        tensor_info_map[
                            node.name if ":" in node.name else node.name + ":0"
                        ].dtype,
                    )

        def process_input_node_inside_a_subgraph(checked_node: tf.NodeDef) -> None:
            for i, inp in enumerate(checked_node.input):
                ctl_prefix = "^" if inp.startswith("^") else ""
                inp_node_name = inp.strip().strip("^").split(":")[0]

                # Check if the node input is from a subgraph
                subgraph_id = 0
                out_node_id = 0
                for out_node_names in subgraph_output_node_names:
                    if inp_node_name in out_node_names:
                        out_node_id = out_node_names.index(inp_node_name)
                        break
                    subgraph_id += 1
                # Input node found
                if subgraph_id < len(subgraph_output_node_names):
                    if ":" not in inp:
                        checked_node.input[i] = "{}subgraph_{}:{}".format(
                            ctl_prefix,
                            subgraph_id,
                            str(subgraph_output_node_offsets[subgraph_id][out_node_id]),
                        )
                    else:
                        checked_node.input[i] = "{}subgraph_{}:{}".format(
                            ctl_prefix,
                            subgraph_id,
                            str(
                                subgraph_output_node_offsets[subgraph_id][out_node_id]
                                + int(inp.split(":")[1])
                            ),
                        )
                    if subgraph_output_node_offsets[subgraph_id][-1] == 1:
                        assert checked_node.input[i].endswith(":0")
                        # remove ":0" for there"s only one output
                        checked_node.input[i] = checked_node.input[i][:-2]

        # Check again to fix the input if it has been segmented into another subgraph
        for subgraph_node in subgraph_nodes_in_main_graph:
            process_input_node_inside_a_subgraph(subgraph_node)

        # Process nodes that are not processed in any segments
        for node in self.ori_graph._graph_def.node:
            # Filter out the nodes that are not reachable to this outpus, or has been
            # included in a subgraph
            if (
                self.ori_graph.name2index(node.name) in self.main_graph_reachable
                and node.name not in visited
            ):
                visited.add(node.name)
                unfused_node = self._partitioned_main_graph.node.add()
                unfused_node.MergeFrom(node)
                # Check if any input is in a subgraph
                process_input_node_inside_a_subgraph(unfused_node)

        if add_function_def:
            fd_list = []
            for i, segment in enumerate(self.graph_segment_list):
                fd_list.append(
                    function._from_definition(
                        segment.to_functiondef(i, replicate_const_inputs)
                    )
                )

            tf.reset_default_graph()
            graph = tf.Graph()
            # The function is a runtime resource, so must be processed in tf.Graph()
            with graph.as_default():
                for i, segment in enumerate(self.graph_segment_list):
                    graph._add_function(fd_list[i])
                # Merge the main graph with these function def
                tf.import_graph_def(self._partitioned_main_graph, name="")

            # Update the partitioned main graph
            self._partitioned_main_graph = graph.as_graph_def()

        return (
            self._partitioned_main_graph,
            self._partitioned_subgraphs,
            self._partitioned_subgraphs_ori_inputs,
            self._partitioned_subgraphs_new_inputs,
            self._partitioned_subgraphs_outputs,
        )
