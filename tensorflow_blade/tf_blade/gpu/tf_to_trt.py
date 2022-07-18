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
import logging
from typing import Any, Iterator, List, Tuple

import onnx
from tensorflow.core.framework import attr_value_pb2, tensor_shape_pb2

try:
    from tf_blade._tf_blade import tensorrt  # type: ignore
except ImportError:
    pass
from tf_blade.util.simple_graph import GraphDefPartitioner
from tf_blade.util.tf2onnx_import_helper import tf2onnx
from tf_blade.util.tf_conversion_util import (
    TRT_SUPPORTED_LIST,
    get_subgraph_test_inputs_shapes,
)
from tf_blade.util.tf_graph_transform_util import OpType, set_attr_byte
from tf_blade.util.tf_import_helper import tf

Shape = List[int]


@contextlib.contextmanager
def builder_flags_context(flags: int) -> Iterator[int]:
    old_flags = tensorrt.set_builder_flags(flags)
    try:
        yield  # type: ignore
    finally:
        tensorrt.set_builder_flags(old_flags)


class Tf2TrtOpt:
    OPT_CUSTOM_OP_TYPE = "BladeTrtEngine"

    def __init__(
        self,
        supported_list: List[str] = list(TRT_SUPPORTED_LIST),
        minimum_segment_size: int = 50,
        dynamic_opt: bool = False,
        dump_dir: str = '',
    ) -> None:
        super().__init__()
        self._supported_list = supported_list
        self._minimum_segment_size = minimum_segment_size
        self._dynamic_shape_opt_enabled = dynamic_opt
        self._dump_dir = dump_dir
        self._subgraph_inputs_shapes: List[List[List[Shape]]] = list()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _insert_trt_engine_op(
        self,
        index: int,
        main_graph: tf.GraphDef,
        engine_bytes: bytes,
        new_input_names: List[str],
        subgraph_outputs: List[str],
    ) -> None:
        subgraph_op = "subgraph_{}".format(index)
        for node in main_graph.node:
            if node.op == subgraph_op:
                node.op = Tf2TrtOpt.OPT_CUSTOM_OP_TYPE
                node.attr.pop("input_names", None)
                set_attr_byte(node, "engine_bytes", engine_bytes)
                node.attr["input_names"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        list=attr_value_pb2.AttrValue.ListValue(
                            s=[(o + ":0").encode("utf-8") for o in new_input_names]
                        )
                    )
                )
                node.attr["output_names"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        list=attr_value_pb2.AttrValue.ListValue(
                            s=[(o + ":0").encode("utf-8") for o in subgraph_outputs]
                        )
                    )
                )
                # Trt op hold a func attr, make Function with name as f"{subgraph_op}"
                # reachable, then grappler will not prune the Function
                node.attr["fallback_function"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        func=attr_value_pb2.NameAttrList(name=subgraph_op)
                    )
                )

    def _convert_subgraph_to_onnx(
        self,
        index: int,
        subgraph: tf.GraphDef,
        new_input_names: List[str],
        subgraph_outputs: List[str],
        dynamic_shapes: List[Shape],
        opset: int,
    ) -> onnx.ModelProto:
        for node in subgraph.node:
            if node.op == OpType.PLACEHOLDER.value:
                shape = dynamic_shapes[new_input_names.index(node.name)]
                node.attr["shape"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        shape=tensor_shape_pb2.TensorShapeProto(
                            dim=[
                                tensor_shape_pb2.TensorShapeProto.Dim(size=d)
                                for d in shape
                            ]
                        )
                    )
                )
                node.attr.pop("_output_shapes", None)
        model_proto, _ = tf2onnx.convert.from_graph_def(
            subgraph,
            input_names=[name + ":0" for name in new_input_names],  # :0 is needed
            output_names=[name + ":0" for name in subgraph_outputs],
            opset=opset,
        )
        if self._dump_dir:
            logging.info(f"Dumping subgraph model to {self._dump_dir}")
            with tf.io.gfile.GFile(
                f"{self._dump_dir}/{self.name}_subgraph_{index}.onnx", "wb",
            ) as f:
                f.write(model_proto.SerializeToString())
            with tf.io.gfile.GFile(
                f"{self._dump_dir}/{self.name}_subgraph_{index}.pb", "wb",
            ) as f:
                f.write(subgraph.SerializeToString())
        return model_proto

    def _build_tensorrt_engine(
        self,
        model_proto: onnx.ModelProto,
        disable_fp16: bool,
        shapes_min: List[Shape],
        shapes_max: List[Shape],
        dynamic_shapes: List[Shape],
        shapes_opt: List[List[Shape]],
    ) -> bytes:
        dynamic_settings = tensorrt.TrtDynamicRanges()
        if self._dynamic_shape_opt_enabled:
            dynamic_settings = tensorrt.TrtDynamicRanges(
                shapes_min, shapes_max, dynamic_shapes, shapes_opt
            )
        # try to convert onnx to tensorrt engine
        flags = tensorrt.get_builder_flags()
        # enable fp16
        if not disable_fp16:
            flags = 1 << int(tensorrt.BuilderFlag.FP16)
            logging.info(f"Fp16 optimization enabled for tensorrt in {self.name} pass")
        with builder_flags_context(flags):
            engine_bytes, type_map, build_flags = tensorrt.cvt_onnx_to_tensorrt(
                model_proto.SerializeToString(),
                dynamic_shapes,
                dynamic_settings,
                dict(),
            )
        return engine_bytes  # type: ignore

    # subgraph_test_inputs_shapes_list: Each subgraph"s each input"s shapes for each test data
    # index: subgraph"s index
    # ori_input_names: subgraphs" input names List
    # For each subgraph"s certain input, if input shapes for each test data are the not same,
    # the corresponding dimension should be set to -1
    # This info is needed when trt doing dynamic shape tuning
    def _get_dynamic_tuning_shapes_info(
        self,
        subgraph_test_inputs_shapes_list: List[List[List[Shape]]],
        index: int,
        ori_input_names: List[List[str]],
    ) -> Tuple[List[Shape], List[Shape], List[Shape], List[List[Shape]]]:
        # determin each inputs" shapes
        dynamic_shapes = list()
        shapes_min = list()
        shapes_max = list()
        shapes_opt = list()
        if self._dynamic_shape_opt_enabled:
            shapes_min = subgraph_test_inputs_shapes_list[0][index]
            shapes_max = subgraph_test_inputs_shapes_list[1][index]
            for item in subgraph_test_inputs_shapes_list[2:]:
                shapes_opt.append(item[index])
            for j in range(len(ori_input_names[index])):
                shape = list()
                for k in range(len(shapes_min[j])):
                    if shapes_min[j][k] == shapes_max[j][k]:
                        shape.append(shapes_min[j][k])
                    else:
                        shape.append(-1)
                dynamic_shapes.append(shape)
        else:
            dynamic_shapes = subgraph_test_inputs_shapes_list[0][index]
        return (shapes_min, shapes_max, dynamic_shapes, shapes_opt)

    def _post_process_function(self, i: int, main_graph: tf.GraphDef) -> None:
        subgraph_name = f"subgraph_{i}"
        # when MetaOptimizer optimize Function from tf.GraphDef
        # it requires all function"s attr should only have the type "type"
        # Here we delete all the attr to avoid the
        # type check error in grappler::MakeGrapplerFunctionItem
        for func in main_graph.library.function:
            if func.signature.name == subgraph_name:
                del func.signature.attr[:]
        for n in main_graph.node:
            if n.op == subgraph_name:
                n.attr.clear()

    def _process_subgraph(
        self,
        subgraph_test_inputs_shapes_list: List[List[List[Shape]]],
        i: int,
        ori_input_names: List[List[str]],
        subgraph: tf.GraphDef,
        new_input_names: List[str],
        subgraph_outputs: List[str],
        main_graph: tf.GraphDef,
        disable_fp16: bool,
        opset: int,
    ) -> None:
        (
            shapes_min,
            shapes_max,
            dynamic_shapes,
            shapes_opt,
        ) = self._get_dynamic_tuning_shapes_info(
            subgraph_test_inputs_shapes_list, i, ori_input_names
        )

        # use opt_shapes[0] for onnx conversion
        model_proto = self._convert_subgraph_to_onnx(
            i, subgraph, new_input_names, subgraph_outputs, dynamic_shapes, opset,
        )
        engine_bytes = self._build_tensorrt_engine(
            model_proto,
            disable_fp16,
            shapes_min,
            shapes_max,
            dynamic_shapes,
            shapes_opt,
        )

        if len(engine_bytes) == 0:
            logging.info(f"Tensorrt optimization failed for subgraph_{i}.")
        else:
            self._insert_trt_engine_op(
                i, main_graph, engine_bytes, new_input_names, subgraph_outputs
            )
        self._post_process_function(i, main_graph)

    def optimize_graph_def(
        self,
        graph_def: tf.GraphDef,
        model_outputs: List[str],
        test_data: List[Any],
        disable_fp16: bool,
        opset: int = 11,
    ) -> tf.GraphDef:
        try:
            (
                main_graph,
                subgraphs,
                ori_input_names,
                new_input_names,
                subgraph_outputs,
            ) = GraphDefPartitioner(
                graph_def,
                set(self._supported_list),
                skip_while_loop=True,
                outputs=model_outputs,
                minimum_segment_size=self._minimum_segment_size,
            ).generate_subgraph_from_segment(
                add_function_def=True, replicate_const_inputs=True,
            )
        except Exception as err:
            raise Exception(f"Failed to partition graph def due to {str(err)}")

        self._subgraph_inputs_shapes = get_subgraph_test_inputs_shapes(
            graph_def, test_data, len(test_data), ori_input_names
        )
        if len(self._subgraph_inputs_shapes) == 0:
            raise Exception("Failed to get subgraph input shapes")
        # Convert each subgraphs to onnx and build tensorrt engine
        for i in range(len(subgraphs)):
            self._process_subgraph(
                self._subgraph_inputs_shapes,
                i,
                ori_input_names,
                subgraphs[i],
                new_input_names[i],
                subgraph_outputs[i],
                main_graph,
                disable_fp16,
                opset,
            )
        return main_graph
