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
from typing import Any, List

from tf_blade.util.tf_import_helper import tf

Shape = List[int]

# This list is referenced to tf_trt supported ops in tensorflow 1.15
TRT_SUPPORTED_LIST = [
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "AddN",
    "AddV2",
    # "ArgMax", get error msg from trt that ArgMax cannot be supported
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "AvgPool",
    "BatchMatMul",
    "BatchMatMulV2",
    "BiasAdd",
    "Cast",
    "Ceil",
    "ClipByValue",
    "CombinedNonMaxSuppression",
    "ConcatV2",
    "Const",
    "Conv2D",
    "Conv2DBackpropInput",
    "Conv3D",
    "Conv3DBackpropInputV2",
    "Cos",
    "Cosh",
    "DepthToSpace",
    "DepthwiseConv2dNative",
    "Div",
    "Elu",
    "Erf",
    "Exp",
    "ExpandDims",
    "Fill",
    "Floor",
    "FloorDiv",
    "FusedBatchNorm",
    "FusedBatchNormV2",
    "FusedBatchNormV3",
    "FusedConv2DBiasActivation",
    "GatherV2",
    "Identity",
    "LeakyRelu",
    "Log",
    "MatMul",
    "Max",
    "Maximum",
    "MaxPool",
    "Mean",
    "Min",
    "Minimum",
    "Mul",
    "Neg",
    "Pack",
    "Pad",
    "Pow",
    "Prod",
    "RealDiv",
    "Reciprocal",
    "Relu",
    "Relu6",
    "Reshape",
    "Rsqrt",
    "Selu",
    "Shape",
    "Sigmoid",
    "Sin",
    "Sinh",
    "Slice",
    "Snapshot",
    "Softmax",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Split",
    "Sqrt",
    "Square",
    "SquaredDifference",
    "Squeeze",
    "StopGradient",
    "StridedSlice",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "TopKV2",
    "Transpose",
    "Unpack",
]

HIE_SUPPORTED_LIST = TRT_SUPPORTED_LIST + ["OneHot", "Range", "Erf", "TruncatedNormal"]


# List[List[List[Shape]]]
# 1st List for subgraphs
# 2nd List for test datas
# 3rd List for each inputs
def get_subgraph_test_inputs_shapes(
    graph_def: tf.GraphDef,
    test_data: List[Any],
    num_test_data: int,
    subgraph_ori_input_names: List[List[str]],
) -> List[List[List[Shape]]]:
    tf.reset_default_graph()
    graph = tf.Graph()
    res_tensors_list = [None] * num_test_data
    res_tensors = None
    with graph.as_default():
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.import_graph_def(graph_def, name="")
            i = 0
            for feed_dict in test_data:
                try:
                    res_tensors = sess.run(subgraph_ori_input_names, feed_dict)
                except Exception as err:
                    logging.info(
                        f"Session run in get_subgraph_test_inputs_shapes has err:{err}!"
                    )
                finally:
                    if res_tensors:
                        res_tensors_list[i] = res_tensors
                i = i + 1
    subgraph_test_inputs_shapes_list = [[[[0]]]] * num_test_data
    for idx_data, res_tensors in enumerate(res_tensors_list):
        subgraph_test_inputs_shapes = [[[0]]] * len(subgraph_ori_input_names)
        if res_tensors is not None:
            for idx_subgraph, input_names in enumerate(subgraph_ori_input_names):
                feed_shape = [[0]] * len(input_names)
                for idx_inp in range(len(input_names)):
                    feed_shape[idx_inp] = list(res_tensors[idx_subgraph][idx_inp].shape)
                subgraph_test_inputs_shapes[idx_subgraph] = feed_shape
            subgraph_test_inputs_shapes_list[idx_data] = subgraph_test_inputs_shapes

    return subgraph_test_inputs_shapes_list
