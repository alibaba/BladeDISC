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

import argparse
from pickletools import optimize
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.saved_model import loader, loader_impl
from tensorflow.python.framework import dtypes
import logging
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Iterator
import yaml
import os
from benchmark import optimize_methods


class Shape:
    def __init__(self, shape: List[Optional[int]], is_scalar: bool = False) -> None:
        self._shape_list = shape
        self.is_scalar = is_scalar

    @property
    def rank(self) -> int:
        return len(self._shape_list)

    @property
    def is_static(self) -> bool:
        return -1 not in self._shape_list

    @property
    def dims(self) -> Iterator[int]:
        return iter(self._shape_list)

    def dim(self, i: int) -> int:
        return self._shape_list[i]

    def aslist(self) -> List[int]:
        return self._shape_list

    def astuple(self) -> Tuple:
        return tuple(self._shape_list)


class TFModel:
    def __init__(
        self,
        _model_dir: Optional[str] = None,
        _model_info: Optional[dict] = None,
        # chosen_signature: str = "",
        # test_data: List[Dict[str, np.ndarray]] = [],
    ) -> None:
        """Create TfModel. Maybe an empty one or load from disk directory.

        Parameters
        ----------
        model_dir : Optional[str], optional
            Path to frozen pb file or saved_model.pb/saved_model.pbtxt file under saved
            model directory. Note that it's not saved model directory as you may think.
            By default None.
        chosen_signature : str, optional
            Specify the signature to optimize if it's a saved model with multiple
            signature, by default "".
        """
        # input nodes
        self.inputs: List[str] = list()
        # output nodes
        self.outputs: List[str] = list()
        # names of fetch tensors
        self.fetch_list: List[str] = list()
        # dtype of input tensors
        self.input_to_dtype: Dict[str, type] = dict()
        # shape of input tensors
        self.input_to_shape: Dict[str, Shape] = dict()
        # saved model related
        self.model_dir: Optional[str] = _model_dir
        # model related info like shape hint
        self.model_info = _model_info

        if self.model_dir is not None:
            self.load_and_init(self.model_dir)

    def get_default_func(self, model_dir: str, shape_hint: dict = {}) -> Any:
        tag_sets = saved_model_utils.get_saved_model_tag_sets(model_dir)
        if len(tag_sets) > 1:
            logging.warning("more than one tagset, choose the first one by default!")
        tag_set = tag_sets[0]
        tag_set = ",".join(tag_set)

        meta_graph_def = saved_model_utils.get_meta_graph_def(model_dir, tag_set)
        signature_def_map = meta_graph_def.signature_def
        keys_with_inputs = [
            key for key, sdef in signature_def_map.items() if len(sdef.inputs) > 0
        ]
        assert len(keys_with_inputs) > 0, "no valid signatures found!"
        default_key = (
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        )
        key = default_key if default_key in keys_with_inputs else keys_with_inputs[0]

        for model_input in signature_def_map[key].inputs.values():
            tensor_name = model_input.name.split(":")[0]
            shape = [int(dim.size) for dim in model_input.tensor_shape.dim]
            self.inputs.append(tensor_name)
            if shape_hint.get(tensor_name, None):
                self.input_to_shape[tensor_name] = Shape(shape_hint[tensor_name])
            else:
                self.input_to_shape[tensor_name] = Shape(
                    [int(dim.size) for dim in model_input.tensor_shape.dim]
                )
            self.input_to_dtype[tensor_name] = dtypes.as_dtype(
                model_input.dtype
            ).as_numpy_dtype
            # print(tensor_name, shape, self.input_to_shape[tensor_name].aslist(), self.input_to_dtype[tensor_name])
        for model_output in signature_def_map[key].outputs.values():
            self.fetch_list.append(model_output.name.split(":")[0])
        loaded = tf2.saved_model.load(model_dir, tags=tag_set)
        func = loaded.signatures[key]
        func = convert_variables_to_constants_v2(func)
        return func

    # the main function to load the saved model
    def load_and_init(self, model_dir: str) -> None:
        self._saved_model_dir = model_dir
        func = self.get_default_func(model_dir, self.model_info.get("shape", {}))

        self.graph_def = func.graph.as_graph_def()

    def gen_ndarray(self, input_name: str, shape: Shape, np_dtype) -> np.ndarray:
        np.random.seed(5)
        if np.issubdtype(np_dtype, np.integer):
            if shape.is_scalar:
                return np.random.randint(low=0, high=4, dtype=np_dtype)
            else:
                return np.random.randint(
                    low=0, high=4, size=shape.astuple(), dtype=np_dtype
                )
        elif np.issubdtype(np_dtype, np.floating):
            return np.random.random(shape.astuple()).astype(np_dtype)
        elif np_dtype == np.bool:
            return np.random.randint(
                low=0, high=2, size=shape.astuple(), dtype=np_dtype
            ).astype(np.bool)
        else:
            logging.warning(
                "Failed to generate ndarray for input {}, shape {}, dtype: {}".format(
                    input_name, shape, np_dtype
                )
            )

    # TODO[FIX]: there is a bug about the input tensor name, whether end with ":0"
    def gen_input_data(self) -> Optional[Dict[str, np.ndarray]]:
        input_to_shape = self.input_to_shape
        if len(input_to_shape) == 0:
            return None
        feed_dict = dict()
        # print(self.input_to_shape) with :0
        # print(self.inputs) # without :0
        for input_node in self.inputs:
            tensor_name = f"{input_node}:0"
            shape = self.input_to_shape[input_node]
            dtype = self.input_to_dtype[input_node]
            feed_dict[tensor_name] = self.gen_ndarray(input_node, shape, dtype)
        return feed_dict

    def get_default_sess_config(self):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        return config

    def inference(self, optimize_method: str) -> Optional[List]:
        config = self.get_default_sess_config()
        if optimize_method is None:
            pass
        elif optimize_method == "disc":
            import blade_disc_tf as disc

            disc.enable()
        elif optimize_method == "xla":
            # TODO: use xla
            value = tf.OptimizerOptions.ON_1
            config.graph_options.optimizer_options.global_jit_level = value
        elif optimize_method == "tvm":
            # TODO: refer to "https://tvm.apache.org/docs/how_to/compile_models/from_tensorflow.html#sphx-glr-how-to-compile-models-from-tensorflow-py"
            pass
        elif optimize_method == "tf_trt":
            # TODO: use tf-trt
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
        elif optimize_method == "tf_blade":
            # TODO: use tf-blade
            from tf_blade.gpu.tf_to_trt import Tf2TrtOpt

        # Prepare model and its feed & fetch
        saved_model = loader_impl._parse_saved_model(self.model_dir)
        # sess = tf.Session(config=config)
        tags = saved_model.meta_graphs[0].meta_info_def.tags
        # _ = tf.saved_model.loader.load(sess, tags, self.model_dir)

        with tf.Session(graph=tf.Graph(), config=config) as sess:
            tf.saved_model.loader.load(sess, tags, self.model_dir)
            fetch_list = self.fetch_list
            feed_dict = self.gen_input_data()

            # [Get runtime] #
            def get_runtime():
                tic = time.time()
                _ = sess.run(fetch_list, feed_dict=feed_dict)
                return time.time() - tic

            # Warm up
            warmup_num = 30
            run_num = 100
            [get_runtime() for i in range(warmup_num)]
            # Inference

            runtimes = [get_runtime() for i in range(run_num)]

        rt_str = " ".join(["{:.2f}".format(rt * 1000) for rt in runtimes])
        # print("Runtimes(ms): {}".format(rt_str))
        runtime = round(np.median(runtimes), 4)
        # print("Runtime(ms): {:.2f}".format(runtime * 1000))
        return runtime

    def amp(self):
        # TODO: turn the amp mode on
        pass


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-c", help="model config json file")
    parser.add_argument("--model", "-m", help="benchmark model name")
    parser.add_argument("--cache-path", "-p", help="benchmark model cache path")
    parser.add_argument("--result-file", "-r", help="result file path")
    parser.add_argument("--amp", "-a", help="enable amp")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, "r") as f:
        model_info = yaml.safe_load(f)[args.model]
    model_path = os.path.join(args.cache_path, args.model)
    model = TFModel(model_path, model_info)
    if args.amp:
        model.amp()

    results_str = args.model
    tf_runtime = model.inference(None)
    results_str += f",{tf_runtime}"
    for optimize_method in optimize_methods:
        runtime = model.inference(optimize_method)
        speedup = round(tf_runtime / runtime, 3)
        results_str += f",{runtime},{speedup}"
    results_str += "\n"
    print("*" * 100)
    print(results_str)
    with open(args.result_file, "a") as f:
        f.writelines(results_str)
