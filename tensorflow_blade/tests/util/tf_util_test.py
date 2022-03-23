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

import unittest

from tf_blade.util import tf_util
from tf_blade.util.tf_import_helper import tf


class UtilTest(unittest.TestCase):
    def test_to_metagraph(self) -> None:
        graph = tf.Graph()
        with graph.as_default():
            # Create a graph
            dtype = tf.float32
            shape = [2, 3]
            px = tf.placeholder(shape=shape, dtype=dtype, name="px")
            py = tf.placeholder(shape=shape, dtype=dtype, name="py")
            t1 = px + py
            result = tf.matmul(t1, tf.transpose(t1), name="result")  # noqa

            # convert to meta graph
            meta_graph = tf_util.graph_def_to_meta_graph(
                graph.as_graph_def(), ["px", "py"], ["result"]
            )
            signature = meta_graph.signature_def['serving_default']

            # check signature
            for key, input in signature.inputs.items():
                self.assertEqual(input.name, key + ":0")
                self.assertEqual(input.dtype, tf.float32.as_datatype_enum)
                self.assertEqual(len(input.tensor_shape.dim), 2)
                self.assertEqual(input.tensor_shape.dim[0].size, 2)
                self.assertEqual(input.tensor_shape.dim[1].size, 3)
            output = signature.outputs['result']
            self.assertEqual(output.name, "result:0")


class UtilityFunctionTest(unittest.TestCase):
    def test_get_canonical_tensor_name(self) -> None:
        self.assertEqual(tf_util.get_canonical_tensor_name("a"), "a:0")
        self.assertEqual(tf_util.get_canonical_tensor_name("a:1"), "a:1")
        self.assertEqual(tf_util.get_canonical_tensor_name("^a"), "^a")
        with self.assertRaises(Exception):
            tf_util.get_canonical_tensor_name("^a:1")
        with self.assertRaises(Exception):
            tf_util.get_canonical_tensor_name("a:1:9")
        with self.assertRaises(Exception):
            tf_util.get_canonical_tensor_name("a:a")


if __name__ == "__main__":
    unittest.main()
