# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch_blade.mlir.cache import DiscCompilationCache, CompilationResult
from torch_blade.mlir.hash import get_graph_hash
from tests.disc.testing_base import DiscTestCase
import unittest
import torch_blade

class TestCache(DiscTestCase):
    def test_disc_cache(self):
        gstr = """graph(%x.1 : Float(*, *),
                %y.1 : Float(*, *)):
            %4 : Tensor = aten::mul(%x.1, %y.1)
            return (%4)"""
        graph = torch._C.parse_ir(gstr)

        so_bytes = "so_bytes"
        pb_bytes = "pb_bytes"
        input_dev_str = "cpu"
        output_dev_str = "cpu"

        cache = DiscCompilationCache()

        result = CompilationResult(so_bytes, pb_bytes, input_dev_str, output_dev_str)
        hash_value = get_graph_hash(graph)
        cache.set(hash_value, result)

        expect_result = cache.get(hash_value)
        self.assertEqual(expect_result._so_bytes, bytes(so_bytes, "utf-8"))
        self.assertEqual(expect_result._pb_bytes, bytes(pb_bytes, "utf-8"))
        self.assertEqual(expect_result._inputs_device, bytes(input_dev_str, "utf-8"))
        self.assertEqual(expect_result._outputs_device, bytes(output_dev_str, "utf-8"))



if __name__ == "__main__":
    unittest.main()