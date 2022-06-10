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

import pytest

from tests.tf_test_common import ConvNetMaker
from tf_blade.gpu.tf_to_trt import Tf2TrtOpt
from tf_blade.util.tf_conversion_util import TRT_SUPPORTED_LIST
from tf_blade.util.tf_import_helper import tf

tf.disable_v2_behavior()


@pytest.mark.gpu_only
class Tf2TrtPlusTest(unittest.TestCase):
    def setUp(self) -> None:
        conv_net_maker = ConvNetMaker()
        self.static_graph_def = conv_net_maker.gen_simple_conv_net(32)
        self.dynamic_graph_def = conv_net_maker.gen_simple_conv_net(-1)
        self.opt_pass_static = Tf2TrtOpt(TRT_SUPPORTED_LIST, 2, False, "")
        self.opt_pass_dynamic = Tf2TrtOpt(TRT_SUPPORTED_LIST, 2, True, "")

    def test_dynamic_shape_fp16(self) -> None:
        # set test data
        feed_dict_b8 = ConvNetMaker().get_simple_conv_net_feed_dict(8)
        feed_dict_b16 = ConvNetMaker().get_simple_conv_net_feed_dict(16)
        feed_dict_b32 = ConvNetMaker().get_simple_conv_net_feed_dict(32)
        ret_graph_def = self.opt_pass_dynamic.optimize_graph_def(
            self.dynamic_graph_def,
            ["out"],
            [feed_dict_b8, feed_dict_b32, feed_dict_b8, feed_dict_b16, feed_dict_b32],
            False,
        )
        func_names = list()
        count = 0
        for node in ret_graph_def.node:
            if node.op == Tf2TrtOpt.OPT_CUSTOM_OP_TYPE:
                func_names.append(node.name)
                count += 1
        self.assertGreater(count, 0)
        for func in ret_graph_def.library.function:
            if func.signature.name in func_names:
                self.assertTrue(len(func.signature.attr) == 0)

    def test_static_shape_fp32(self) -> None:
        feed_dict = ConvNetMaker().get_simple_conv_net_feed_dict(32)
        ret_graph_def = self.opt_pass_static.optimize_graph_def(
            self.static_graph_def, ["out"], [feed_dict], True
        )
        func_names = list()
        count = 0
        for node in ret_graph_def.node:
            if node.op == Tf2TrtOpt.OPT_CUSTOM_OP_TYPE:
                func_names.append(node.name)
                count += 1
        self.assertGreater(count, 0)
        for func in ret_graph_def.library.function:
            if func.signature.name in func_names:
                self.assertTrue(len(func.signature.attr) == 0)

    def test_static_shape_fp16(self) -> None:
        feed_dict = ConvNetMaker().get_simple_conv_net_feed_dict(32)
        ret_graph_def = self.opt_pass_static.optimize_graph_def(
            self.static_graph_def, ["out"], [feed_dict], False
        )
        func_names = list()
        count = 0
        for node in ret_graph_def.node:
            if node.op == Tf2TrtOpt.OPT_CUSTOM_OP_TYPE:
                func_names.append(node.name)
                count += 1
        self.assertGreater(count, 0)
        for func in ret_graph_def.library.function:
            if func.signature.name in func_names:
                self.assertTrue(len(func.signature.attr) == 0)


if __name__ == "__main__":
    unittest.main()
