# Copyright 2021 The BladeDISC Authors. All rights reserved.
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

import torch
from tests.quantization import QuantizationTestCase
from torch import nn
from torch.testing import FileCheck
from torch_blade import tensorrt
from torch_blade.config import Config
from torch_blade.clustering.support_fusion_group import supported_node_fusion
from torch_blade.pass_manager import _optimize_common
from torch_blade.quantization.prepare_data import DataCollectObserver, DataPreparer
from torch_blade.tensorrt import is_available as is_tensorrt_available


def prepare_for_data_collect(model):
    cfg = Config.get_current_context_or_new()
    cfg.optimization_pipeline = tensorrt.backend_name()
    with cfg:
        optimized_c_module = _optimize_common(model._c)
        model._reconstruct(optimized_c_module)
        graph = model._c.forward.graph
        unsupported = tensorrt.get_unsupported_nodes(graph)
        supported_node_fusion(graph, graph, unsupported)
    return model


class TestDataCollectorObserver(QuantizationTestCase):
    def test_data_collector_observer(self):
        observer = DataCollectObserver()
        script_observer = torch.jit.script(observer)
        origin_data = [
            torch.randn(1, 2).to(self.device), torch.randn(3).to(self.device)
        ]
        new_data = script_observer.collect(origin_data)
        self.assertEqual(len(origin_data), len(new_data))
        for o, n in zip(origin_data, new_data):
            self.assertTrue(torch.equal(o, n))
        record_data_list = script_observer.data
        self.assertEqual(len(record_data_list), 2)
        record_data = record_data_list[1]
        for o, r in zip(origin_data, record_data):
            self.assertTrue(torch.equal(o.cpu(), r))


class TestDataPreparer(QuantizationTestCase):
    @unittest.skipIf(not is_tensorrt_available(), "TensorRT is not available")
    def test_naive_model(self):
        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = nn.Conv2d(1, 2, (3, 3), padding=1, bias=False)
                self.conv2 = nn.Conv2d(2, 4, (3, 3), padding=1, bias=False)
                self.conv3 = nn.Conv2d(4, 1, (3, 3), padding=1, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                return x

        dummy = torch.randn(1, 1, 5, 5).to(self.device)
        model = MyModel().eval().to(self.device)
        traced_model = torch.jit.trace(model, dummy)
        origin_output = traced_model(dummy)
        prepared_model = prepare_for_data_collect(traced_model)

        expect_graph_str = """
            graph(%self.1 : MyModel, %input.1 : Tensor:
                # CHECK: prim::FusionGroup_0
                %1 : Tensor = prim::FusionGroup_0(%input.1)
                return (%1)
        """
        c_module = prepared_model._c
        graph = c_module.forward.graph
        FileCheck().run(expect_graph_str, graph)

        data_preparer = DataPreparer(c_module, [(dummy,)])

        data_preparer.make_fusion_group_executable()

        # check the origin graph is not changed
        FileCheck().run(expect_graph_str, graph)

        now_graph = data_preparer.graph
        expect_graph_str = """
            graph(%self.1 : MyModel, 
                  %input.1 : Tensor，
                  %3 : ScriptModule:
                # CHECK-NOT: prim::FusionGroup_0
                # CHECK: prim::GetAttr
                %4 : libtorch_executable_fusion_group_0 = prim::GetAttr[name="libtorch_executable_fusion_group_0"](%3)
                # CHECK: prim::CallMethod
                %5 : Tensor[] = prim::CallMethod[name="forward"](%4, %input.1)
                # CHECK: prim::ListUnpack
                %7 : Tensor = prim::ListUnpack(%5)
                return (%7)
        """
        FileCheck().run(expect_graph_str, now_graph)
        c_module.create_method_from_graph("inference", now_graph)
        now_output = c_module.inference(dummy, data_preparer.custom_module_owner)
        self.assertTrue(torch.equal(origin_output, now_output))
        c_module.unsafe_remove_method("inference")

        data_preparer.insert_observer()
        now_graph = data_preparer.graph
        expect_graph_str = """
            graph(%self.1 : MyModel, 
                  %input.1 : Tensor，
                  %3 : ScriptModule:
                # CHECK: prim::GetAttr
                %4 : libtorch_executable_fusion_group_0 = prim::GetAttr[name="libtorch_executable_fusion_group_0"](%3)
                # CHECK: prim::ListConstruct
                %8 : Tensor[] = prim::ListConstruct(%input.1)
                # CHECK: prim::GetAttr
                %9 : DataCollectObserver = prim::GetAttr[name="collect_observer_0"](%3)
                # CHECK: prim::CallMethod
                %10 : Tensor[] = prim::CallMethod[name="forward"](%9, %8)
                # CHECK: prim::ListUnpack
                %12 : Tensor = prim::ListUnpack(%10)
                # CHECK: prim::CallMethod
                %5 : Tensor[] = prim::CallMethod[name="forward"](%4, %12)
                # CHECK: prim::ListUnpack
                %7 : Tensor = prim::ListUnpack(%5)
                return (%7)
        """
        FileCheck().run(expect_graph_str, now_graph)

        all_data = data_preparer.get_calib_data_for_each_group()
        record_data = all_data[0][0][0]
        self.assertTrue(torch.equal(dummy.cpu(), record_data))


if __name__ == "__main__":
    unittest.main()
