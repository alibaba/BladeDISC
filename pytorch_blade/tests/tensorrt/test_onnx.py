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
import io
import onnx
import torch
import torch.nn.functional as F
import torch_blade.pass_manager as pass_manager

from torch import nn
from torch.testing import FileCheck
from torch_blade import exporter
from torch_blade import utils
from torch_blade.testing.common_utils import TestCase
from tests.tensorrt import skipIfNoTensorRT

@skipIfNoTensorRT()
class TestOnnx(TestCase):
    def test_convolution(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                return x

        script_module = torch.jit.script(Net()).eval()
        inputs = torch.ones([1, 3, 20, 20])
        script_module = exporter.export(script_module, model_inputs=(inputs,))
        script_module = pass_manager._optimize_common(script_module._c)
        onnx_graph, _ = pass_manager._jit_pass_lower_to_onnx(script_module.forward.graph)
        expect_gstr = """
graph(%self.1 : __torch__.___torch_mangle_0.Net,
      %x.1 : Float(1, 3, 20, 20, strides=[1200, 400, 20, 1], requires_grad=0, device=cpu)):
  %7 : Float(16, strides=[1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>]()
  %8 : Float(16, 6, 5, 5, strides=[150, 25, 5, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>]()
  %9 : Float(6, strides=[1], requires_grad=0, device=cpu) = onnx::Constant[value=-0.0429  0.0213  0.0239  0.1149 -0.0042 -0.0935 [ CPUFloatType{6} ]]()
  %10 : Float(6, 3, 5, 5, strides=[75, 25, 5, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>]()
  # CHECK-COUNT-EXACTLY-1: onnx::Conv
  %11 : Float(1, 6, 16, 16, strides=[1536, 256, 16, 1], requires_grad=0, device=cpu) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[0, 0, 0, 0], strides=[1, 1]](%x.1, %10, %9)
  # CHECK-COUNT-EXACTLY-1: onnx::Relu
  %12 : Float(1, 6, 16, 16, strides=[1536, 256, 16, 1], requires_grad=0, device=cpu) = onnx::Relu(%11)
  # CHECK-COUNT-EXACTLY-1: onnx::MaxPool
  %13 : Float(1, 6, 8, 8, strides=[384, 64, 8, 1], requires_grad=0, device=cpu) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2]](%12)
  # CHECK-COUNT-EXACTLY-1: onnx::Conv
  %14 : Float(1, 16, 4, 4, strides=[256, 16, 4, 1], requires_grad=0, device=cpu) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[0, 0, 0, 0], strides=[1, 1]](%13, %8, %7)
  # CHECK-COUNT-EXACTLY-1: onnx::Relu
  %15 : Float(1, 16, 4, 4, strides=[256, 16, 4, 1], requires_grad=0, device=cpu) = onnx::Relu(%14)
  # CHECK-COUNT-EXACTLY-1: onnx::MaxPool
  %16 : Float(1, 16, 2, 2, strides=[64, 4, 2, 1], requires_grad=0, device=cpu) = onnx::MaxPool[kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2]](%15)
  return (%16)"""
        FileCheck().run(expect_gstr, onnx_graph)
        onnx_proto = pass_manager._export_onnx(onnx_graph, dict())
        with io.BytesIO() as f:
            onnx.save(onnx_proto, f)

    @unittest.skipIf(utils.torch_version_number() <= utils.parse_version("1.7.1"), 'failed')
    def test_fuse_conv_bn(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(3, 6, 5)
                self.bn = nn.BatchNorm2d(6)

            def forward(self, x):
                x = F.relu(self.bn(self.conv(x)))
                return x

        script_module = torch.jit.script(Net()).eval()
        inputs = torch.ones([1, 3, 20, 20])
        script_module = exporter.export(script_module, model_inputs=(inputs,))
        script_module = pass_manager._optimize_common(script_module._c)
        onnx_graph, _ = pass_manager._jit_pass_lower_to_onnx(script_module.forward.graph)
        expect_gstr = """
graph(%self.1 : __torch__.tests.tensorrt.test_onnx.___torch_mangle_2.Net,
  %x.1 : Float(1, 3, 20, 20, strides=[1200, 400, 20, 1], requires_grad=0, device=cpu)):
  %self.conv.weight_fused_bn : Float(6, 3, 5, 5, strides=[75, 25, 5, 1], requires_grad=0, device=cpu) = onnx::Constant[value=<Tensor>]()
  %self.conv.bias_fused_bn : Float(6, strides=[1], requires_grad=0, device=cpu) = onnx::Constant[value=0.01 *  3.2080 -10.6066 -0.7376 -1.8886 -7.9712  8.1410 [ CPUFloatType{6} ]]()
  # CHECK-COUNT-EXACTLY-1: onnx::Conv
  # CHECK-NOT: onnx::BatchNormalization
  %7 : Float(1, 6, 16, 16, strides=[1536, 256, 16, 1], requires_grad=0, device=cpu) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[0, 0, 0, 0], strides=[1, 1]](%x.1, %self.conv.weight_fused_bn, %self.conv.bias_fused_bn) # /usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py:453:15
  %result : Float(1, 6, 16, 16, strides=[1536, 256, 16, 1], requires_grad=0, device=cpu) = onnx::Relu(%7) # /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1457:17
  return (%result)
        """
        FileCheck().run(expect_gstr, onnx_graph)
        onnx_proto = pass_manager._export_onnx(onnx_graph, dict())
        with io.BytesIO() as f:
            onnx.save(onnx_proto, f)
if __name__ == '__main__':
    unittest.main()
