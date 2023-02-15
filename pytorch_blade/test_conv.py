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

import torch
import torch.nn.functional as F
import torch_blade
import unittest

from tests.disc.testing_base import DiscTestCase, skipTorchLE

@skipTorchLE("1.6.1")
class TestMlirConvolution(DiscTestCase):
    def _test_conv(self, conv_func, inp_test_data=None):
        if inp_test_data is not None:
            test_data = inp_test_data
        else:
            test_data = torch.randn([20, 16, 50, 100], device=self.device)
        if (isinstance(test_data, torch.Tensor)):
            test_data = (test_data.to(self.device),)
        self._test_cvt_to_disc(conv_func, test_data)

    def test_convolution_backward(self):
        class ConvBackward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.G = 1
                self.N = 1
                self.Cin = 2
                self.Cout = 512
                self.padding=[0,0]
                self.stride=[1,1]
                self.dilation=[1,1]
                self.W=26
                self.H=12
                self.kW=3
                self.kH=3
                self.Hout = (self.H + self.padding[0] * 2 - self.dilation[0] * (self.kH -1) - 1) // self.stride[0] + 1
                self.Wout = (self.W + self.padding[1] * 2 - self.dilation[1] * (self.kW -1) - 1) // self.stride[1] + 1

                self.inputs = torch.randn(self.N, self.G * self.Cin, self.H, self.W, requires_grad=True)
                self.weights = torch.randn(self.Cout * self.G, self.Cin, self.kH, self.kW, requires_grad=True)

                val = F.conv2d(self.inputs, self.weights, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.G)
                gt = torch.ones_like(val)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(val, gt)
                grads = torch.autograd.grad(outputs=loss, inputs=(val, self.inputs, self.weights))
                self.grad_output = grads[0]

            def forward(self, inputs, weights, grad_output):
                return torch.ops.aten.convolution_backward(
                    grad_output,
                    inputs,
                    weights,
                    bias_sizes=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    transposed=False,
                    output_padding=[0],
                    groups=self.G,
                    output_mask=[True, True, True])


        conv_backprop = ConvBackward()
        script_backprop = torch.jit.script(conv_backprop)

        examples = (conv_backprop.inputs, conv_backprop.weights, conv_backprop.grad_output)
        grads0 = conv_backprop(*examples)
        grads1 = script_backprop(*examples)
        self.assertEqual(grads0, grads1)

        self._test_conv(script_backprop, examples)

if __name__ == "__main__":
    unittest.main()
