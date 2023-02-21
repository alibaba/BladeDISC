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

import torch
from torch_quant.observer import LSQObserver


class TestLSQObserver(unittest.TestCase):
    def test_init_params(self):
        ob = LSQObserver()
        self.assertTrue(torch.equal(ob.scale, torch.tensor([1.])))
        self.assertTrue(torch.equal(ob.zero_point, torch.tensor([0.])))

        init_scale = torch.tensor([0.1])
        init_zero_point = torch.tensor([100.1])
        ob = LSQObserver(init_scale=init_scale, init_zp=init_zero_point)
        self.assertTrue(torch.equal(ob.scale, init_scale))
        self.assertTrue(torch.equal(ob.zero_point, init_zero_point))

    def test_fake_quant_output(self):
        # per-tensor + affine
        init_scale = torch.tensor([0.1])
        init_zero_point = torch.tensor([100.1])
        ob = LSQObserver(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine,
            init_scale=init_scale, init_zp=init_zero_point)
        ob.fake_quant = True
        inp = torch.randn(100)
        output = ob(inp)
        expected_output = torch.fake_quantize_per_tensor_affine(inp, init_scale, init_zero_point, 0, 255)
        self.assertTrue(torch.equal(output, expected_output))

        # per-channel + symmetry
        inp = torch.randn(10, 20)
        init_scale = torch.randn(10)
        init_zero_point = torch.zeros(10, dtype=torch.int32)
        ob = LSQObserver(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
            init_scale=init_scale, init_zp=init_zero_point
        )
        output = ob(inp)
        expected_output = torch.fake_quantize_per_channel_affine(inp, init_scale, init_zero_point, 0, -128, 127)
        self.assertTrue(torch.equal(output, expected_output))

        ob.fake_quant = False
        output = ob(inp)
        self.assertTrue(torch.equal(output, inp))

    def test_from_qparams(self):
        init_scale = torch.randn(10)
        init_zero_point = torch.zeros(10, dtype=torch.int32)
        ob = LSQObserver(
            dtype=torch.qint8, qscheme=torch.per_channel_affine,
            init_scale=init_scale, init_zp=init_zero_point
        )
        ob = LSQObserver.from_qparams(ob.qparams)
        self.assertTrue(torch.equal(ob.scale, init_scale))
        self.assertTrue(torch.equal(ob.zero_point, init_zero_point.float()))


if __name__ == "__main__":
    unittest.main()
