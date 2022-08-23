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
import torch_blade
from torch_blade import utils


@unittest.skipIf(
    utils.torch_version_number() < utils.parse_version("1.8.0"),
    'Quant support since 1.8.0'
)
class TestTorchBladeFakeQuant(unittest.TestCase):
    def test_op_creation(self):
        qmin = -127
        qmax = 128
        num_bits = 8
        axis = []
        signed = True
        symatric = True
        dynamic = False
        per_channel = False
        op = torch.classes.torch_blade.FakeQuant((
            qmin, qmax, num_bits, axis, signed, symatric, dynamic, per_channel
        ))
        self.assertEqual(op.quant_min(), qmin)
        self.assertEqual(op.quant_max(), qmax)
        self.assertEqual(op.axis(), axis)
        self.assertEqual(op.num_bits(), num_bits)
        self.assertTrue(op.is_signed())
        self.assertTrue(op.is_symmetric())
        self.assertFalse(op.is_dynamic())
        self.assertFalse(op.is_per_channel())

    def test_invalid_op_creation(self):
        for axis, per_channel in [([], True), ([2], False)]:
            self.assertRaises(RuntimeError, 
                torch.classes.torch_blade.FakeQuant, (-127, 128, 8, axis, True, True, False, per_channel))

    def test_forward_per_tensor(self):
        qmin = -127
        qmax = 128
        num_bits = 8
        axis = []
        signed = True
        symatric = True
        dynamic = False
        per_channel = False
        op = torch.classes.torch_blade.FakeQuant((
            qmin, qmax, num_bits, axis, signed, symatric, dynamic, per_channel
        ))
        input = torch.rand(2, 3, 4)
        scale = torch.tensor([0.1])
        zero_point = torch.tensor([0], dtype=torch.int32)
        res1 = op.forward(input, scale, zero_point)
        res2 = torch.fake_quantize_per_tensor_affine(input, scale, zero_point, qmin, qmax)
        self.assertTrue(torch.allclose(res1, res2))

    def test_forward_per_channel(self):
        qmin = -127
        qmax = 128
        num_bits = 8
        axis = [1]
        signed = True
        symatric = True
        dynamic = False
        per_channel = True 
        op = torch.classes.torch_blade.FakeQuant((
            qmin, qmax, num_bits, axis, signed, symatric, dynamic, per_channel
        ))
        input = torch.rand(2, 3, 4, 4)
        scale = torch.tensor([0.1, 0.2, 0.3])
        zero_point = torch.tensor([0, 0, 0], dtype=torch.int32)
        res1 = op.forward(input, scale, zero_point)
        res2 = torch.fake_quantize_per_channel_affine(input, scale, zero_point, axis[0], qmin, qmax)
        self.assertTrue(torch.allclose(res1, res2))


if __name__ == "__main__":
    unittest.main()
