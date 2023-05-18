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
import tempfile
from torch_blade import mlir, optimize, Config

from tests.disc.testing_base import DiscTestCase

class TestDiscEngine(DiscTestCase):
    def test_no_output_overwrite(self):
        class Triple(torch.nn.Module):
            def forward(self, x):
                return 3.0 * x + 0.0 + 0.0

        x = torch.randn(1, device=self.device)
        triple = self.cvt_to_disc(Triple().eval(), x)

        self.assertEqual(mlir.num_engines(triple), 1)
        self.assertEqual(len(mlir.num_compiled_nodes(triple)) , mlir.num_engines(triple))
        
        one = torch.tensor([1], dtype=torch.float, device=self.device)
        two = torch.tensor([2], dtype=torch.float, device=self.device)

        three = triple(one)
        self.assertEqual(three, 3 * one)

        six = triple(two)
        self.assertEqual(six, 3 * two)

        self.assertEqual(three, 3 * one)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(10, 10))
        self.a = torch.nn.ReLU()
        self.b = torch.nn.ReLU()
        self.c = torch.nn.ReLU()
        self.d = torch.nn.ReLU()
        self.e = torch.nn.ReLU()

    def forward(self, x):
        x = torch.matmul(x, self.w)
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        return x


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, 
    "Multi GPUs not available"
)
class TestDiscEngineDevice(unittest.TestCase):
    def setUp(self):
        self.old_device = torch.cuda.current_device()
        self.input = torch.randn(10).cuda()
        model = Dummy().eval().cuda()
        self.ref_output = model(self.input)
        traced = torch.jit.trace(model, self.input)
        with Config():
            pt_model = optimize(traced, model_inputs=(self.input))
        self.model_f = tempfile.NamedTemporaryFile()
        torch.jit.save(pt_model, self.model_f)
        self.model_f.seek(0)

    def tearDown(self):
        self.model_f.close()
        torch.cuda.set_device(self.old_device)

    def test_change_device_before_infer(self):
        # Load in one device
        torch.cuda.set_device(0)
        model = torch.jit.load(self.model_f)
        # Infer in another device
        torch.cuda.set_device(1)
        output = model(self.input.cuda(1))
        self.assertTrue(torch.allclose(output.cpu(), self.ref_output.cpu()))
        # Infer again
        output = model(self.input.cuda(1))
        self.assertTrue(torch.allclose(output.cpu(), self.ref_output.cpu()))

    def test_change_device_during_infer(self):
        # Load in one device
        torch.cuda.set_device(0)
        model = torch.jit.load(self.model_f)
        # Infer once
        _ = model(self.input.cuda(0))
        torch.cuda.set_device(1)
        # Infer in another device
        with self.assertRaisesRegex(RuntimeError,
                                    "Device changed during inference\. .*"):
            _ = model(self.input.cuda(1))


if __name__ == "__main__":
    unittest.main()
