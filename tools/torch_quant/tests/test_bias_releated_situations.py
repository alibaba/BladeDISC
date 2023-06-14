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

import unittest

import torch
from tests.models import SimpleLinearWithBias
from torch_quant.observer import BiasObserver
from torch_quant.quantizer import Backend, Quantizer


class TestBiasReleatedSituations(unittest.TestCase):
    def test_bias_exist(self):
        model = SimpleLinearWithBias().eval()
        dummy = torch.randn(1, 3)
        quantizer = Quantizer(backend=Backend.DISC)
        calib_model = quantizer.calib(model)
        calib_model(dummy)
        quantized_model = quantizer.quantize(model)
        self.assertTrue(isinstance(quantized_model.linear.bias_ob, BiasObserver))

        qat_model = quantizer.qat(model)
        self.assertTrue(qat_model.linear.bias_ob is None)

        quantized_model = quantizer.quantize(model)
        self.assertTrue(isinstance(quantized_model.linear.bias_ob, BiasObserver))


if __name__ == "__main__":
    unittest.main()
