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

import tempfile
import unittest
from typing import Optional

import torch
from parameterized import parameterized

from tests.models import SimpleModule, SubModule, UntraceableSimpleModule
from torch_quant.module import ModuleFilter
from torch_quant.observer import toggle_observer
from torch_quant.quantizer import Backend, Quantizer


class QuantizerTest(unittest.TestCase):
    @parameterized.expand([
        (Backend.REFERENCE, ),
        # (Backend.FBGEMM, ), TODO(litan.ls): select test according to ci env
        (Backend.DISC, ),
    ])
    def test_calib_and_quantize(self, backend: Backend) -> None:
        model = SimpleModule()
        quantizer = Quantizer(backend=backend)
        dummy_input = torch.randn((1, 2, 5, 5))
        original_output = model(dummy_input)

        calib_model = quantizer.calib(model)
        calib_output = calib_model(dummy_input)
        self.assertTrue(torch.equal(original_output, calib_output))

        quant_model = quantizer.quantize(model)
        quant_output = quant_model(dummy_input)
        self.assertFalse(torch.equal(original_output, quant_output))
        torch.testing.assert_close(
            quant_output, original_output, rtol=0.1, atol=0.5)

    # TODO(litan.ls): QAT is more suitable for this case
    @parameterized.expand([
        (Backend.REFERENCE, ),
        (Backend.DISC, ),
    ])
    def test_load_from_state_dict(self, backend: Backend) -> None:
        model = SimpleModule()
        quantizer = Quantizer(backend=backend)
        dummy_input = torch.randn((1, 2, 5, 5))

        quantizer.calib(model)(dummy_input)
        with tempfile.NamedTemporaryFile() as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)
            loaded_model = SimpleModule()
            quantizer.calib(loaded_model).load_state_dict(
                torch.load(tmp_file.name))
            loaded_quant_output = quantizer.quantize(loaded_model)(dummy_input)
        quant_output = quantizer.quantize(model)(dummy_input)
        self.assertTrue(torch.equal(loaded_quant_output, quant_output))

    @parameterized.expand([
        (Backend.REFERENCE, ),
        (Backend.DISC, ),
    ])
    def test_save_and_load_quantized(self, backend: Backend) -> None:
        model = SimpleModule()
        quantizer = Quantizer(backend=backend)
        dummy_input = torch.randn((1, 2, 5, 5))
        calib_model = quantizer.calib(model)
        calib_model(dummy_input)

        quant_model = quantizer.quantize(model)
        quant_output = quant_model(dummy_input)
        ts_quant_model = torch.jit.trace(quant_model, dummy_input)

        with tempfile.NamedTemporaryFile() as tmp_file:
            torch.jit.save(ts_quant_model, tmp_file.name)
            loaded = torch.jit.load(tmp_file.name)
        loaded_output = loaded(dummy_input)
        torch.testing.assert_close(quant_output, loaded_output)

    def test_calib_and_quantize_with_bias_observer(self):
        dummy_input = torch.randn((1, 2, 5, 5))
        model = SimpleModule()
        quantizer = Quantizer(backend=Backend.DISC)
        calib_model = quantizer.calib(model)
        calib_model(dummy_input)

        toggle_observer(calib_model, observe=False, fake_quant=True)
        out1 = calib_model(dummy_input)

        fake_quant_model = quantizer.quantize(model)
        out2 = fake_quant_model(dummy_input)
        self.assertTrue(torch.equal(out1, out2))

    def test_calib_quantize_qat_quantize_state_equal(self):
        dummy_input = torch.randn((1, 2, 5, 5))
        model = SimpleModule()
        quantizer = Quantizer(backend=Backend.DISC)
        calib_model = quantizer.calib(model)
        calib_model(dummy_input)
        fake_quant_model1 = quantizer.quantize(model)
        out1 = fake_quant_model1(dummy_input)

        qat_model = quantizer.qat(model)
        out2 = qat_model(dummy_input)
        self.assertTrue(torch.equal(out1, out2))

        quant_model = quantizer.quantize(model)
        out3 = quant_model(dummy_input)
        self.assertTrue(torch.equal(out2, out3))

    @parameterized.expand(
        [
            (Backend.REFERENCE, ModuleFilter(include_op_types=[torch.nn.Linear])),
            (Backend.DISC, ModuleFilter(exclude_op_types=[torch.nn.Conv2d])),
        ]
    )
    def test_calib_and_quantize_with_op_types_filter(
        self, backend: Backend, module_filter: ModuleFilter
    ) -> None:
        model = SimpleModule()
        quantizer = Quantizer(backend=backend, module_filter=module_filter)
        dummy_input = torch.randn((1, 2, 5, 5))
        original_output = model(dummy_input)

        calib_model = quantizer.calib(model)
        calib_output = calib_model(dummy_input)
        self.assertTrue(torch.equal(original_output, calib_output))

        quant_model = quantizer.quantize(model)
        self.assertTrue(isinstance(quant_model.conv, torch.nn.Conv2d))
        self.assertTrue(isinstance(quant_model.sub.conv, torch.nn.Conv2d))
        self.assertEqual(quant_model.conv, model.conv)
        self.assertEqual(quant_model.sub.conv, model.sub.conv)
        self.assertNotEqual(quant_model.linear, model.linear)

        quant_output = quant_model(dummy_input)
        self.assertFalse(torch.equal(original_output, quant_output))
        torch.testing.assert_close(quant_output, original_output, rtol=0.1, atol=0.5)

    @parameterized.expand(
        [
            (
                Backend.REFERENCE,
                ModuleFilter(
                    include_names=['traceable_sub'],
                    exclude_names=['traceable_sub.sub.conv'],
                    exclude_op_types=[torch.nn.Linear],
                ),
            ),
            (
                Backend.DISC,
                ModuleFilter(
                    include_classes=[SimpleModule],
                    exclude_classes=[SubModule],
                    include_op_types=[torch.nn.Conv2d],
                ),
            ),
        ]
    )
    def test_calib_and_quantize_with_module_filter(
        self, backend: Backend, module_filter: ModuleFilter
    ) -> None:
        model = UntraceableSimpleModule()
        quantizer = Quantizer(backend=backend, module_filter=module_filter)
        dummy_input = torch.randn((1, 2, 5, 5))
        original_output = model(dummy_input)

        calib_model = quantizer.calib(model)
        calib_output = calib_model(dummy_input)
        self.assertTrue(torch.equal(original_output, calib_output))

        qmodel = quantizer.quantize(model)
        self.assertTrue(isinstance(qmodel.traceable_sub.linear, torch.nn.Linear))
        self.assertTrue(isinstance(qmodel.traceable_sub.sub.conv, torch.nn.Conv2d))
        self.assertEqual(qmodel.traceable_sub.linear, model.traceable_sub.linear)
        self.assertEqual(qmodel.traceable_sub.sub.conv, model.traceable_sub.sub.conv)
        self.assertNotEqual(qmodel.traceable_sub.conv, model.traceable_sub.conv)

        quant_output = qmodel(dummy_input)
        torch.testing.assert_close(quant_output, original_output, rtol=0.1, atol=0.5)


if __name__ == '__main__':
    unittest.main()
