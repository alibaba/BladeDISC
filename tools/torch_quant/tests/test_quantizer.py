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
from typing import List, Optional

import torch
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.quantized._reference as nnqr
from parameterized import parameterized
from tests.models import SimpleModule, SubModule, UntraceableSimpleModule
from torch_quant.amp_module import AmpModule
from torch_quant.module import ModuleFilter
from torch_quant.quantizer import (
    DEFAULT_ACT_OB_CTR,
    DEFAULT_QAT_OB_CTR,
    DEFAULT_W_OB_CTR,
    Backend,
    Device,
    Quantizer
)


def parameterized_with_backends(parameters: Optional[List] = None):
    if parameters is None:
        parameters = [
            (Backend.REFERENCE,),
            (Backend.FBGEMM,),
            (Backend.DISC,),
            (Backend.MNN,),
        ]
    # skip if fbgemm not available
    if torch.backends.quantized.engine != 'fbgemm':
        parameters = [param for param in parameters if Backend.FBGEMM not in param]
    return parameterized.expand(parameters)


class QuantizerTest(unittest.TestCase):
    @parameterized_with_backends()
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
        torch.testing.assert_close(quant_output, original_output, rtol=0.1, atol=0.5)

        if backend == Backend.MNN:
            _, compress_proto = quantizer.export_mnn_params(quant_model, dummy_input)
            self.assertEqual(len(compress_proto.algo[0].quant_params.layer), 3)

    # TODO(litan.ls): QAT is more suitable for this case
    @parameterized_with_backends()
    def test_load_from_state_dict(self, backend: Backend) -> None:
        model = SimpleModule()
        quantizer = Quantizer(backend=backend)
        dummy_input = torch.randn((1, 2, 5, 5))

        quantizer.calib(model)(dummy_input)
        with tempfile.NamedTemporaryFile() as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)
            loaded_model = SimpleModule()
            quantizer.calib(loaded_model)
            loaded_model.load_state_dict(torch.load(tmp_file.name))
            loaded_quant_output = quantizer.quantize(loaded_model)(dummy_input)
        quant_output = quantizer.quantize(model)(dummy_input)
        self.assertTrue(torch.equal(loaded_quant_output, quant_output))

    @parameterized_with_backends()
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
            loaded = torch.jit.load(tmp_file.name)
        loaded_output = loaded(dummy_input)
        torch.testing.assert_close(quant_output, loaded_output)

    @parameterized_with_backends()
    def test_calib_quantize_qat_quantize_state_equal(self, backend):
        dummy_input = torch.randn((1, 2, 5, 5))
        model = SimpleModule()
        quantizer = Quantizer(backend=backend)
        calib_model = quantizer.calib(model)
        calib_model(dummy_input)
        fake_quant_model1 = quantizer.quantize(model)
        out1 = fake_quant_model1(dummy_input)

        qat_model = quantizer.qat(model)
        out2 = qat_model(dummy_input)
        # TODO(wanchen.swc): skip only for x86 cpu without vnni or reduce range
        if backend != Backend.FBGEMM:
            self.assertTrue(torch.equal(out1, out2))

        quant_model = quantizer.quantize(model)
        out3 = quant_model(dummy_input)
        self.assertTrue(torch.equal(out1, out3))

        optimizer = torch.optim.SGD(qat_model.parameters(), lr=0.1)
        qat_model(dummy_input).sum().backward()
        optimizer.step()
        out4 = qat_model(dummy_input)
        self.assertFalse(torch.equal(out2, out4))

        quant_model = quantizer.quantize(model)
        out5 = quant_model(dummy_input)
        if backend != Backend.FBGEMM:
            self.assertTrue(torch.equal(out4, out5))

    @parameterized_with_backends()
    def test_calib_amp_quantize(self, backend: Backend) -> None:
        model = SimpleModule()
        dummy_input = torch.randn((1, 2, 5, 5))
        quantizer = Quantizer(backend=backend)
        original_output = model(dummy_input)

        calib_model = quantizer.calib(model)
        calib_output = calib_model(dummy_input)
        self.assertTrue(torch.equal(original_output, calib_output))

        amp_model = quantizer.amp(model)
        amp_modules = dict(amp_model.named_modules())
        for name, mod in model.named_modules():
            if type(mod) in [torch.nn.Conv2d, torch.nn.Linear]:
                self.assertTrue(isinstance(amp_modules[name], AmpModule))
        amp_output = amp_model(dummy_input)
        self.assertTrue(torch.equal(original_output, amp_output))
        quantizer.fallback(amp_model, num=2)
        self.assertEqual(len(quantizer.module_filter.exclude_names), 2)

        quant_model = quantizer.quantize(model)
        torch.jit.trace(quant_model, dummy_input)
        modules = dict(model.named_modules())
        quant_modules = dict(quant_model.named_modules())
        for name in quantizer.module_filter.exclude_names:
            self.assertEqual(quant_modules[name], modules[name])
        quant_output = quant_model(dummy_input)
        self.assertFalse(torch.equal(original_output, quant_output))
        torch.testing.assert_close(quant_output, original_output, rtol=0.1, atol=0.5)

    @parameterized_with_backends(
        [
            (Backend.FBGEMM, ModuleFilter(include_op_types=[torch.nn.Linear])),
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

    @parameterized_with_backends(
        [
            (
                Backend.FBGEMM,
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

    @parameterized_with_backends([(Backend.REFERENCE,), (Backend.FBGEMM,)])
    def test_calib_and_quantize_with_module_fusion(self, backend):
        model = SimpleModule()
        quantizer = Quantizer(backend=backend)
        dummy_input = torch.randn((1, 2, 5, 5))
        calib_model = quantizer.calib(model)
        calib_model(dummy_input)
        quant_model = quantizer.quantize(model)
        self.assertTrue(isinstance(calib_model.sub.conv, nni.ConvReLU2d))
        if backend == Backend.REFERENCE:
            self.assertTrue(isinstance(quant_model.sub.conv[0], nnqr.Conv2d))
        elif backend == Backend.FBGEMM:
            self.assertTrue(isinstance(quant_model.sub.conv, nniq.ConvReLU2d))

    def _test_observer_type(self, t, target_t):
        self.assertEqual(type(t), type(target_t))
        self.assertEqual(t.dtype, target_t.dtype)
        self.assertEqual(t.qscheme, target_t.qscheme)

    @parameterized_with_backends([
        (Device.X86, Backend.DISC),
        (Device.X86, Backend.REFERENCE),
        (Device.X86, Backend.FBGEMM),
        (Device.AARCH64, Backend.DISC),
        (Device.GPU, Backend.DISC)
    ])
    def test_different_device_backend(self, device, backend):
        def check_each_observer_type(model, target_w_ob, target_act_ob):
            linear = model.linear
            conv = model.conv
            if hasattr(linear, "w_ob"):
                self._test_observer_type(linear.w_ob, target_w_ob)
            if hasattr(conv, "w_ob"):
                self._test_observer_type(conv.w_ob, target_w_ob)
            self._test_observer_type(model.x_ob, target_act_ob)
            self._test_observer_type(model.conv_ob, target_act_ob)
            self._test_observer_type(model.flatten_ob, target_act_ob)
            self._test_observer_type(model.linear_ob, target_act_ob)

        dummy_input = torch.randn((1, 2, 5, 5))
        model = SimpleModule()
        quantizer = Quantizer(backend=backend, device=device)
        calib_model = quantizer.calib(model)
        ptq_w_ob = DEFAULT_W_OB_CTR[device][backend]()
        ptq_act_ob = DEFAULT_ACT_OB_CTR[device][backend]()
        check_each_observer_type(calib_model, ptq_w_ob, ptq_act_ob)
        calib_model(dummy_input)
        qat_model = quantizer.qat(model)
        kwds = lambda ob: {k: getattr(ob, k) for k in ['qscheme', 'dtype']}
        qat_w_ob = DEFAULT_QAT_OB_CTR(**kwds(ptq_w_ob))
        qat_act_ob = DEFAULT_QAT_OB_CTR(**kwds(ptq_act_ob))
        check_each_observer_type(qat_model, qat_w_ob, qat_act_ob)


if __name__ == '__main__':
    unittest.main()
