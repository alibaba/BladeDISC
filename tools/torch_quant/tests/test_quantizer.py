import logging
import tempfile
import unittest

import torch
from parameterized import parameterized
from tests.models import SimpleModule

from torch_quant.quantizer import Backend, Quantizer


class QuantizerTest(unittest.TestCase):
    @parameterized.expand([
        (Backend.REFERENCE, ),
        (Backend.FBGEMM, ),
    ])
    def test_calib_and_quantize(self, backend) -> None:
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
    def test_load_from_state_dict(self) -> None:
        model = SimpleModule()
        quantizer = Quantizer()
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

    def test_save_and_load_quantized(self) -> None:
        model = SimpleModule()
        quantizer = Quantizer()
        dummy_input = torch.randn((1, 2, 5, 5))

        quant_model = quantizer.quantize(model)
        quant_output = quant_model(dummy_input)
        ts_quant_model = torch.jit.trace(quant_model, dummy_input)

        with tempfile.NamedTemporaryFile() as tmp_file:
            torch.jit.save(ts_quant_model, tmp_file.name)
            loaded = torch.jit.load(tmp_file.name)
        loaded_output = loaded(dummy_input)
        torch.testing.assert_close(quant_output, loaded_output)


if __name__ == '__main__':
    unittest.main()
