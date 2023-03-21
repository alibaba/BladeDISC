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
        self.assertTrue(isinstance(model.linear.bias_ob, BiasObserver))
        calib_model(dummy)
        quantized_model = quantizer.quantize(model)
        self.assertTrue(isinstance(quantized_model.linear.bias_ob, BiasObserver))

        qat_model = quantizer.qat(model)
        self.assertTrue(qat_model.linear.bias_ob is None)


if __name__ == "__main__":
    unittest.main()
