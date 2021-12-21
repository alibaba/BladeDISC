import torch
import unittest

from tests.mlir.testing_utils import DiscTestCase
from torch_addons import version

class Linear(torch.nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.weight = torch.ones(256, 256, device=device)
        self.bias = torch.ones(120, 256, device=device)

    def forward(self, x, y):
        out_y = torch.matmul(y, self.weight)
        out_b = out_y + self.bias
        return x + out_b


@unittest.skipIf(not version.cuda_available, "TorchAddons CUDA not available")
class TestDiscMultiStreams(DiscTestCase):
    def test_linear(self):
        x = torch.randn(4, 120, 256, device=self.device)
        y = torch.randn(4, 120, 256, device=self.device)
        test_data = (x, y)
        linear = Linear().to(self.device).eval()
        opt_module = self.cvt_to_disc(linear, test_data)

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        # Wait for the above tensors to initialise.
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            out1 = opt_module.forward(*test_data)
        with torch.cuda.stream(s2):
            out2 = opt_module.forward(*test_data)
        # Wait for C and D to be computed.
        torch.cuda.synchronize()
        self.assertEqual(out1, out2, rtol=5e-5, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
