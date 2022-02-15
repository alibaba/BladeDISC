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
import torch_blade
import unittest

from tests.mlir.testing_utils import DiscTestCase

class TestMlirConvolution(DiscTestCase):
    def _test_conv(self, conv_func, inp_test_data=None):
        if inp_test_data is not None:
            test_data = inp_test_data
        else:
            test_data = torch.randn([20, 16, 50, 100], device=self.device)
        if (isinstance(test_data, torch.Tensor)):
            test_data = (test_data.to(self.device),)
        self._test_cvt_to_disc(conv_func, test_data)

    @unittest.skipIf(torch_blade.version.cuda_available, "disc-gpu not support 1d conv yet.")
    def test_conv1d(self):
        conv = torch.nn.Conv1d(16, 33, 3, stride=2, padding=2)
        self._test_conv(conv, torch.randn([20, 16, 60], device=self.device))

    def test_conv2d(self):
        conv = torch.nn.Conv2d(16, 33, (3, 4), stride=2, padding=[2, 1], dilation=2)
        self._test_conv(conv)

    @unittest.skipIf(torch_blade.version.cuda_available, "disc-gpu not support 3d conv yet.")
    def test_conv3d(self):
        conv = torch.nn.Conv3d(16, 33, (3, 4, 5), stride=[2, 1, 3], padding=2)
        self._test_conv(conv, torch.randn([20, 16, 60, 50, 100], device=self.device))


if __name__ == "__main__":
    unittest.main()
