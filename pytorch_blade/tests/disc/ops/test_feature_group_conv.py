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
from torch import nn
import torch_blade
import unittest
from tests.disc.testing_base import DiscTestCase

class TestMlirConvolution(DiscTestCase):
    def _test_conv(self, conv_func, inp_test_data=None):
        if inp_test_data is not None:
            test_data = inp_test_data
        else:
            test_data = torch.randn([1, 112, 50, 50], device=self.device)
        if (isinstance(test_data, torch.Tensor)):
            test_data = (test_data.to(self.device),)
        self._test_cvt_to_disc(conv_func, test_data)

    def test_feature_group_convolution(self):
        # traced to aten::_convolution
        conv = nn.Conv2d(112, 896, 3, stride=1, padding=1, groups=2)
        conv_func = nn.Sequential(conv, nn.ReLU(), nn.ReLU(), nn.ReLU())
        self._test_conv(conv_func)

if __name__ == "__main__":
    unittest.main()
