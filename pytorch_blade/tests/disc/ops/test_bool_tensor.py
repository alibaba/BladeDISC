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
import unittest
from tests.disc.testing_base import DiscTestCase, skipTorchLE

@skipTorchLE("1.6.1")
class TestMlirBoolTensor(DiscTestCase):
    def _test_bool_tensor(self, conv_func, inp_test_data=None):
        if inp_test_data is not None:
            test_data = inp_test_data
        else:
            test_data = torch.randn([2, 2], device=self.device)
        if (isinstance(test_data, torch.Tensor)):
            test_data = (test_data.to(self.device),)
        self._test_cvt_to_disc(conv_func, test_data)

    def test_bool_tensor(self):
        @torch.jit.script
        def where_func(mat):
            mask = torch.tensor([[True, True], [False, True]])
            where = torch.where(mask, mat, 0.0)
            return where
        self._test_bool_tensor(where_func)

if __name__ == "__main__":
    unittest.main()
