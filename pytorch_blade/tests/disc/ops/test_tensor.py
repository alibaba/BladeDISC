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

from tests.disc.testing_base import DiscTestCase


class TestDiscTensor(DiscTestCase):
    def test_tensor_cat(self):
        @torch.jit.script
        def tensor_cat(x, y, z):
            return torch.cat([x, y, z], dim=1)

        x = torch.randn([2, 1, 4, 4]).to(self.device)
        y = torch.randn([2, 2, 4, 4]).to(self.device)
        z = torch.randn([2, 3, 4, 4]).to(self.device)
        test_data = (x, y, z)
        self._test_cvt_to_disc(tensor_cat, test_data)

        z = torch.randn([2, 0, 4, 4]).to(self.device)
        test_data = (x, y, z)
        self._test_cvt_to_disc(tensor_cat, test_data)

        x = torch.randint(3, [2, 1, 4, 4]).to(self.device)
        y = torch.randint(3, [2, 2, 4, 4]).to(self.device)
        z = torch.randint(3, [2, 3, 4, 4]).to(self.device)
        test_data = (x, y, z)
        self._test_cvt_to_disc(tensor_cat, test_data)

        z = torch.randint(3, [2, 0, 4, 4]).to(self.device)
        test_data = (x, y, z)
        self._test_cvt_to_disc(tensor_cat, test_data)

    def test_aten_item(self):
        @torch.jit.script
        def test_item(tensor):
            x = int(tensor.item())
            return torch.tensor(x)

        # TODO: aten:Int only support int32_t
        self._test_cvt_to_disc(test_item, (torch.tensor((1 << 31) - 1, dtype=torch.int64),))
        self._test_cvt_to_disc(test_item, (torch.tensor(-2),))

        @torch.jit.script
        def test_item_2(tensor):
            # Integer division of tensors using div or / is not supported only in torch
            # 1.6, but works in 1.7 and 1.8, while the error message says it is
            # supported until 1.6. So use // instead.
            x = tensor // torch.tensor(2)
            x = int(x)

            return torch.tensor(x) + tensor

        self._test_cvt_to_disc(test_item_2, (torch.tensor(-2),))


if __name__ == "__main__":
    unittest.main()
