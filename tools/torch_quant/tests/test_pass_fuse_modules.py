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

import torch
import unittest

from tests.models import LinearReLU, SimpleModule, create_ctx
from torch_quant.graph import fuse_modules


class FuseModulesTest(unittest.TestCase):
    def test_conv_relu(self) -> None:
        ctx = create_ctx(SimpleModule())
        fuse_modules(ctx)
        self.assertTrue(isinstance(ctx.gm.sub.conv, torch.nn.intrinsic.ConvReLU2d))

    def test_linear_relu(self) -> None:
        ctx = create_ctx(LinearReLU())
        fuse_modules(ctx)
        self.assertTrue(isinstance(ctx.gm.linear, torch.nn.intrinsic.LinearReLU))


if __name__ == '__main__':
    unittest.main()
