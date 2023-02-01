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

from torch_blade import utils
from torch_blade.version import cuda_available
from tests.disc.testing_base import \
    DiscTestCase, skipTorchGE, skipTorchLT, isTorchMlirEnable


class TestDiscNNOps(DiscTestCase):

    def _test_nn_ops(self, nn_ops_func, x=None):
        test_data = torch.randn([2, 3, 224, 224], device=self.device) \
            if x is None else x
        if (isinstance(test_data, torch.Tensor)):
            test_data = (test_data.to(self.device),)
        self._test_cvt_to_disc(nn_ops_func, test_data)

    def test_softmax(self):
        softmax = torch.nn.Softmax(dim=-1)
        self._test_nn_ops(softmax)

        if isTorchMlirEnable() or utils.torch_version_number() \
                >= utils.parse_version("1.12.0"):
            # TODO(yancey.yx): support i32 input
            return

        @torch.jit.script
        def softmax_func(x):
            return torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)

        self._test_nn_ops(
            softmax_func, torch.randint(-3, 3, [2, 3, 4], dtype=torch.int32)
        )

    def test_log_softmax(self):
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        self._test_nn_ops(log_softmax)

        if isTorchMlirEnable() or utils.torch_version_number() \
                >= utils.parse_version("1.12.0"):
            # TODO(tanyo): support i32 input
            return

        @torch.jit.script
        def log_softmax_func(x):
            return torch.nn.functional.log_softmax(x, dim=-1,
                                                   dtype=torch.float32)

        self._test_nn_ops(
            log_softmax_func,
            torch.randint(-3, 3, [2, 3, 4], dtype=torch.int32)
        )

    @unittest.skipIf(not cuda_available, "run only on cuda")
    def test_softmax_return_half(self):
        @torch.jit.script
        def softmax_func_return_half(x):
            return torch.nn.functional.softmax(x, dim=1, dtype=torch.half)

        @torch.jit.script
        def log_softmax_func_return_half(x):
            return torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.half)

        inputs = (torch.randn([2, 3, 4], device=self.device),)
        self._test_cvt_to_disc(
            softmax_func_return_half, inputs, atol=1e-2
        )
        self._test_cvt_to_disc(
            log_softmax_func_return_half, inputs, atol=1e-2
        )

    @unittest.skipIf(not cuda_available, "Please fix incorrect results")
    @skipTorchGE("1.12.0")
    def test_softmax_func(self):
        @torch.jit.script
        def softmax(x):
            reduce_dim = -1
            x = torch.sub(x, 255)
            exp = x.exp()
            sum_exp = exp.sum(dim=reduce_dim, keepdim=True)
            b_sum_exp = sum_exp.expand_as(exp)
            return exp / b_sum_exp

        self._test_nn_ops(softmax)

    def test_masked_fill(self):
        @torch.jit.script
        def masked_fill(x, mask):
            return x.masked_fill(mask, 5)

        x = torch.randn([2, 3], device=self.device)
        mask = torch.tensor([[1] * 3] * 2, device=self.device).to(torch.bool)
        self._test_nn_ops(masked_fill, x=(x, mask))

        x = torch.randn([2, 3, 224, 224], device=self.device)
        mask = torch.randint(1, [224], device=self.device).to(torch.bool)
        self._test_nn_ops(masked_fill, x=(x, mask))

    def test_embedding(self):
        embedding = torch.nn.Embedding(10, 3).to(self.device)
        input = torch.LongTensor([[0, 2, 0, 5]]).to(self.device)
        self._test_nn_ops(embedding, input)
        input = torch.LongTensor([0, 2, 0, 5]).to(self.device)
        self._test_nn_ops(embedding, input)
        input = torch.LongTensor([9]).to(self.device)
        self._test_nn_ops(embedding, input)
        input = torch.LongTensor([]).to(self.device)
        self._test_nn_ops(embedding, input)
        input = torch.tensor(8, dtype=torch.int64).to(self.device)
        self._test_nn_ops(embedding, input)

    def test_gru_cell(self):
        # TODO(gty): Support torch.nn.GRUCell(4, 8, false)
        rnn = torch.nn.GRUCell(4, 8).to(self.device)
        input = torch.ones(3, 4, device=self.device)
        hx = torch.ones(3, 8, device=self.device)
        self._test_nn_ops(rnn, (input, hx))

    @skipTorchLT("1.12.0")
    def test_constant_pad(self):
        @torch.jit.script
        def constant_pad(t4d):
            # pad by (0, 1), (2, 1), and (3, 3)
            p3d = (0, 1, 2, 1, 3, 3)
            return t4d.pad(p3d, "constant", 3.0)

        x = torch.randn([2, 3, 4, 5], device=self.device)
        self._test_nn_ops(constant_pad, x=(x,))

        x = torch.randn([2, 3, 224, 224], device=self.device)
        self._test_nn_ops(constant_pad, x=(x,))


if __name__ == "__main__":
    unittest.main()
