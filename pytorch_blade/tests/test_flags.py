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

import unittest

from torch_blade import tools
from torch_blade.testing.common_utils import TestCase


class TestFlags(TestCase):

    def _test_get_set(self, get_flag_func, set_flag_func, new_flag):
        orig_flag = get_flag_func()
        # set_flag_func should return orig flag
        self.assertEqual(orig_flag, set_flag_func(new_flag))
        # get_flag_func should reflect the most recent set flags
        self.assertEqual(new_flag, get_flag_func())
        set_flag_func(orig_flag)

    def _test_context(self, flag_context_func, get_flag_func):
        old_flag = get_flag_func()
        new_flag = not old_flag
        with flag_context_func(new_flag):
            # flags should be set to new_flags inside the context manager
            self.assertEqual(new_flag, get_flag_func())

        # flags should be reset to old_flags after going out of the
        # context manager
        self.assertEqual(old_flag, get_flag_func())

    def test_record_cluster_io_flag(self):
        self._test_get_set(tools.get_record_cluster_io_flag, tools.set_record_cluster_io_flag, True)
        self._test_context(tools.record_cluster_io_context, tools.get_record_cluster_io_flag)

    def test_trust_tracing_shape(self):
        self._test_get_set(tools.get_trust_tracing_shape, tools.set_trust_tracing_shape, True)
        self._test_context(tools.trust_tracing_shape, tools.get_trust_tracing_shape)

if __name__ == "__main__":
    unittest.main()
