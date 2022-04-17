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

import unittest

from torch_blade import tensorrt
from torch_blade.testing.common_utils import TestCase
from tests.tensorrt import skipIfNoTensorRT


@skipIfNoTensorRT()
class TestTensorRTBuilderFlags(TestCase):
    def test_enum(self):
        self.assertTrue(hasattr(tensorrt.BuilderFlag, "FP16"))
        # Make sure the shift operation works
        1 << int(tensorrt.BuilderFlag.FP16)

    def test_get_set(self):
        orig_flags = tensorrt.get_builder_flags()
        new_flags = 1 << int(tensorrt.BuilderFlag.FP16)

        # set_builder_flags should return orig flags
        self.assertEqual(orig_flags, tensorrt.set_builder_flags(new_flags))

        # get_builder_flags should reflect the most recent set flags
        self.assertEqual(new_flags, tensorrt.get_builder_flags())

        # reset
        tensorrt.set_builder_flags(orig_flags)

    def test_context(self):
        old_flags = tensorrt.get_builder_flags()
        new_flags = 1 << int(tensorrt.BuilderFlag.FP16)
        with tensorrt.builder_flags_context(new_flags):
            # flags should be set to new_flags inside the context manager
            self.assertEqual(new_flags, tensorrt.get_builder_flags())

        # flags should be reset to old_flags after going out of the
        # context manager
        self.assertEqual(old_flags, tensorrt.get_builder_flags())


if __name__ == "__main__":
    unittest.main()
