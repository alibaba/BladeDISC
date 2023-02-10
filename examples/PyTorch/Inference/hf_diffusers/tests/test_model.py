# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from tempfile import TemporaryDirectory

import torch
from blade_adapter import BladeCLIPTextModel
from transformers import CLIPTextModel

CACHED_DIR = 'model_cache/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819'


class ModelTest(unittest.TestCase):
    def test_text_encoder(self):
        model_dir = os.path.join(CACHED_DIR, 'text_encoder')
        opt_model = BladeCLIPTextModel.from_original(model_dir)
        original_model = CLIPTextModel.from_pretrained(model_dir)
        example_inputs = BladeCLIPTextModel.gen_example_input()
        opt_output = opt_model(example_inputs)
        golden_output = original_model(example_inputs)
        torch.testing.assert_close(opt_output[0], golden_output[0])

        with TemporaryDirectory() as tmpdir:
            opt_model.save_pretrained(tmpdir)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, 'model.jit')))

            opt_model_2 = BladeCLIPTextModel.from_opt(tmpdir)
            opt_output_2 = opt_model_2(example_inputs)
            torch.testing.assert_close(opt_output_2[0], golden_output[0])

    # TODO(litan.ls): other model test


if __name__ == '__main__':
    unittest.main()
