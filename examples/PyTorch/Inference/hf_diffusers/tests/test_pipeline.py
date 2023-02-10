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

import unittest
from tempfile import TemporaryDirectory

from blade_adapter import BladeStableDiffusionPipeline

CACHED_DIR = 'model_cache/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819'
PIPE_ID = 'runwayml/stable-diffusion-v1-5'


class PipelineTest(unittest.TestCase):
    def test_overwrite_config(self):
        with TemporaryDirectory() as tmpdir:
            BladeStableDiffusionPipeline.overwrite_config(CACHED_DIR, tmpdir)
            new_config = BladeStableDiffusionPipeline.load_config(tmpdir)
            self.assertEqual(new_config['text_encoder'], [
                             'blade_adapter', 'BladeCLIPTextModel'])

    def test_from_pretrained(self):
        self.assertRaises(NotImplementedError,
                          BladeStableDiffusionPipeline.from_pretrained, PIPE_ID)
        pipe = BladeStableDiffusionPipeline.from_pretrained(CACHED_DIR)
        # TODO(litan.ls): compare pipeline output


if __name__ == '__main__':
    unittest.main()
