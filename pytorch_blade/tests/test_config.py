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

import torch
from torch_blade.config import Config
from torch_blade.pass_manager import _get_current_onnx_opset_version, _set_opset_version_from_config
from torch_blade.testing.common_utils import TestCase


class TestConfig(TestCase):
    def test_config_unchange(self):
        self.assertEqual(Config.get_current_context(), None)
        default_cfg = Config.get_current_context_or_new()
        new_cfg = Config()
        new_cfg.fp16_fallback_op_ratio = 0.5
        new_cfg.enable_mlir_amp = True
        new_cfg.disc_cpu_fast_math_level = 0
        new_cfg.disc_compile_for_multi_cuda_targets = False
        with new_cfg:
            self.assertEqual(len(Config.get_contexts()), 1)
            now_cfg = Config.get_current_context_or_new()
            self.assertEqual(now_cfg.disc_cpu_fast_math_level, 0)
            self.assertEqual(now_cfg.disc_compile_for_multi_cuda_targets, False)
        now_cfg = Config.get_current_context_or_new()
        self.assertEqual(default_cfg.fp16_fallback_op_ratio, now_cfg.fp16_fallback_op_ratio)
        self.assertEqual(default_cfg.enable_mlir_amp, now_cfg.enable_mlir_amp)
        self.assertEqual(default_cfg.disc_cpu_fast_math_level, now_cfg.disc_cpu_fast_math_level)
        self.assertEqual(
            default_cfg.disc_compile_for_multi_cuda_targets,
            now_cfg.disc_compile_for_multi_cuda_targets
        )

    def test_config_work(self):
        new_cfg = Config()
        new_cfg.fp16_fallback_op_ratio = 0.5
        new_cfg.enable_mlir_amp = True
        with new_cfg:
            now_cfg = Config.get_current_context()
            self.assertEqual(now_cfg.fp16_fallback_op_ratio, 0.5)
            self.assertTrue(now_cfg.enable_mlir_amp)

    def _test_list_config(self, attr, vals):
        new_cfg = Config()
        setattr(new_cfg, attr, vals)

        with new_cfg:
            cfg = Config.get_current_context()
            attr_vals = getattr(cfg, attr)
            self.assertEqual(attr_vals, vals)

    def _test_numeric_config(self, attr, val, error_val):
        new_cfg = Config()
        setattr(new_cfg, attr, val)
        with new_cfg:
            now_cfg = Config.get_current_context()
            self.assertEqual(getattr(now_cfg, attr), val)

        with self.assertRaises(AssertionError):
            setattr(new_cfg, attr, error_val)
            with new_cfg:
                pass

    def test_preserved_attributes(self):
        self._test_list_config('preserved_attributes', ['a', 'b', 'c'])

    def test_dynamic_tuning_shapes(self):
        # test valid shapes
        shapes = {
            "min": [[1, 5], [1]],
            "max": [[1, 10], [1]],
            "opts": [
                [[1, 6], [1]],
                [[1, 8], [1]],
            ]
        }
        new_cfg = Config()
        new_cfg.dynamic_tuning_shapes = shapes
        with new_cfg:
            now_cfg = Config.get_current_context()
            self.assertEqual(now_cfg.dynamic_tuning_shapes[0]['min'], shapes['min'])
            self.assertEqual(now_cfg.dynamic_tuning_shapes[0]['max'], shapes['max'])
            self.assertEqual(now_cfg.dynamic_tuning_shapes[0]['opts'], shapes['opts'])

        def check_invalid(shapes):
            with self.assertRaises(Exception):
                new_cfg = Config()
                new_cfg.dynamic_tuning_shapes = shapes

        # test invalid shapes
        # min range is larger than max range
        shapes = {
            "min": [[1, 10], [1]],
            "max": [[1, 5], [1]],
            "opts": [
                [[1, 6], [1]],
                [[1, 8], [1]],
            ]
        }
        check_invalid(shapes)

        # dimension mismatch in max
        shapes = {
            "min": [[1, 10], [1]],
            "max": [[1, 5]],
            "opts": [
                [[1, 6], [1]],
                [[1, 8], [1]],
            ]
        }
        check_invalid(shapes)

        # dimension mismatch in opts
        shapes = {
            "min": [[1, 10], [1]],
            "max": [[1, 5], [1]],
            "opts": [
                [[1, 6]],
                [[1, 8], [1]],
            ]
        }
        check_invalid(shapes)

        # missing max
        shapes = {
            "min": [[1, 10], [1]],
            "opts": [
                [[1, 6], [1]],
                [[1, 8], [1]],
            ]
        }
        check_invalid(shapes)

        # check multiple dynamic ranges
        shapes1 = {
            "min": [[1, 5], [1]],
            "max": [[1, 10], [1]],
            "opts": [
                [[1, 6], [1]],
                [[1, 8], [1]],
            ]
        }
        shapes2 = {
            "min": [[5, 25], [2]],
            "max": [[20, 200], [2]],
            "opts": [
                [[5, 25], [2]],
                [[15, 150], [2]],
            ]
        }
        shapes = [shapes1, shapes2]
        new_cfg = Config()
        new_cfg.dynamic_tuning_shapes = shapes
        with new_cfg:
            now_cfg = Config.get_current_context()
            self.assertEqual(now_cfg.dynamic_tuning_shapes[0]['min'], shapes1['min'])
            self.assertEqual(now_cfg.dynamic_tuning_shapes[0]['max'], shapes1['max'])
            self.assertEqual(now_cfg.dynamic_tuning_shapes[0]['opts'], shapes1['opts'])

            self.assertEqual(now_cfg.dynamic_tuning_shapes[1]['min'], shapes2['min'])
            self.assertEqual(now_cfg.dynamic_tuning_shapes[1]['max'], shapes2['max'])
            self.assertEqual(now_cfg.dynamic_tuning_shapes[1]['opts'], shapes2['opts'])

    def test_dynamic_tuning_inputs(self):
        min_inp = {1: torch.tensor(1, device=self.device)}
        max_inp = {3: torch.tensor(3, device=self.device)}
        opt_inp = {2: torch.tensor(1, device=self.device)}
        cfg = Config()
        cfg.dynamic_tuning_inputs = {
            "min": [min_inp],
            "max": [max_inp],
            "opts": [
                [opt_inp]
            ]
        }
        with cfg:
            trt_dynamic_inputs = Config.get_current_context_or_new().dynamic_tuning_inputs
            self.assertEqual(len(trt_dynamic_inputs), 1)
            self.assertEqual(trt_dynamic_inputs[0]['min'], [min_inp])
            self.assertEqual(trt_dynamic_inputs[0]['max'], [max_inp])
            self.assertEqual(trt_dynamic_inputs[0]['opts'], [[opt_inp]])

    def test_customize_onnx_version(self):
        self._test_numeric_config('customize_onnx_opset_version', 9, -1)

        # test get_current_onnx_opset_version
        new_cfg = Config()
        new_cfg.customize_onnx_opset_version = 10
        with new_cfg:
            _set_opset_version_from_config()
            current_onnx_opset_version = _get_current_onnx_opset_version()
            self.assertEqual(current_onnx_opset_version, 10)

    def test_enable_force_to_cuda(self):
        self._test_numeric_config('enable_force_to_cuda', True, -1)

    def test_customize_op_black_list(self):
        self._test_list_config('customize_op_black_list', ['aten::size', 'aten::view'])

    def test_customize_op_white_list(self):
        self._test_list_config('customize_op_white_list', ['aten::size', 'aten::view'])


if __name__ == "__main__":
    unittest.main()
