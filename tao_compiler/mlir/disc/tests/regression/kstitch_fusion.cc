// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/disc/tests/mlir_feature_test.h"
#include "mlir/disc/tests/mlir_test.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir_test {

const std::string c_ft_path = "mlir/disc/tests/regression/data/";

// The column size is small enough to enable warp-wise reduction schedule.
TEST(KStitchFusionGPUTest, KStitchSimpleSmallColumnF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_simple.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"11000x123xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// The column size is large enough to enable block-wise reduction schedule.
TEST(KStitchFusionGPUTest, KStitchSimpleLargeColumnF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_simple.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"11000x12345xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// There is a root op that is not a skeleton op in the kStitch fusion.
TEST(KStitchFusionGPUTest, KStitchNonSkeletonOutputF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  for (int i = 0; i < 2; ++i) {
    bool enable_shape_constraint_ir = (i == 0);
    setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR",
           enable_shape_constraint_ir ? "true" : "false", 1);
    if (enable_shape_constraint_ir) {
      setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
    } else {
      // TODO: even though it means kStitch works when there are 2 kernels
      // formed, we can further reduce the kernel number to 1 after optimizing
      // shape ops like compute_reshape_shape successfully.
      setenv("DISC_EXPECTED_KERNELS_IN_UT", "2", 1);
    }
    EXPECT_TRUE(feature_test_main(
        /*mlir_file_path*/ c_ft_path + "kstitch_fusion_non_skl_output.mlir",
        /*backend_types*/ {BackendType::kCuda},
        /*num_inputs*/ 3,
        /*num_outputs*/ 2,
        /*input_descriptors*/ {"128x768xf32_X", "1x128x768xf32_X", "768xf32_X"},
        /*output_descriptors*/ {"f32_X", "f32_X"}));
  }
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// There are adjacent skeleton ops. The last output has different shoape whith
// that of sub-root. The data type is FP16.
TEST(KStitchFusionGPUTest, KStitchAdjacentSkeletonWithSmallOutputF16) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "kstitch_fusion_adj_skl_small_output_f16.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"110x100xf16_X"},
      /*output_descriptors*/ {"f16_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// There is an output cannot be covered by sub-root, thus cannot be fused into
// the kStitch fusion.
TEST(KStitchFusionGPUTest, KStitchNonCoverOutputF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_non_cover_output.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"200x300xf32_X", "2xi32_h", "2xi32_h"},
      /*output_descriptors*/ {"f32_X", "f32_X"},
      /*input_vals*/ {{}, {23, 40}, {123, -1}}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// There is an irregular xroot in the kStitch fusion.
TEST(KStitchFusionGPUTest, KStitchIrregularXrootF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  // TODO: even though it means kStitch works when there are 2 kernels formed,
  // we can further reduce the kernel number to 1 after optimizing shape ops
  // like compute_reshape_shape successfully.
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "2", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_irregular_xroot.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"10x112xf32_X", "10x112xf32_X", "10x112xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// The sub-root (i.e., row-reduction) is 3D. The datatype is FP64.
TEST(KStitchFusionGPUTest, KStitch3DF64) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_3d_f64.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"1x128x768xf64_X"},
      /*output_descriptors*/ {"f64_X", "f64_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// With splat const in the kStitch fusion
TEST(KStitchFusionGPUTest, KStitchSimpleWithSplatConstF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_with_splat_const.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"11000x123xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// With transpose in the kStitch fusion
TEST(KStitchFusionGPUTest, KStitchSimpleWithTransposeF32) {
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_with_transpose.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"123x11000xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
}

// The column size is large to enable block-wise reduction schedule.
TEST(KStitchFusionGPUTest, KStitchElemwiseShmCacheLarge) {
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "kstitch_fusion_elemwise_shm_cache_large.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 4,
      /*input_descriptors*/
      {"96x512x512xf16_X", "8x12x512x512xf16_X", "8x12x512x512xf32_X"},
      /*output_descriptors*/ {"f16_X", "f32_X", "f32_X", "f16_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
}

// The column size is small to enable warp-wise reduction schedule.
TEST(KStitchFusionGPUTest, KStitchElemwiseShmCacheSmall) {
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  setenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL", "true", 1);
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path +
          "kstitch_fusion_elemwise_shm_cache_small.mlir",
      /*backend_types*/ {BackendType::kCuda},
      /*num_inputs*/ 3,
      /*num_outputs*/ 4,
      /*input_descriptors*/
      {"96x512x64xf16_X", "8x12x512x64xf16_X", "8x12x512x64xf32_X"},
      /*output_descriptors*/ {"f16_X", "f32_X", "f32_X", "f16_X"}));
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
}

// multi-outputs
TEST(KStitchFusionCPUTest, MultiOutputs) {
  std::vector<float> input_val;
  for (int64_t i = 0; i < 1 * 128 * 768; i++) {
    input_val.push_back(0.5);
  }
  setenv("DISC_ENABLE_STITCH", "true", 1);
  setenv("DISC_EXPECTED_KERNELS_IN_UT", "1", 1);
  setenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR", "true", 1);
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "kstitch_fusion_cpu_multioutputs.mlir",
      /*backend_types*/ kSupportedCPUBackendList,
      /*num_inputs*/ 1,
      /*num_outputs*/ 2,
      /*input_descriptors*/ {"1x128x768xf32_X"},
      /*output_descriptors*/ {"f32_X", "f32_X"}, {input_val}));
  unsetenv("DISC_ENABLE_STITCH");
  unsetenv("DISC_EXPECTED_KERNELS_IN_UT");
  unsetenv("DISC_ENABLE_SHAPE_CONSTRAINT_IR");
}

}  // namespace mlir_test
