// RUN: disc-opt -split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<128x128xf32>) -> tensor<4x4x32x32xf32> {
  %0 = tensor.empty() : tensor<4x4x32x32xf32>
  // CHECK: disc_linalg_ext.multi_level_pack
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3] into %0 : (tensor<128x128xf32> tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
  return %1 : tensor<4x4x32x32xf32>
}