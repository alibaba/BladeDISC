// RUN: disc-opt --canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: @multi_level_pack_fold
func.func @multi_level_pack_fold() -> tensor<1x2x2x2xf32> {
  // CHECK: %[[CST:.*]] =  arith.constant
  // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00], [4.000000e+00, 5.000000e+00]], [[2.000000e+00, 3.000000e+00], [6.000000e+00, 7.000000e+00]]]]> : tensor<1x2x2x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]> : tensor<2x4xf32>
  %0 = tensor.empty() : tensor<1x2x2x2xf32>
  %1 = disc_linalg_ext.multi_level_pack %cst with tile_levels = [1, 1] tile_sizes = [2, 2] permutation = [0, 2, 1, 3] into %0 : (tensor<2x4xf32> tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}

// -----

// CHECK-LABEL: @multi_level_pack_fold_2
func.func @multi_level_pack_fold_2() -> tensor<1x2x2x2xf32> {
  // CHECK: %[[CST:.*]] =  arith.constant
  // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 4.000000e+00], [1.000000e+00, 5.000000e+00]], [[2.000000e+00, 6.000000e+00], [3.000000e+00, 7.000000e+00]]]]> : tensor<1x2x2x2xf32>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]> : tensor<2x4xf32>
  %0 = tensor.empty() : tensor<1x2x2x2xf32>
  %1 = disc_linalg_ext.multi_level_pack %cst with tile_levels = [1, 1] tile_sizes = [2, 2] permutation = [0, 2, 3, 1] into %0 : (tensor<2x4xf32> tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}

// -----

// CHECK-LABEL: @multi_level_pack_fold_with_pad
func.func @multi_level_pack_fold_with_pad() -> tensor<1x3x2x2xf32> {
  // CHECK: %[[CST:.*]] =  arith.constant
  // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00], [5.000000e+00, 6.000000e+00]], [[2.000000e+00, 3.000000e+00], [7.000000e+00, 8.000000e+00]], [[4.000000e+00, 1.280000e+02], [9.000000e+00, 1.280000e+02]]]]>
  // CHECK: return %[[CST]]
  %cst = arith.constant dense<[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]> : tensor<2x5xf32>
  %init = arith.constant 128.0 : f32
  %0 = tensor.empty() : tensor<1x3x2x2xf32>
  %1 = disc_linalg_ext.multi_level_pack %cst with padding_value(%init: f32) tile_levels = [1, 1] tile_sizes = [2, 2] permutation = [0, 2, 1, 3] into %0 : (tensor<2x5xf32> tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %1 : tensor<1x3x2x2xf32>
}

