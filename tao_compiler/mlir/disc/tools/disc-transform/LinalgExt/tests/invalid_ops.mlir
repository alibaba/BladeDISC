// RUN: disc-opt -split-input-file %s -verify-diagnostics

func.func @main(%arg0: tensor<128x128xf32>) -> tensor<4x4x32x32xf32> {
  %0 = tensor.empty() : tensor<4x4x32x32xf32>
  // expected-error@+1 {{mismatch input rank and the size of tile_levels 2 vs 3}}
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3] into %0 : (tensor<128x128xf32> tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
  return %1 : tensor<4x4x32x32xf32>
}

// -----

func.func @main(%arg0: tensor<128x128xf32>) -> tensor<4x4x32x32xf32> {
  %0 = tensor.empty() : tensor<4x4x32x32xf32>
  // expected-error@+1 {{mismatch expected output rank and the size of permutation 4 vs 5}}
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3, 5] into %0 : (tensor<128x128xf32> tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
  return %1 : tensor<4x4x32x32xf32>
}

// -----

func.func @main(%arg0: tensor<128x128xf32>) -> tensor<4x4x32x32x1xf32> {
  %0 = tensor.empty() : tensor<4x4x32x32x1xf32>
  // expected-error@+1 {{mismatch expected output rank and the rank of the output operand 4 vs 5}}
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3] into %0 : (tensor<128x128xf32> tensor<4x4x32x32x1xf32>) -> tensor<4x4x32x32x1xf32>
  return %1 : tensor<4x4x32x32x1xf32>
}

// -----

func.func @main(%arg0: tensor<128x128xf32>) -> tensor<4x4x32x32xf32> {
  %0 = tensor.empty() : tensor<4x4x32x32xf32>
  // expected-error@+1 {{not a valid permutation setting}}
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 2] into %0 : (tensor<128x128xf32> tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
  return %1 : tensor<4x4x32x32xf32>
}

// -----

func.func @main(%arg0: tensor<129x128xf32>) -> tensor<4x4x32x32xf32> {
  %0 = tensor.empty() : tensor<4x4x32x32xf32>
  // expected-error@+1 {{mismatch expected output type and actual output type 'tensor<5x4x32x32xf32>' vs 'tensor<4x4x32x32xf32>'}}
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3] into %0 : (tensor<129x128xf32> tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
  return %1 : tensor<4x4x32x32xf32>
}