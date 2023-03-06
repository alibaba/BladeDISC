// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @multi_level_pack
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multi_level_pack(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?x32x32xf32> {
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[ARG1]], %[[ARG2]]) : tensor<?x?x32x32xf32>
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %c1 : tensor<?x?xf32>
  // CHECK: %[[T0:.*]] = scf.for %[[IV0:.*]] = %c0 to %[[DIM0]] step %c32 iter_args(%[[V0:.*]] = %[[INIT]]) -> (tensor<?x?x32x32xf32>) {
  // CHECK:   %[[T1:.*]] = scf.for %[[IV1:.*]] = %c0 to %[[DIM1]] step %c32 iter_args(%[[V1:.*]] = %[[V0]]) -> (tensor<?x?x32x32xf32>) {
  // CHECK:     %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME: %[[IV0]], %[[IV1]]
  // CHECK-SAME: 32, 32
  // CHECK-SAME: 1, 1
  // CHECK:     %[[UPDATE:.*]] = tensor.insert_slice %[[SLICE]] into %[[V1]]
  // CHECK:     scf.yield %[[UPDATE]] : tensor<?x?x32x32xf32>
  // CHECK:   scf.yield %[[T1]] : tensor<?x?x32x32xf32>
  // CHECK: return %[[T0]]
  %0 = tensor.empty(%arg1, %arg2) : tensor<?x?x32x32xf32>
  %1 = disc_linalg_ext.multi_level_pack %arg0 with tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3] into %0 : (tensor<?x?xf32> tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
  return %1 : tensor<?x?x32x32xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %pack_op = transform.structured.match ops{["disc_linalg_ext.multi_level_pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.lower_multi_level_pack_to_loop %pack_op
}

// -----

// CHECK-LABEL: @multi_level_pack_with_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multi_level_pack_with_pad(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?x32x32xf32> {
  %cst = arith.constant 1.0 : f32
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[ARG1]], %[[ARG2]]) : tensor<?x?x32x32xf32>
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %c1 : tensor<?x?xf32>
  // CHECK: %[[T0:.*]] = scf.for %[[IV0:.*]] = %c0 to %[[DIM0]] step %c32 iter_args(%[[V0:.*]] = %[[INIT]]) -> (tensor<?x?x32x32xf32>) {
  // CHECK:   %[[T1:.*]] = scf.for %[[IV1:.*]] = %c0 to %[[DIM1]] step %c32 iter_args(%[[V1:.*]] = %[[V0]]) -> (tensor<?x?x32x32xf32>) {
  // CHECK:     %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME: %[[IV0]], %[[IV1]]
  // CHECK:     %[[PAD:.*]] = tensor.pad %[[SLICE]]
  // CHECK:     %[[UPDATE:.*]] = tensor.insert_slice %[[PAD]] into %[[V1]]
  // CHECK:     scf.yield %[[UPDATE]] : tensor<?x?x32x32xf32>
  // CHECK:   scf.yield %[[T1]] : tensor<?x?x32x32xf32>
  // CHECK: return %[[T0]]
  %0 = tensor.empty(%arg1, %arg2) : tensor<?x?x32x32xf32>
  %1 = disc_linalg_ext.multi_level_pack %arg0 with padding_value(%cst : f32) tile_levels = [1, 1] tile_sizes = [32, 32] permutation = [0, 2, 1, 3] into %0 : (tensor<?x?xf32> tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
  return %1 : tensor<?x?x32x32xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %pack_op = transform.structured.match ops{["disc_linalg_ext.multi_level_pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.lower_multi_level_pack_to_loop %pack_op
}

// -----

// CHECK-LABEL: @multi_level_pack_with_pad_2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multi_level_pack_with_pad_2(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?x32x1xf32> {
  %cst = arith.constant 1.0 : f32
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[ARG1]], %[[ARG2]]) : tensor<?x?x32x1xf32>
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %c1 : tensor<?x?xf32>
  // CHECK: %[[T0:.*]] = scf.for %[[IV0:.*]] = %c0 to %[[DIM0]] step %c32 iter_args(%[[V0:.*]] = %[[INIT]]) -> (tensor<?x?x32x1xf32>) {
  // CHECK:   %[[T1:.*]] = scf.for %[[IV1:.*]] = %c0 to %[[DIM1]] step %c1 iter_args(%[[V1:.*]] = %[[V0]]) -> (tensor<?x?x32x1xf32>) {
  // CHECK:     %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME: %[[IV0]], %[[IV1]]
  // CHECK:     %[[PAD:.*]] = tensor.pad %[[SLICE]]
  // CHECK-SAME: low[0, 0]
  // CHECK-SAME: high[%[[DIM_HIGH:.*]], %c0]
  // CHECK:     %[[UPDATE:.*]] = tensor.insert_slice %[[PAD]] into %[[V1]]
  // CHECK:     scf.yield %[[UPDATE]] : tensor<?x?x32x1xf32>
  // CHECK:   scf.yield %[[T1]] : tensor<?x?x32x1xf32>
  // CHECK: return %[[T0]]
  %0 = tensor.empty(%arg1, %arg2) : tensor<?x?x32x1xf32>
  %1 = disc_linalg_ext.multi_level_pack %arg0 with padding_value(%cst : f32) tile_levels = [1, 1] tile_sizes = [32, 1] permutation = [0, 2, 1, 3] into %0 : (tensor<?x?xf32> tensor<?x?x32x1xf32>) -> tensor<?x?x32x1xf32>
  return %1 : tensor<?x?x32x1xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %pack_op = transform.structured.match ops{["disc_linalg_ext.multi_level_pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.lower_multi_level_pack_to_loop %pack_op
}