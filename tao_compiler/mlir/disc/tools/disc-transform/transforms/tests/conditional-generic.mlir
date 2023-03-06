// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s --dump-input=always

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @tile_conditional_generic
func.func @tile_conditional_generic(%pred : i1, %arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.extract_slice
  // CHECK-NEXT: disc_linalg_ext.conditional_generic
  %out = disc_linalg_ext.conditional_generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "parallel"]}
      ins(%pred : i1)
      outs(%arg0 : tensor<?x?xf32>) {
  ^bb0(%arg4: i1, %arg3: f32):
    disc_linalg_ext.yield %cst : f32
  } -> tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["disc_linalg_ext.conditional_generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile %0 [288, 512] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// -----

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @fuse_into_containing_op_conditional_generic
func.func @fuse_into_containing_op_conditional_generic(
    %pred : i1, %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = disc_linalg_ext.conditional_generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "parallel"]}
      ins(%pred : i1)
      outs(%arg0 : tensor<?x?xf32>) {
  ^bb0(%arg4: i1, %arg3: f32):
    disc_linalg_ext.yield %cst : f32
  } -> tensor<?x?xf32>
  // CHECK: scf.for
  // CHECK:   %[[T0:.*]] = disc_linalg_ext.conditional_generic
  // CHECK:   scf.for
  // CHECK:     linalg.matmul
  // CHECK-SAME:  %[[T0]]
  %1 = linalg.matmul ins(%0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["disc_linalg_ext.conditional_generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %1 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %2, %loops:2 = transform.structured.tile %1 [2, 3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
    transform.structured.fuse_into_containing_op %0 into %loops#0
}

// -----

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (6, d0)>
#map3 = affine_map<(d0) -> (16, d0)>

// CHECK-LABEL: @pad_conditional_generic
func.func @pad_conditional_generic(%pred : i1, %arg1 : index, %arg2 : index, %arg_out : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = affine.min #map2(%arg1)
  %d1 = affine.min #map2(%arg2)
  %1 = tensor.extract_slice %arg_out[0, 0] [%d0, %d1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  // CHECK: %[[T0:.*]] = tensor.pad
  // CHECK: %[[T1:.*]] = disc_linalg_ext.conditional_generic
  // CHECK-SAME: outs(%[[T0]] : tensor<6x6xf32>)
  // CHECK: tensor.extract_slice
  // CHECK-SAME: %[[T1]]
  %out = disc_linalg_ext.conditional_generic {indexing_maps = [#map0, #map1],
      iterator_types = ["parallel", "parallel"]}
      ins(%pred : i1)
      outs(%1 : tensor<?x?xf32>) {
  ^bb0(%arg4: i1, %arg3: f32):
    disc_linalg_ext.yield %cst : f32
  } -> tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["disc_linalg_ext.conditional_generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.pad %0 {padding_dimensions = [0, 1], padding_values = [0 : i1, 0.000000e+00 : f32]}
}

