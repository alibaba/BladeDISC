// RUN: DISC_ENABLE_TRANSFORM_SCHEDULE=1 disc-opt -split-input-file --disc-duplicate-computation-after-fusion -cse -loop-invariant-code-motion --canonicalize %s -o - | FileCheck %s --check-prefix=TRANSFORM

// TRANSFORM-LABEL: @duplicate_scalar_weight_for_ktransform
func.func @duplicate_scalar_weight_for_ktransform(
    %arg0 : memref<128x128xf32>, %arg1 : memref<128x128xf32>,
    %arg2 : memref<128x128xf32>, %arg3: memref<f32>, %arg4: memref<2xindex>,
    %arg5 : memref<128x128xf32>, %arg6 : memref<128x128xf32>) -> (memref<128x128xf32>) {
  "lmhlo.constant"(%arg3) {disc.device = "cpu", value = dense<2.000000e-01> : tensor<f32>} : (memref<f32>) -> ()
  // TRANSFORM: "lmhlo.fusion"
  // TRANSFORM-NEXT: "lmhlo.constant"(%[[T:.*]]) {disc.device = "cpu", value = dense<2.000000e-01> : tensor<f32>} : (memref<f32>) -> ()
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%arg0) {disc.device = "cpu", value = dense<0.1> : tensor<128x128xf32>} : (memref<128x128xf32>) -> ()
    "lmhlo.dot_general"(%arg1, %arg0, %arg2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%arg3, %arg4, %arg5) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<f32>, memref<2xindex>, memref<128x128xf32>) -> ()
    "lmhlo.add"(%arg2, %arg5, %arg6) : (memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "xxx0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg6 :  memref<128x128xf32>
}