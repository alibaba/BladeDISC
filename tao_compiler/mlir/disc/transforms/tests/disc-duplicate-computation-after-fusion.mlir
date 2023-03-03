// RUN: DISC_ENABLE_TRANSFORM_SCHEDULE=1 disc-opt -split-input-file --disc-duplicate-computation-after-fusion %s -o - | FileCheck %s --check-prefix=TRANSFORM

// TRANSFORM-LABEL: @duplicate_scalar_weight_for_ktransform
func.func @duplicate_scalar_weight_for_ktransform(
    %arg0 : memref<128x128xf32, "cpu">, %arg1 : memref<128x128xf32, "cpu">,
    %arg2 : memref<128x128xf32, "cpu">, %arg3: memref<f32, "cpu">, %arg4: memref<2xindex, "cpu">,
    %arg5 : memref<128x128xf32, "cpu">, %arg6 : memref<128x128xf32, "cpu">) -> (memref<128x128xf32, "cpu">) {
  "lmhlo.constant"(%arg3) {disc.device = "cpu", value = dense<2.000000e-01> : tensor<f32>} : (memref<f32, "cpu">) -> ()
  // TRANSFORM: "lmhlo.fusion"
  // TRANSFORM-NEXT: "lmhlo.constant"(%[[T:.*]]) {disc.device = "cpu", value = dense<2.000000e-01> : tensor<f32>} : (memref<f32, "cpu">) -> ()
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%arg0) {disc.device = "cpu", value = dense<0.1> : tensor<128x128xf32>} : (memref<128x128xf32, "cpu">) -> ()
    "lmhlo.dot_general"(%arg1, %arg0, %arg2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x128xf32, "cpu">, memref<128x128xf32, "cpu">, memref<128x128xf32, "cpu">) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%arg3, %arg4, %arg5) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<f32, "cpu">, memref<2xindex, "cpu">, memref<128x128xf32, "cpu">) -> ()
    "lmhlo.add"(%arg2, %arg5, %arg6) : (memref<128x128xf32, "cpu">, memref<128x128xf32, "cpu">, memref<128x128xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "xxx0", disc.fusion_type = "kTransform"} : () -> ()
  return %arg6 :  memref<128x128xf32, "cpu">
}