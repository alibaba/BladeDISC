// RUN: disc-opt -split-input-file -pass-pipeline='builtin.module(func.func(disc-fusion{gpu-enabled=false fusion-strategy=base}))' %s | FileCheck %s --check-prefix=BASE
// RUN: DISC_ENABLE_TRANSFORM_SCHEDULE=1 disc-opt -split-input-file -pass-pipeline='builtin.module(func.func(disc-fusion{gpu-enabled=false fusion-strategy=stitch}))' %s -o - | FileCheck %s --check-prefix=TRANSFORM

// BASE-LABEL: @custom_call_op
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "cpu">, %[[ARG1:.*]]: memref<?x?xf32, "cpu">, %[[ARG2:.*]]: memref<?x?xf32, "cpu">, %[[ARG3:.*]]: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu">
func.func @custom_call_op(%arg0: memref<?x?xf32, "cpu">, %arg1: memref<?x?xf32, "cpu">,
                     %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // BASE-NEXT: lmhlo.abs
  // BASE-NEXT: lmhlo_disc.custom_call
  // BASE-NEXT: lmhlo.add
  // BASE-NEXT: return
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo_disc.custom_call"(%arg1, %arg2) {backend_config = "{}", call_target_name = "test", disc.device = "cpu", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// TRANSFORM-LABEL: @matmul_nn
func.func @matmul_nn(%arg0: memref<?x?xf32, "cpu">, %arg1: memref<?x?xf32, "cpu">,
                     %arg2: memref<?x?xf32, "cpu">, %arg3: memref<?x?xf32, "cpu">) -> memref<?x?xf32, "cpu"> {
  // TRANSFORM: "lmhlo.fusion"() ({
  // TRANSFORM-NEXT: lmhlo.dot_general
  // TRANSFORM-NEXT: lmhlo.terminator
  // TRANSFORM-NEXT: })
  // TRANSFORM-SAME: disc.fusion_type = "kTransform"
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  "lmhlo.dot_general"(%arg1, %arg2, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">, memref<?x?xf32, "cpu">) -> ()
  return %arg3 : memref<?x?xf32, "cpu">
}

// -----

// TRANSFORM-LABEL: @matmul_nn_const_weight
func.func @matmul_nn_const_weight(%arg0: memref<1024x1024xf32, "cpu">, %arg1: memref<1024x1024xf32, "cpu">,
                                  %arg2: memref<?x1024xf32, "cpu">, %arg3: memref<?x1024xf32, "cpu">) -> memref<?x1024xf32, "cpu"> {
  // TRANSFORM: lmhlo.constant
  // TRANSFORM-NEXT: "lmhlo.fusion"() ({
  // TRANSFORM-NEXT: lmhlo.constant
  // TRANSFORM-NEXT: lmhlo.dot_general
  // TRANSFORM-NEXT: lmhlo.terminator
  // TRANSFORM-NEXT: })
  // TRANSFORM-SAME: disc.fusion_type = "kTransform"
  "lmhlo.constant"(%arg0) {disc.device = "cpu", value = dense<1.0> : tensor<1024x1024xf32>} : (memref<1024x1024xf32, "cpu">) -> ()
  "lmhlo.constant"(%arg1) {disc.device = "cpu", value = dense<-1.0> : tensor<1024x1024xf32>} : (memref<1024x1024xf32, "cpu">) -> ()
  "lmhlo.dot_general"(%arg2, %arg1, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x1024xf32, "cpu">, memref<1024x1024xf32, "cpu">, memref<?x1024xf32, "cpu">) -> ()
  return %arg3 : memref<?x1024xf32, "cpu">
}

// -----

// TRANSFORM-LABEL: @matmul_nn_const_weight_with_external_user
func.func @matmul_nn_const_weight_with_external_user(%arg1: memref<1024x1024xf32, "cpu">,
                                                     %arg2: memref<?x1024xf32, "cpu">, %arg3: memref<?x1024xf32, "cpu">) -> (memref<1024x1024xf32, "cpu">, memref<?x1024xf32, "cpu">) {
  // TRANSFORM: lmhlo.constant
  // TRANSFORM-NEXT: "lmhlo.fusion"() ({
  // TRANSFORM-NEXT: lmhlo.dot_general
  // TRANSFORM-NEXT: lmhlo.terminator
  // TRANSFORM-NEXT: })
  // TRANSFORM-SAME: disc.fusion_type = "kTransform"
  // TRANSFORM-NEXT: return
  "lmhlo.constant"(%arg1) {disc.device = "cpu", value = dense<-1.0> : tensor<1024x1024xf32>} : (memref<1024x1024xf32, "cpu">) -> ()
  "lmhlo.dot_general"(%arg2, %arg1, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x1024xf32, "cpu">, memref<1024x1024xf32, "cpu">, memref<?x1024xf32, "cpu">) -> ()
  return %arg1, %arg3 : memref<1024x1024xf32, "cpu">, memref<?x1024xf32, "cpu">
}

// -----

// TRANSFORM-LABEL: @matmul_nn_const_weight_with_epilogue0
func.func @matmul_nn_const_weight_with_epilogue0(%arg1: memref<1024x1024xf32, "cpu">,
                                                 %arg2: memref<?x1024xf32, "cpu">, %arg3: memref<?x1024xf32, "cpu">,
                                                 %arg4: memref<f32, "cpu">,
                                                 %arg5: memref<2xindex, "cpu">) -> (memref<?x1024xf32, "cpu">) {
  "lmhlo.constant"(%arg1) {disc.device = "cpu", value = dense<-1.0> : tensor<1024x1024xf32>} : (memref<1024x1024xf32, "cpu">) -> ()
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg2, %c0 : memref<?x1024xf32, "cpu">
  %t0 = memref.alloc(%d0) {kDiscSymbolicDimAttr = [@S0, @C1024]} : memref<?x1024xf32, "cpu">
  // TRANSFORM: "lmhlo.fusion"() ({
  // TRANSFORM-NEXT: lmhlo.constant
  // TRANSFORM-SAME: value = dense<-1.000000e+00>
  // TRANSFORM-NEXT: lmhlo.dot_general
  // TRANSFORM-NEXT: lmhlo.constant
  // TRANSFORM-SAME: value = dense<1.000000e+00>
  // TRANSFORM-NEXT: lmhlo.dynamic_broadcast_in_dim
  // TRANSFORM-NEXT: lmhlo.add
  // TRANSFORM-NEXT: lmhlo.terminator
  // TRANSFORM-NEXT: })
  // TRANSFORM-SAME: disc.fusion_type = "kTransform"
  "lmhlo.dot_general"(%arg2, %arg1, %t0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x1024xf32, "cpu">, memref<1024x1024xf32, "cpu">, memref<?x1024xf32, "cpu">) -> ()
  "lmhlo.constant"(%arg4) {disc.device = "cpu", value = dense<1.0> : tensor<f32>} : (memref<f32, "cpu">) -> ()
  %t1 = memref.alloc(%d0) {kDiscSymbolicDimAttr = [@S0, @C1024]} : memref<?x1024xf32, "cpu">
  "lmhlo.dynamic_broadcast_in_dim"(%arg4, %arg5, %t1) {disc.device = "cpu", broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<f32, "cpu">, memref<2xindex, "cpu">, memref<?x1024xf32, "cpu">) -> ()
  %t2 = memref.alloc(%d0) {kDiscSymbolicDimAttr = [@S0, @C1024]} : memref<?x1024xf32, "cpu">
  "lmhlo.add"(%t0, %t1, %t2) :  (memref<?x1024xf32, "cpu">, memref<?x1024xf32, "cpu">, memref<?x1024xf32, "cpu">) -> ()
  return %t2 : memref<?x1024xf32, "cpu">
}

"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = true, sym_name = "C1024", value = 1024 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}
