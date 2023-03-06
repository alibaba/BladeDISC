// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=1 DISC_ENABLE_HORIZONTAL_FUSION=1 disc-opt \
// RUN:   -pass-pipeline='builtin.module(func.func(disc-specialize-fusion-with-speculation{core-count=72 max-threads-per-core=1536}))' \
// RUN:   %s --split-input-file | FileCheck %s

// CHECK-LABEL: simple_broadcast_specialization
func.func @simple_broadcast_specialization(%arg0: !disc_ral.context) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32, "gpu">
  %1 = memref.dim %0, %c0 : memref<?x?xf32, "gpu">
  %2 = memref.dim %0, %c1 : memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %0 to offset: [0], sizes: [%1, %2], strides: [%2, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = tensor.from_elements %1, %2 : tensor<2xindex>
  %5 = bufferization.to_memref %4 : memref<2xindex>
  %6 = memref.alloc(%1, %2) {kDiscSymbolicDimAttr = [@S2, @S3]} : memref<?x?xf32, "gpu">
  %7 = memref.alloc(%1, %2) {kDiscSymbolicDimAttr = [@S2, @S3]} : memref<?x?xf32, "gpu">
  %8 = memref.alloc(%1, %2) {kDiscSymbolicDimAttr = [@S2, @S3]} : memref<?x?xf32, "gpu">
  %9 = memref.alloc() : memref<f32, "gpu">
  // CHECK: %[[T0:.*]] = "disc_ral.recv_input"
  // CHECK: %[[T3:.*]] = memref.reinterpret_cast %[[T0]]
  // CHECK: %[[T6:.*]] = memref.alloc{{.*}} : memref<?x?xf32, "gpu">
  // CHECK: %[[T7:.*]] = memref.alloc{{.*}} : memref<?x?xf32, "gpu">
  // CHECK: %[[T8:.*]] = memref.alloc{{.*}} : memref<?x?xf32, "gpu">
  // CHECK: %[[T9:.*]] = memref.alloc{{.*}} : memref<f32, "gpu">

  // CHECK-DAG: %[[TT3:.*]] = memref.reinterpret_cast %[[T3]]
  // CHECK-DAG: %[[TT6:.*]] = memref.reinterpret_cast %[[T6]]
  // CHECK-DAG: %[[TT7:.*]] = memref.reinterpret_cast %[[T7]]
  // CHECK-DAG: %[[TT8:.*]] = memref.reinterpret_cast %[[T8]]

  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[TT9:.*]] = memref.dim %[[TT3]], %[[c0_1]] : memref<?x?xf32, "gpu">
  // CHECK: %[[c0_2:.*]] = arith.constant 0 : index
  // CHECK: %[[TT10:.*]] = memref.dim %[[TT7]], %[[c0_2]] : memref<?x?xf32, "gpu">
  // CHECK: %[[TT11:.*]] = arith.cmpi eq, %[[TT9]], %[[TT10]] : index
  // CHECK: %[[c1_3:.*]] = arith.constant 1 : index
  // CHECK: %[[TT12:.*]] = memref.dim %[[TT3]], %[[c1_3]] : memref<?x?xf32, "gpu">
  // CHECK: %[[c1_4:.*]] = arith.constant 1 : index
  // CHECK: %[[TT13:.*]] = memref.dim %[[TT7]], %[[c1_4]] : memref<?x?xf32, "gpu">
  // CHECK: %[[TT14:.*]] = arith.cmpi eq, %[[TT12]], %[[TT13]] : index
  // CHECK: %[[TT15:.*]] = arith.andi %[[TT14]], %[[TT11]] : i1
  // CHECK: scf.if %[[TT15]] {
  // CHECK-DAG: %[[TTT3:.*]] = memref.reinterpret_cast %[[TT3]]
  // CHECK-DAG: %[[TTT6:.*]] = memref.reinterpret_cast %[[TT6]]
  // CHECK-DAG: %[[TTT8:.*]] = memref.reinterpret_cast %[[TT8]]
  // CHECK:   "lmhlo.fusion"() ({
  // CHECK-NEXT:     "lmhlo.constant"
  // CHECK-NEXT:     "lmhlo.dynamic_broadcast_in_dim"(%[[T9]], %[[T5:.*]], %[[TTT6]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (memref<f32, "gpu">, memref<2xindex>, memref<?x?xf32, "gpu">) -> ()
  // CHECK-NEXT:     "lmhlo.add"(%[[TTT6]], %[[TTT3]], %[[TTT8]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK-NEXT:     "lmhlo.terminator"() : () -> ()
  // CHECK-NEXT:   })
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   "lmhlo.fusion"() ({
  // CHECK-NEXT:     "lmhlo.constant"
  // CHECK-NEXT:     "lmhlo.dynamic_broadcast_in_dim"(%[[T9]], %[[T5]], %[[T6]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (memref<f32, "gpu">, memref<2xindex>, memref<?x?xf32, "gpu">) -> ()
  // CHECK-NEXT:     "lmhlo.dynamic_broadcast_in_dim"(%[[T3]], %[[T5]], %[[T7]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<?x?xf32, "gpu">, memref<2xindex>, memref<?x?xf32, "gpu">) -> ()
  // CHECK-NEXT:     "lmhlo.add"(%[[T6]], %[[T7]], %[[T8]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK-NEXT:     "lmhlo.terminator"() : () -> ()
  // CHECK-NEXT:   })
  // CHECK-SAME: disc.fusion.name = "test1"
  // CHECK: }
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%9) {value = dense<1.000000e+00> : tensor<f32>} : (memref<f32, "gpu">) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%9, %5, %6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (memref<f32, "gpu">, memref<2xindex>, memref<?x?xf32, "gpu">) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%3, %5, %7) {broadcast_dimensions = dense<[0,1]> : tensor<2xi64>} : (memref<?x?xf32, "gpu">, memref<2xindex>, memref<?x?xf32, "gpu">) -> ()
    "lmhlo.add"(%6, %7, %8) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test1", disc_vectorize_or_tile_hint = 2, disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  %c0_1 = arith.constant 0 : index
  "disc_ral.send_output"(%arg0, %c0_1, %8) : (!disc_ral.context, index, memref<?x?xf32, "gpu">) -> ()
  return
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -9223372036854775808 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}
