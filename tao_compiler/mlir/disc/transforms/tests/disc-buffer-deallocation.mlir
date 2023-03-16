// RUN: disc-opt --disc-buffer-deallocation -split-input-file %s | FileCheck %s

func.func @main(%arg0: memref<?x?xf32, "cpu">) -> memref<?x71xf32, "cpu"> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0,input1", output_placements = "cpu", outputs = "output0"}} {
  %c71 = arith.constant 71 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c1 : memref<?x?xf32, "cpu">
  %dim_0 = memref.dim %arg0, %c0 : memref<?x?xf32, "cpu">
  %alloca = memref.alloca() : memref<i32, "cpu">
  "lmhlo.constant"(%alloca) {disc.device = "cpu", disc.shape_op = true, value = dense<0> : tensor<i32>} : (memref<i32, "cpu">) -> ()
  %alloca_1 = memref.alloca() : memref<f32, "cpu">
  "lmhlo.constant"(%alloca_1) {disc.device = "cpu", disc.shape_op = true, value = dense<0.00833333284> : tensor<f32>} : (memref<f32, "cpu">) -> ()
  %alloc = memref.alloc() : memref<25x71xf32, "cpu">
  "lmhlo.constant"(%alloc) {disc.device = "cpu", value = dense<-8.000000e-01> : tensor<25x71xf32>} : (memref<25x71xf32, "cpu">) -> ()
  %alloc_2 = memref.alloc() : memref<71xf32, "cpu">
  "lmhlo.constant"(%alloc_2) {disc.device = "cpu", disc.shape_op = true, value = dense<0.0156862698> : tensor<71xf32>} : (memref<71xf32, "cpu">) -> ()
  %alloc_3 = memref.alloc() : memref<71xi32, "cpu">
  "lmhlo.constant"(%alloc_3) {disc.device = "cpu", disc.shape_op = true, value = dense<0> : tensor<71xi32>} : (memref<71xi32, "cpu">) -> ()
  %alloca_4 = memref.alloca() : memref<f32, "cpu">
  "lmhlo.constant"(%alloca_4) {disc.device = "cpu", disc.shape_op = true, value = dense<1.000000e+00> : tensor<f32>} : (memref<f32, "cpu">) -> ()
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%dim_0, %dim], strides: [%dim, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "cpu"> to memref<?x?xf32, "cpu">
  %0 = "lmhlo_disc.custom_call_v2"(%reinterpret_cast, %alloc, %alloca_1, %alloca, %alloc_2, %alloc_3, %alloca_4, %alloca) {call_target_name = "ral_pdll_qgemm", custom_attrs = {transpose_a = false, transpose_b = false}, device = "h", disc.device = "cpu", expected_input_layouts = "", expected_output_layouts = "", has_side_effect = false, input_layouts = "", input_placements = "h,h,s,s,s,s,s,s", output_layouts = "", output_placements = "h"} : (memref<?x?xf32, "cpu">, memref<25x71xf32, "cpu">, memref<f32, "cpu">, memref<i32, "cpu">, memref<71xf32, "cpu">, memref<71xi32, "cpu">, memref<f32, "cpu">, memref<i32, "cpu">) -> memref<?x71xi8, "cpu">
  memref.dealloc %alloc_3 : memref<71xi32, "cpu">
  memref.dealloc %alloc_2 : memref<71xf32, "cpu">
  memref.dealloc %alloc : memref<25x71xf32, "cpu">
  %dim_5 = memref.dim %0, %c0 : memref<?x71xi8, "cpu">
  %reinterpret_cast_6 = memref.reinterpret_cast %0 to offset: [0], sizes: [%dim_5, 71], strides: [71, 1] {kDiscSymbolicDimAttr = [@S2, @C71]} : memref<?x71xi8, "cpu"> to memref<?x71xi8, "cpu">
  %alloca_7 = memref.alloca() : memref<2xindex, "cpu">
  memref.store %dim_5, %alloca_7[%c0] : memref<2xindex, "cpu">
  memref.store %c71, %alloca_7[%c1] : memref<2xindex, "cpu">
  %alloc_8 = memref.alloc(%dim_5) {kDiscSymbolicDimAttr = [@S2, @C71]} : memref<?x71xi32, "cpu">
  %alloc_9 = memref.alloc(%dim_5) {kDiscSymbolicDimAttr = [@S2, @C71]} : memref<?x71xi32, "cpu">
  %alloc_10 = memref.alloc(%dim_5) {kDiscSymbolicDimAttr = [@S2, @C71]} : memref<?x71xi32, "cpu">
  %alloc_11 = memref.alloc(%dim_5) {kDiscSymbolicDimAttr = [@S2, @C71]} : memref<?x71xf32, "cpu">
  "lmhlo.fusion"() ({
    "lmhlo.dynamic_broadcast_in_dim"(%alloca, %alloca_7, %alloc_8) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "cpu"} : (memref<i32, "cpu">, memref<2xindex, "cpu">, memref<?x71xi32, "cpu">) -> ()
    "lmhlo.convert"(%reinterpret_cast_6, %alloc_9) {disc.device = "cpu"} : (memref<?x71xi8, "cpu">, memref<?x71xi32, "cpu">) -> ()
    "lmhlo.subtract"(%alloc_9, %alloc_8, %alloc_10) {disc.device = "cpu"} : (memref<?x71xi32, "cpu">, memref<?x71xi32, "cpu">, memref<?x71xi32, "cpu">) -> ()
    "lmhlo.convert"(%alloc_10, %alloc_11) {disc.device = "cpu"} : (memref<?x71xi32, "cpu">, memref<?x71xf32, "cpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "cpu", disc.fusion.name = "main_kLoop_convert__4_1_0", disc.fusion_type = "kLoop"} : () -> ()
  memref.dealloc %alloc_10 : memref<?x71xi32, "cpu">
  memref.dealloc %alloc_9 : memref<?x71xi32, "cpu">
  memref.dealloc %alloc_8 : memref<?x71xi32, "cpu">
  // CHECK: %[[T0:.*]] = "lmhlo_disc.custom_call_v2"
  // CHECK: %[[T1:.*]] = memref.reinterpret_cast %[[T0]]
  // CHECK: "lmhlo.fusion"()  ({
  // CHECK: }) {disc.device = "cpu"
  // CHECK: memref.dealloc %[[T0]]
  return %alloc_11 : memref<?x71xf32, "cpu">
}

"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = true, sym_name = "C71", value = 71 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}
