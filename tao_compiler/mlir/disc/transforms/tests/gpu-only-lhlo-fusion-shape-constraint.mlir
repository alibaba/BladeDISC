// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=1 DISC_ENABLE_HORIZONTAL_FUSION=1 disc-opt -pass-pipeline='func.func(disc-fusion{gpu-enabled=true fusion-strategy=base})' -split-input-file %s -o - | FileCheck %s --check-prefix=BASE
// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=1 DISC_ENABLE_HORIZONTAL_FUSION=1 disc-opt -pass-pipeline='func.func(disc-fusion{gpu-enabled=true fusion-strategy=stitch})' -split-input-file %s -o - | FileCheck %s --check-prefix=STITCH

// BASE-LABEL: @simple_kloop_fusion
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">) -> memref<?x?xf32, "gpu">
func.func @simple_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> memref<?x?xf32, "gpu"> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %5 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  // BASE: %[[T3:.*]] = memref.reinterpret_cast %[[ARG3]]
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[T1]], %[[T2]], %[[T3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[T3]] : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%2, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%3, %4, %5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %5 : memref<?x?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @simple_multi_output_kloop_fusion
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
func.func @simple_multi_output_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                                       %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %5 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  // BASE: %[[T3:.*]] = memref.reinterpret_cast %[[ARG3]]
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[T1]], %[[T2]], %[[T3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[T1]], %[[T3]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  "lmhlo.abs"(%2, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%3, %4, %5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %3, %5 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @simple_multi_output_kloop_fusion_with_reorder
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?x?xf32, "gpu">, %[[ARG4:.*]]: memref<2xindex, "cpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">)
func.func @simple_multi_output_kloop_fusion_with_reorder(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">,
                          %arg4: memref<2xindex, "cpu">, %arg5:  memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  %2 = memref.dim %arg5, %c0 : memref<?x?xf32, "gpu">
  %3 = memref.dim %arg5, %c1 : memref<?x?xf32, "gpu">

  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  // BASE: %[[T3:.*]] = memref.reinterpret_cast %[[ARG3]]
  // BASE: %[[T5:.*]] = memref.reinterpret_cast %[[ARG5]]
  %4 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %5 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %6 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %7 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %8 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%2, %3], strides: [%3, 1] {kDiscSymbolicDimAttr = [@S2, @S3]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">

  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[T1]], %[[T2]], %[[T3]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: "lmhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[ARG4]], %[[T5]])
  // BASE: return %[[T1]], %[[T3]], %[[T5]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
  "lmhlo.abs"(%4, %5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.dynamic_broadcast_in_dim"(%5, %arg4, %8) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<?x?xf32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%5, %6, %7) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %5, %7, %8 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}

"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @same_num_elements_multi_output_kloop_fusion
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<2xi64>, %[[ARG3:.*]]: memref<?x?x?xf32, "gpu">, %[[ARG4:.*]]: memref<?x?x?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?x?xf32, "gpu">)
func.func @same_num_elements_multi_output_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">,
                          %arg2: memref<2xi64>, %arg3: memref<?x?x?xf32, "gpu">,
                          %arg4: memref<?x?x?xf32, "gpu">, %arg5:  memref<?x?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  %2 = memref.dim %arg3, %c0 : memref<?x?x?xf32, "gpu">
  %3 = memref.dim %arg3, %c1 : memref<?x?x?xf32, "gpu">
  %4 = memref.dim %arg3, %c2 : memref<?x?x?xf32, "gpu">
  %5 = arith.muli %3, %4 : index

  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T3:.*]] = memref.reinterpret_cast %[[ARG3]]
  // BASE: %[[T4:.*]] = memref.reinterpret_cast %[[ARG4]]
  // BASE: %[[T5:.*]] = memref.reinterpret_cast %[[ARG5]]
  %6 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %7 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %8 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%2, %3, %4], strides: [%5, %4, 1] {kDiscSymbolicDimAttr = [@S2, @S3, @S4]} : memref<?x?x?xf32, "gpu"> to memref<?x?x?xf32, "gpu">
  %9 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%2, %3, %4], strides: [%5, %4, 1] {kDiscSymbolicDimAttr = [@S2, @S3, @S4]} : memref<?x?x?xf32, "gpu"> to memref<?x?x?xf32, "gpu">
  %10 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%2, %3, %4], strides: [%5, %4, 1] {kDiscSymbolicDimAttr = [@S2, @S3, @S4]} : memref<?x?x?xf32, "gpu"> to memref<?x?x?xf32, "gpu">

  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.dynamic_reshape"(%[[T1]], %[[ARG2]], %[[T3]])
  // BASE: "lmhlo.add"(%[[T3]], %[[T4]], %[[T5]]) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[T1]], %[[T5]] : memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
  "lmhlo.abs"(%6, %7) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.dynamic_reshape"(%7, %arg2, %8) : (memref<?x?xf32, "gpu">, memref<2xi64>, memref<?x?x?xf32, "gpu">) -> ()
  "lmhlo.add"(%8, %9, %10) : (memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">) -> ()
  return %7, %10 : memref<?x?xf32, "gpu">, memref<?x?x?xf32, "gpu">
}

"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S4", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  %0 = "disc_shape.dim"() {name = @S0} : () -> index
  %1 = "disc_shape.dim"() {name = @S1} : () -> index
  %2 = "disc_shape.dim"() {name = @S2} : () -> index
  %3 = "disc_shape.dim"() {name = @S3} : () -> index
  %4 = "disc_shape.dim"() {name = @S4} : () -> index
  "disc_shape.tie_product_equal"(%0, %1, %2, %3, %4) {operand_segment_sizes = array<i32: 2, 3>} : (index, index, index, index, index) -> ()
  return
}

// -----

// BASE-LABEL: @check_not_kloop_fusion
func.func @check_not_kloop_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  // BASE-NOT: "lmhlo.fusion"
  "lmhlo.add"(%arg0, %arg0, %arg1) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.subtract"(%arg2, %arg2, %arg3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %arg1, %arg3: memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @const_should_not_be_output
func.func @const_should_not_be_output(%arg0: memref<f32, "gpu">) -> (memref<f32, "gpu">, memref<f32, "gpu">) {
  // BASE-NOT: lmhlo.fusion
  %0 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%0) {value = dense<1.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  %1 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.add"(%arg0, %0, %1) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
  return %0, %1 : memref<f32, "gpu">, memref<f32, "gpu">
}

// -----

// BASE-LABEL: @simple_kloop_fusion_static_shape
// BASE-SAME: (%[[ARG0:.*]]: memref<10x11xf32, "gpu">, %[[ARG1:.*]]: memref<10x11xf32, "gpu">, %[[ARG2:.*]]: memref<10x11xf32, "gpu">, %[[ARG3:.*]]: memref<10x11xf32, "gpu">) -> memref<10x11xf32, "gpu">
func.func @simple_kloop_fusion_static_shape(%arg0: memref<10x11xf32, "gpu">, %arg1: memref<10x11xf32, "gpu">,
                                       %arg2: memref<10x11xf32, "gpu">, %arg3: memref<10x11xf32, "gpu">) -> memref<10x11xf32, "gpu"> {
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[ARG0]], %[[ARG1]]) : (memref<10x11xf32, "gpu">, memref<10x11xf32, "gpu">) -> ()
  // BASE: "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[ARG3]]) : (memref<10x11xf32, "gpu">, memref<10x11xf32, "gpu">, memref<10x11xf32, "gpu">) -> ()
  // BASE: })
  // BASE: return %[[ARG3]] : memref<10x11xf32, "gpu">
  "lmhlo.abs"(%arg0, %arg1) : (memref<10x11xf32, "gpu">, memref<10x11xf32, "gpu">) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<10x11xf32, "gpu">, memref<10x11xf32, "gpu">, memref<10x11xf32, "gpu">) -> ()
  return %arg3 : memref<10x11xf32, "gpu">
}

// -----

// BASE-LABEL: @kloop_fusion_with_dealloc
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">)
func.func @kloop_fusion_with_dealloc(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">

  // BASE: %[[T4:.*]] = memref.alloc
  // BASE: %[[T5:.*]] = memref.alloc
  // BASE: %[[T6:.*]] = memref.alloc
  // BASE: %[[T7:.*]] = memref.alloc
  // BASE: %[[T8:.*]] = memref.alloc
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[T0]], %[[T1]], %[[T4]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.multiply"(%[[T0]], %[[T1]], %[[T5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[T4]], %[[T6]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[T5]], %[[T7]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.multiply"(%[[T6]], %[[T7]], %[[T8]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: })
  // BASE: memref.dealloc %[[T4]] : memref<?x?xf32, "gpu">
  // BASE: memref.dealloc %[[T5]] : memref<?x?xf32, "gpu">
  // BASE: memref.dealloc %[[T7]] : memref<?x?xf32, "gpu">
  // BASE: return %[[T6]], %[[T8]] : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">

  %4 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.add"(%2, %3, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %5 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.multiply"(%2, %3, %5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %6 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%4, %6) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %4 : memref<?x?xf32, "gpu">
  %7 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%5, %7) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %5 : memref<?x?xf32, "gpu">
  %8 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.multiply"(%6, %7, %8) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %7 : memref<?x?xf32, "gpu">
  return %6, %8 : memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @simple_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?xf32, "gpu">, %[[ARG3:.*]]: memref<f32, "gpu">) -> memref<?xf32, "gpu">
func.func @simple_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?xf32, "gpu">, %init: memref<f32, "gpu">) -> memref<?xf32, "gpu"> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%1], strides: [1] {kDiscSymbolicDimAttr = [@S1]} : memref<?xf32, "gpu"> to memref<?xf32, "gpu">

  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[T1]], %[[ARG3]], %[[T2]]) ({
  // BASE: })
  // BASE: return %[[T2]] : memref<?xf32, "gpu">
  "lmhlo.abs"(%2, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%3, %init, %4) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %4: memref<?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @multi_outputs_kinput
// BASE-SAME: (%[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?xf32, "gpu">, %[[ARG3:.*]]: memref<f32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">)
func.func @multi_outputs_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%1], strides: [1] {kDiscSymbolicDimAttr = [@S1]} : memref<?xf32, "gpu"> to memref<?xf32, "gpu">

  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[T1]], %[[ARG3]], %[[T2]]) ({
  // BASE: })
  // BASE: return %[[T1]], %[[T2]] : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.abs"(%2, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%3, %init, %4) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %3, %4: memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @row_red_and_row_red_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?xf32, "gpu">, %[[ARG4:.*]]: memref<?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">, %[[ARG6:.*]]: memref<f32, "gpu">
func.func @row_red_and_row_red_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?xf32, "gpu">, %arg4: memref<?xf32, "gpu">, %arg5: memref<?x?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  // BASE: %[[T3:.*]] = memref.reinterpret_cast %[[ARG3]]
  // BASE: %[[T4:.*]] = memref.reinterpret_cast %[[ARG4]]
  // BASE: %[[T5:.*]] = memref.reinterpret_cast %[[ARG5]]
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %5 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%1], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
  %6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%1], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
  %7 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">

  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[T0]], %[[T1]], %[[T2]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[T2]], %[[T5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[T5]], %[[ARG6]], %[[T3]]) ({
  // BASE: "lmhlo.reduce"(%[[T2]], %[[ARG6]], %[[T4]]) ({
  // BASE: })
  // BASE: return %[[T3]], %[[T4]] : memref<?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.add"(%2, %3, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%4, %7) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%7, %init, %5) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%4, %init, %6) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %5, %6: memref<?xf32, "gpu">, memref<?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @row_red_and_col_red_kinput
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">, %[[ARG2:.*]]: memref<?x?xf32, "gpu">, %[[ARG3:.*]]: memref<?xf32, "gpu">, %[[ARG4:.*]]: memref<?xf32, "gpu">, %[[ARG5:.*]]: memref<?x?xf32, "gpu">, %[[ARG6:.*]]: memref<f32, "gpu">
func.func @row_red_and_col_red_kinput(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">, %arg2: memref<?x?xf32, "gpu">, %arg3: memref<?xf32, "gpu">, %arg4: memref<?xf32, "gpu">, %arg5: memref<?x?xf32, "gpu">, %init: memref<f32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  // BASE: %[[T2:.*]] = memref.reinterpret_cast %[[ARG2]]
  // BASE: %[[T3:.*]] = memref.reinterpret_cast %[[ARG3]]
  // BASE: %[[T4:.*]] = memref.reinterpret_cast %[[ARG4]]
  // BASE: %[[T5:.*]] = memref.reinterpret_cast %[[ARG5]]
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %5 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%1], strides: [1] {kDiscSymbolicDimAttr = [@S0]} : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
  %6 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%1], strides: [1] {kDiscSymbolicDimAttr = [@S1]} : memref<?xf32, "gpu"> to memref<?xf32, "gpu">
  %7 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">

  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[T0]], %[[T1]], %[[T2]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.abs"(%[[T2]], %[[T5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[T5]], %[[ARG6]], %[[T3]]) ({
  // BASE: "lmhlo.reduce"(%[[T2]], %[[ARG6]], %[[T4]]) ({
  // BASE: })
  // BASE: return %[[T3]], %[[T4]] : memref<?xf32, "gpu">, memref<?xf32, "gpu">
  "lmhlo.add"(%2, %3, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%4, %7) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%7, %init, %5) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[1]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  "lmhlo.reduce"(%4, %init, %6) ({
  ^bb0(%targ1: memref<f32, "gpu">, %targ2: memref<f32, "gpu">, %tresult: memref<f32, "gpu">):
    "lmhlo.add"(%targ1, %targ2, %tresult) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  return %5, %6: memref<?xf32, "gpu">, memref<?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// BASE-LABEL: @reduce_should_not_have_consumer_in_the_fusion
// BASE-SAME: %[[ARG0:.*]]: memref<?x?xf32, "gpu">, %[[ARG1:.*]]: memref<?x?xf32, "gpu">
func.func @reduce_should_not_have_consumer_in_the_fusion(%arg0: memref<?x?xf32, "gpu">, %arg1: memref<?x?xf32, "gpu">)
-> (memref<?x?xf32, "gpu">, memref<?xf32, "gpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  // BASE: %[[T0:.*]] = memref.reinterpret_cast %[[ARG0]]
  // BASE: %[[T1:.*]] = memref.reinterpret_cast %[[ARG1]]
  %2 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0, %1], strides: [%1, 1] {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">

  // BASE: %[[T4:.*]] = memref.alloc
  // BASE: %[[T5:.*]] = memref.alloc
  // BASE: %[[T6:.*]] = memref.alloc
  // BASE: %[[T7:.*]] = memref.alloc
  // BASE: "lmhlo.fusion"() ({
  // BASE: "lmhlo.add"(%[[T0]], %[[T1]], %[[T4]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.subtract"(%[[T0]], %[[T4]], %[[T5]]) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // BASE: "lmhlo.constant"(%[[T6]]) {value = dense<0.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  // BASE: "lmhlo.reduce"(%[[T5]], %[[T6]], %[[T7]]) ({
  // BASE: })
  // BASE: memref.dealloc %[[T4]] : memref<?x?xf32, "gpu">
  // BASE: memref.dealloc %[[T6]] : memref<f32, "gpu">
  // BASE: %[[T8:.*]] = memref.alloc
  // BASE: "lmhlo.add"(%[[T7]], %[[T7]], %[[T8]]) : (memref<?xf32, "gpu">, memref<?xf32, "gpu">, memref<?xf32, "gpu">) -> ()
  // BASE: memref.dealloc %[[T7]] : memref<?xf32, "gpu">
  // BASE: return %[[T5]], %[[T8]] : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
  %4 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.add"(%2, %3, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %5 = memref.alloc(%0, %1) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.subtract"(%2, %4, %5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  memref.dealloc %4 : memref<?x?xf32, "gpu">
  %6 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%6) {value = dense<0.000000e+00> : tensor<f32, "gpu">} : (memref<f32, "gpu">) -> ()
  %7 = memref.alloc(%0) : memref<?xf32, "gpu">
  "lmhlo.reduce"(%5, %6, %7) ({
  ^bb0(%arg2: memref<f32, "gpu">, %arg3: memref<f32, "gpu">, %arg4: memref<f32, "gpu">):  // no predecessors
    "lmhlo.add"(%arg2, %arg3, %arg4) : (memref<f32, "gpu">, memref<f32, "gpu">, memref<f32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  memref.dealloc %6 : memref<f32, "gpu">
  %8 = memref.alloc(%0) : memref<?xf32, "gpu">
  "lmhlo.add"(%7, %7, %8) : (memref<?xf32, "gpu">, memref<?xf32, "gpu">, memref<?xf32, "gpu">) -> ()
  memref.dealloc %7 : memref<?xf32, "gpu">
  return %5, %8 : memref<?x?xf32, "gpu">, memref<?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  return
}

// -----

// STITCH-LABEL: @kstitch_fusion_mean
func.func @kstitch_fusion_mean(%arg0: memref<?x?x?xf32, "gpu">) -> memref<?x?xf32, "gpu"> attributes {tf.entry_function = {input_placements = "gpu", inputs = "input0", output_placements = "gpu", outputs = "output0"}} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "gpu">) -> ()
  %1 = memref.dim %arg0, %c0 : memref<?x?x?xf32, "gpu">
  %2 = memref.dim %arg0, %c1 : memref<?x?x?xf32, "gpu">
  %3 = memref.dim %arg0, %c2 : memref<?x?x?xf32, "gpu">
  %4 = arith.muli %2, %3 : index
  %5 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%1, %2, %3], strides: [%4, %3, 1] {kDiscSymbolicDimAttr = [@S0, @S1, @S2]} : memref<?x?x?xf32, "gpu"> to memref<?x?x?xf32, "gpu">
  %6 = memref.alloca() : memref<2xindex, "cpu">
  %7 = arith.muli %1, %2 : index
  memref.store %7, %6[%c0] : memref<2xindex, "cpu">
  memref.store %3, %6[%c1] : memref<2xindex, "cpu">
  %8 = memref.alloc(%7, %3) {kDiscSymbolicDimAttr = [@S3, @S2]} : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_reshape"(%5, %6, %8) {disc.device = "gpu"} : (memref<?x?x?xf32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %9 = memref.alloc(%7) {kDiscSymbolicDimAttr = [@S0]}  : memref<?xf32, "gpu">
  "lmhlo.reduce"(%8, %0, %9) ({
  ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>):  // no predecessors
    "lmhlo.add"(%arg1, %arg2, %arg3) {disc.device = "gpu"} : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, disc.device = "gpu"} : (memref<?x?xf32, "gpu">, memref<f32, "gpu">, memref<?xf32, "gpu">) -> ()
  %10 = memref.alloca() : memref<2xindex, "cpu">
  memref.store %1, %10[%c0] : memref<2xindex, "cpu">
  memref.store %2, %10[%c1] : memref<2xindex, "cpu">
  %11 = memref.alloc(%1, %2) : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_reshape"(%9, %10, %11) {disc.device = "gpu"} : (memref<?xf32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %12 = memref.alloca() : memref<1xi64, "cpu">
  %i64_3 = arith.index_cast %3 : index to i64
  memref.store %i64_3, %12[%c0] : memref<1xi64, "cpu">
  %13 = memref.alloc() : memref<1xi64, "gpu">
  "lmhlo_disc.h2d"(%12, %13) : (memref<1xi64, "cpu">, memref<1xi64, "gpu">) -> ()
  %14 = memref.alloc() : memref<i64, "gpu">
  "lmhlo.reshape"(%13, %14) {disc.device = "gpu"} : (memref<1xi64, "gpu">, memref<i64, "gpu">) -> ()
  %15 = memref.alloc() : memref<f32, "gpu">
  "lmhlo.convert"(%14, %15) {disc.device = "gpu"} : (memref<i64, "gpu">, memref<f32, "gpu">) -> ()
  %16 = memref.alloc(%1, %2) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_broadcast_in_dim"(%15, %10, %16) {broadcast_dimensions = dense<> : tensor<0xi64>, disc.device = "gpu"} : (memref<f32, "gpu">, memref<2xindex, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %17 = memref.alloc(%1, %2) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.divide"(%11, %16, %17) {disc.device = "gpu"} : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // Make sure there is one and only one kStitch fusion.
  // STITCH:      disc.fusion.name
  // STITCH-SAME: disc.fusion_type = "kStitch"
  // STITCH-NOT:  disc.fusion.name
  return %17 : memref<?x?xf32, "gpu">
}
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -1 : i64} : () -> ()
"disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -1 : i64} : () -> ()
func.func @shape_constraint_graph() {
  %0 = "disc_shape.dim"() {name = @S0} : () -> index
  %1 = "disc_shape.dim"() {name = @S1} : () -> index
  %2 = "disc_shape.dim"() {name = @S2} : () -> index
  %3 = "disc_shape.dim"() {name = @S3} : () -> index
  "disc_shape.tie_product_equal"(%3, %0, %1) {operand_segment_sizes = array<i32: 1, 2>} : (index, index, index) -> ()
  return
}

