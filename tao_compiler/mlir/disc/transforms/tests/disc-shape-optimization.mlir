// RUN: disc-opt --disc-shape-optimization -split-input-file %s | FileCheck %s

// Test tensor.cast op

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>) -> tensor<10xf32>
func.func @main(%arg0 : tensor<?xf32>) -> tensor<10xf32> {
  // CHECK: return %[[ARG0]] : tensor<10xf32>
  %0 = tensor.cast %arg0 : tensor<?xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

// Test mhlo.add op

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<10xf32>) -> tensor<10xf32>
func.func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<10xf32>) -> tensor<?xf32> {
  %0 = tensor.cast %arg1 : tensor<10xf32> to tensor<?xf32>
  // CHECK: %[[T1:.*]] = mhlo.add %[[ARG0]], %[[ARG1]] : tensor<10xf32>
  %1 = "mhlo.add"(%arg0, %0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: return %[[T1]] : tensor<10xf32>
  return %1 : tensor<?xf32>
}

// -----

// Test mhlo.concat op
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x10xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: tensor<?x10xf32, [@[[S0]], @[[S1]]]>) -> tensor<?x20xf32, [@[[S0]], @[[S2:.*]]]>
func.func @main(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.concatenate"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x20xf32, [@[[S0]], @[[S2]]]>
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 1 : i64 } : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Test mhlo.dot_general
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]], @[[S3:.*]]]>, %[[ARG1:.*]]: tensor<?x?x?x?xf32, [@[[S0]], @[[S1]], @[[S3]], @[[S4:.*]]]>) -> tensor<?x?x?x?xf32, [@[[S0]], @[[S1]], @[[S2]], @[[S4]]]>
func.func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x?x?x?xf32, [@[[S0]], @[[S1]], @[[S2]], @[[S4]]]>
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// Test mhlo.dot
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: tensor<?x?xf32, [@[[S1]], @[[S2:.*]]]>) -> tensor<?x?xf32, [@[[S0]], @[[S2]]]>
func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = "mhlo.dot"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x?xf32, [@[[S0]], @[[S2]]]>
  %1 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

// -----

// Test mhlo.clamp: zero rank min/max

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32, [@[[S0:.*]]]>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = mhlo.clamp %[[ARG1]], %[[ARG0]], %[[ARG2]]
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg2) : (tensor<f32>, tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.clamp: same shape

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32, [@[[S0:.*]]]>, %[[ARG1:.*]]: tensor<?xf32, [@[[S0]]]>, %[[ARG2:.*]]: tensor<?xf32, [@[[S0]]]>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = mhlo.clamp %[[ARG1]], %[[ARG0]], %[[ARG2]]
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.select : zero rank
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i1>, %[[ARG1:.*]]: tensor<?xf32, [@[[S0:.*]]]>, %[[ARG2:.*]]: tensor<?xf32, [@[[S0]]]>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0: tensor<i1>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = mhlo.select
  // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]]
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.select"(%arg0, %arg1, %arg2)  : (tensor<i1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.select : same shape
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xi1, [@[[S0:.*]]]>, %[[ARG1:.*]]: tensor<?xf32, [@[[S0]]]>, %[[ARG2:.*]]: tensor<?xf32, [@[[S0]]]>) -> tensor<?xf32, [@[[S0]]]>
func.func @main(%arg0: tensor<?xi1>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[T0:.*]] = mhlo.select
  // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]]
  // CHECK: return %[[T0]] : tensor<?xf32, [@[[S0]]]>
  %0 = "mhlo.select"(%arg0, %arg1, %arg2)  : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// Test mhlo.einsum
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]]]>, %[[ARG1:.*]]: tensor<?x?x?xf32, [@[[S0]], @[[S2]], @[[S3:.*]]]>) -> tensor<?x?x?xf32, [@[[S0]], @[[S1]], @[[S3]]]>
func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "input0,input1", output_placements = "cpu", outputs = "output0"}} {
  // CHECK: %[[T0:.*]] = "mhlo.einsum"(%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[T0]] : tensor<?x?x?xf32, [@[[S0]], @[[S1]], @[[S3]]]>
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm"} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// Test arith.cmpi
// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]]]>, %[[ARG1:.*]]: tensor<?x?xf32, [@[[S3:.*]], @[[S4:.*]]]>) -> tensor<?x?xf32, [@[[S3]], @[[S4]]]>
func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c-1 = arith.constant -1 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[T0:.*]] = tensor.dim %[[ARG1]], %c0
  // CHECK: %[[T1:.*]] = tensor.dim %[[ARG1]], %c1
  // CHECK: %[[T2:.*]] = tensor.from_elements %[[T0]], %[[T1]] : tensor<2xindex>
  // CHECK: %[[T3:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[T2]]
  // CHECK: return %[[T3]] : tensor<?x?xf32, [@[[S3]], @[[S4]]]>
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = arith.muli %1, %0 : index
  %4 = arith.muli %2, %3 : index
  %5 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %6 = arith.muli %5, %c-1 : index
  %7 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %8 = arith.muli %7, %6 : index
  %9 = arith.cmpi eq, %4, %c0 : index
  %10 = arith.cmpi slt, %8, %c0 : index
  %11 = arith.ori %9, %10 : i1
  %12 = arith.select %11, %c1, %8 : index
  %13 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %14 = arith.cmpi eq, %13, %c-1 : index
  %15 = arith.divui %4, %12 : index
  %16 = arith.select %14, %15, %13 : index
  %17 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %18 = arith.cmpi eq, %17, %c-1 : index
  %19 = arith.divui %4, %12 : index
  %20 = arith.select %18, %19, %17 : index
  %21 = tensor.from_elements %16, %20 : tensor<2xindex>
  %22 = "mhlo.dynamic_reshape"(%arg0, %21) : (tensor<?x?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %22 : tensor<?x?xf32>
}

// -----

// Test arith.cmpi + arith::muli + arith::add
// CHECK-LABEL: main
func.func @main(%arg0: tensor<?x?x?xf32>) -> (index, index) {
  // CHECK: %c0 = arith.constant 0 : index
  // CHECK: return %c0, %c0 : index
  %c-1 = arith.constant -1 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = arith.muli %0, %1 : index
  %4 = arith.addi %3, %2 : index
  %5 = arith.cmpi eq, %4, %c-1 : index
  %6 = arith.select %5, %c1, %c0 : index
  %7 = arith.addi %4, %c1 : index
  %8 = arith.cmpi eq, %7, %c0 : index
  %9 = arith.select %8, %c1, %c0 : index
  return %6, %9 : index, index
}

// -----

// Test arith.cmpi + arith::trunci
// CHECK-LABEL: main
func.func @main(%arg0: tensor<?x?x?xf32>) -> (index) {
  // CHECK: %c0 = arith.constant 0 : index
  // CHECK: return %c0 : index
  %c-1 = arith.constant -1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = arith.index_cast %0 : index to i64
  %2 = arith.trunci %1 : i64 to i32
  %3 = arith.cmpi eq, %2, %c-1 : i32
  %4 = arith.select %3, %c1, %c0 : index
  return %4 : index
}

// -----

// Test arith.cmpi + tie_shape
// CHECK-LABEL: main
func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<2xindex>) -> (tensor<?x?xf32>, index) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: return %[[RET0:.*]], %[[C0]]
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = tensor.extract %arg1[%c0] : tensor<2xindex>
  %2 = arith.cmpi eq, %1, %c-1 : index
  %3 = arith.select %2, %c1, %c0 : index
  return %0, %3 : tensor<?x?xf32>, index
}

// -----

// Test disc_shape.tie_product_equal op case 1
// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: tensor<1xindex>, %[[ARG2:.*]]: tensor<3xindex>) -> tensor<?x?x?xf32, [@[[S3:.*]], @[[S4:.*]], @[[S5:.*]]]>
func.func @main(%arg0 : tensor<?x?xf32>, %arg1 : tensor<1xindex>, %arg2 : tensor<3xindex>) -> tensor<?x?x?xf32> {
   // CHECK-NEXT: %[[T0:.*]] = mhlo.dynamic_reshape %[[ARG0]], %[[ARG1]]
   // CHECK-SAME: tensor<?xf32, [@[[S2:.*]]]>
   %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<1xindex>) -> tensor<?xf32>
   // CHECK-NEXT: %[[T1:.*]] = mhlo.abs %[[T0]] : tensor<?xf32, [@[[S2]]]>
   %1 = "mhlo.abs"(%0) : (tensor<?xf32>) -> tensor<?xf32>
   // CHECK-NEXT: %[[T2:.*]] = mhlo.dynamic_reshape %[[T1]], %[[ARG2]]
   // CHECK-SAME: tensor<?x?x?xf32, [@[[S3]], @[[S4]], @[[S5]]]>
   %2 = "mhlo.dynamic_reshape"(%1, %arg2) : (tensor<?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
   // CHECK-NEXT: return %[[T2]] : tensor<?x?x?xf32, [@[[S3]], @[[S4]], @[[S5]]]>
   return %2 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @shape_constraint_graph
// CHECK-DAG: %[[TT0:.*]] = "disc_shape.dim"() {name = @[[S0]]} : () -> index
// CHECK-DAG: %[[TT1:.*]] = "disc_shape.dim"() {name = @[[S1]]} : () -> index
// CHECK-DAG: %[[TT2:.*]] = "disc_shape.dim"() {name = @[[S2]]} : () -> index
// CHECK-DAG: %[[TT3:.*]] = "disc_shape.dim"() {name = @[[S3]]} : () -> index
// CHECK-DAG: %[[TT4:.*]] = "disc_shape.dim"() {name = @[[S4]]} : () -> index
// CHECK-DAG: %[[TT5:.*]] = "disc_shape.dim"() {name = @[[S5]]} : () -> index
// CHECK-DAG: "disc_shape.tie_product_equal"(%[[TT2]], %[[TT3]], %[[TT4]], %[[TT5]])
// CHECK-DAG: "disc_shape.tie_product_equal"(%[[TT2]], %[[TT0]], %[[TT1]])
// CHECK-DAG: "disc_shape.tie_product_equal"(%[[TT0]], %[[TT1]], %[[TT3]], %[[TT4]], %[[TT5]])

// -----

// Test disc_shape.tie_product_equal op case 2

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]]]>) -> tensor<?x?xf32, [@[[S1]], @[[S2]]]>
func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  %1 = mhlo.abs %arg0 : tensor<?x?x?xf32>
  %2 = tensor.dim %1, %c0 : tensor<?x?x?xf32>
  %3 = tensor.dim %1, %c1 : tensor<?x?x?xf32>
  %4 = tensor.dim %1, %c2 : tensor<?x?x?xf32>
  %5 = arith.muli %3, %4 : index
  %6 = tensor.from_elements %2, %5 : tensor<2xindex>
  %7 = "mhlo.dynamic_reshape"(%1, %6) : (tensor<?x?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %8 = mhlo.reduce(%7 init: %0) applies mhlo.add across dimensions = [0] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  %9 = tensor.dim %1, %c1 : tensor<?x?x?xf32>
  %10 = tensor.dim %1, %c2 : tensor<?x?x?xf32>
  %11 = tensor.from_elements %9, %10 : tensor<2xindex>
  %12 = "mhlo.dynamic_reshape"(%8, %11) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %12 : tensor<?x?xf32>
}

// CHECK-LABEL: @shape_constraint_graph
// CHECK-DAG: %[[TT1:.*]] = "disc_shape.dim"() {name = @[[S1]]} : () -> index
// CHECK-DAG: %[[TT2:.*]] = "disc_shape.dim"() {name = @[[S2]]} : () -> index
// CHECK-DAG: %[[TT3:.*]] = "disc_shape.dim"() {name = @[[S3]]} : () -> index
// CHECK-DAG: "disc_shape.tie_product_equal"(%[[TT3]], %[[TT1]], %[[TT2]])

// -----

// Regression test: disc_shape.tie_product_equal

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32, [@[[S0:.*]], @[[S1:.*]], @[[S2:.*]]]>) -> tensor<?x?xf32, [@[[S3:.*]], @[[S4:.*]]]>
func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %2 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %3 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %4 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %5 = arith.addi %3, %4 : index
  %6 = arith.addi %2, %3 : index
  %7 = tensor.from_elements %5, %6 : tensor<2xindex>
  %8 = "mhlo.dynamic_reshape"(%arg0, %7) : (tensor<?x?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %8 : tensor<?x?xf32>
}

// CHECK-LABEL: @shape_constraint_graph
// CHECK-DAG: %[[TT0:.*]] = "disc_shape.dim"() {name = @[[S0]]} : () -> index
// CHECK-DAG: %[[TT1:.*]] = "disc_shape.dim"() {name = @[[S1]]} : () -> index
// CHECK-DAG: %[[TT2:.*]] = "disc_shape.dim"() {name = @[[S2]]} : () -> index
// CHECK-DAG: %[[TT3:.*]] = "disc_shape.dim"() {name = @[[S3]]} : () -> index
// CHECK-DAG: %[[TT4:.*]] = "disc_shape.dim"() {name = @[[S4]]} : () -> index
// CHECK-DAG: "disc_shape.tie_product_equal"(%[[TT3]], %[[TT4]], %[[TT0]], %[[TT1]], %[[TT2]])

// -----

// regression test: non-shape-tensor from_element op

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> (tensor<i32>, tensor<2x1xi32>
func.func @main(%arg0 : i32) -> (tensor<i32>, tensor<2x1xi32>) {
  // CHECK: %[[T0:.*]] = tensor.from_elements %[[ARG0]] : tensor<i32>
  // CHECK: %[[T1:.*]] = tensor.from_elements %[[ARG0]], %[[ARG0]] : tensor<2x1xi32>
  // CHECK: return %[[T0]], %[[T1]]
  %0 = tensor.from_elements %arg0 : tensor<i32>
  %1 = tensor.from_elements %arg0, %arg0 : tensor<2x1xi32>
  return %0, %1 : tensor<i32>, tensor<2x1xi32>
}

// -----

// test shape constraint attr attached in op.
// test symbols dim ops 2/3 are not removed (used in kDiscSymbolicDimAttr)

module {
  // CHECK-LABEL: @main
  func.func @main(%arg0: tensor<?x?xf32, [@S0, @S1]>, %arg1: tensor<?x?xf32, [@S0, @S1]>) -> tensor<?x?xf32, [@S0, @S1]> {
    %0 = mhlo.add %arg0, %arg1 {kDiscSymbolicDimAttr = [@S2, @S3]} : tensor<?x?xf32, [@S0, @S1]>
    return %0 : tensor<?x?xf32, [@S0, @S1]>
  }
  // Test symbols dim ops 2/3 are not removed (used in kDiscSymbolicDimAttr)
  // CHECK-DAG: "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64}
  // CHECK-DAG: "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64}
  // CHECK-DAG: "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64}
  // CHECK-DAG: "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -9223372036854775808 : i64}
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -9223372036854775808 : i64} : () -> ()
  func.func @shape_constraint_graph() {
    return
  }
}

// -----

// test shape constraint attr attached in op.
// test symbols dim ops 2/3 are removed.

module {
  // CHECK-LABEL: @main
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: tensor<2xindex>) -> tensor<?x?xf32, [@[[S0]], @[[S1]]]>
  func.func @main(%arg0: tensor<?x?xf32, [@S0, @S1]>, %arg1: tensor<2xindex>) -> tensor<?x?xf32, [@S0, @S1]> {
    // CHECK:  "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: kDiscSymbolicDimAttr = [@[[S0]], @[[S1]]]
    %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {kDiscSymbolicDimAttr = [@S2, @S3], broadcast_dimensions = dense<[0,1]> : tensor<2xi64>} : (tensor<?x?xf32, [@S0, @S1]>, tensor<2xindex>) -> tensor<?x?xf32, [@S2, @S3]>
    // CHECK-NOT: tensor.cast
    %1 = tensor.cast %0 : tensor<?x?xf32, [@S2, @S3]> to tensor<?x?xf32, [@S0, @S1]>
    return %1 : tensor<?x?xf32, [@S0, @S1]>
  }
  // CHECK-DAG: "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64}
  // CHECK-DAG: "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64}
  // CHECK-NOT: "disc_shape.SymbolicDim"()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S3", value = -9223372036854775808 : i64} : () -> ()
  func.func @shape_constraint_graph() {
    return
  }
}

// -----

// test: scalarize mhlo.concat whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) -> (i32, i32)
func.func @main(%arg0 : i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
  // CHECK-NOT: mhlo.concatenate
  // CHECK: return %[[ARG1]], %[[ARG2]] : i32, i32
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = tensor.from_elements %arg1 : tensor<1xi32>
  %2 = tensor.from_elements %arg2 : tensor<1xi32>
  %3 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %4 = "mhlo.concatenate"(%3, %2) { dimension = 0 : i64 } : (tensor<2xi32>, tensor<1xi32>) -> tensor<3xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %5 = tensor.extract %4[%c1] : tensor<3xi32>
  %6 = tensor.extract %4[%c2] : tensor<3xi32>
  return %5, %6 : i32, i32
}

// -----

// test: scalarize mhlo.slice whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) -> i32
func.func @main(%arg0 : i32, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK-NOT: mhlo.slice
  // CHECK: return %[[ARG2]] : i32
  %0 = tensor.from_elements %arg0, %arg1, %arg2 : tensor<3xi32>
  %1 = "mhlo.slice"(%0) {
    start_indices = dense<[1]> : tensor<1xi64>,
    limit_indices = dense<[3]> : tensor<1xi64>,
    strides = dense<[1]> : tensor<1xi64>
  } : (tensor<3xi32>) -> (tensor<2xi32>)
  %c1 = arith.constant 1 : index
  %2 = tensor.extract %1[%c1] : tensor<2xi32>
  return %2 : i32
}

// -----

// test: scalarize mhlo.reshape whenever possible
// test 1: <i32> -> <1xi32>

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
func.func @main(%arg0 : i32) -> i32 {
  // CHECK-NOT: "mhlo.reshape
  // CHECK: return %[[ARG0]] : i32
  %0 = tensor.from_elements %arg0 : tensor<i32>
  %1 = "mhlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  %c0 = arith.constant 0 : index
  %2 = tensor.extract %1[%c0] : tensor<1xi32>
  return %2 : i32
}

// -----

// test: scalarize mhlo.reshape whenever possible
// test 2: <1xi32> -> <i32>

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
func.func @main(%arg0 : i32) -> i32 {
  // CHECK-NOT: "mhlo.reshape
  // CHECK: return %[[ARG0]] : i32
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<1xi32>) -> tensor<i32>
  %2 = tensor.extract %1[] : tensor<i32>
  return %2 : i32
}

// -----

// test: scalarize mhlo.reshape whenever possible
// test 3: <1x2xi32> -> <2x1x1xi32>

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @main(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NOT: "mhlo.reshape
  // CHECK: return %[[ARG1]] : i32
  %0 = tensor.from_elements %arg0, %arg1 : tensor<1x2xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<1x2xi32>) -> tensor<2x1x1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = tensor.extract %1[%c1, %c0, %c0] : tensor<2x1x1xi32>
  return %2 : i32
}

// -----

// test: scalarize arith.constant whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: () -> i32
func.func @main() -> i32 {
  // CHECK-NOT: mhlo.reshape
  // CHECK: %[[T0:.*]] = arith.constant 2 : i32
  // CHECK: return %[[T0]]
  %0 = "arith.constant"() {value = dense<[1,2,3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %c1 = arith.constant 1 : index
  %1 = tensor.extract %0[%c1] : tensor<3xi32>
  return %1 : i32
}

// -----

// test: scalarize mhlo.const whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: () -> i32
func.func @main() -> i32 {
  // CHECK-NOT: mhlo.reshape
  // CHECK: %[[T0:.*]] = arith.constant 2 : i32
  // CHECK: return %[[T0]]
  %0 = "mhlo.constant"() {value = dense<[1,2,3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<3xi32>) -> tensor<1x3x1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %2 = tensor.extract %1[%c0, %c1, %c0] : tensor<1x3x1xi32>
  return %2 : i32
}

// -----

// test: scalarize arith.index_cast whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @main(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NOT: arith.index_cast
  // CHECK: return %[[ARG1]] : i32
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xi32>
  %1 = arith.index_cast %0 : tensor<2xi32> to tensor<2xindex>
  %c1 = arith.constant 1 : index
  %2 = tensor.extract %1[%c1] : tensor<2xindex>
  %3 = arith.index_cast %2 : index to i32
  return %3 : i32
}

// -----

// test: scalarize mhlo.reduce whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @main(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NOT: mhlo.reduce
  // CHECK: %[[RET:.*]] = arith.muli %[[ARG0]], %[[ARG1]]
  // CHECK: return %[[RET]] : i32
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xi32>
  %1 = "mhlo.reshape"(%0) : (tensor<2xi32>) -> tensor<2x1xi32>
  %2 = "mhlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %3 = mhlo.reduce(%1 init: %2) applies mhlo.multiply across dimensions = [0] : (tensor<2x1xi32>, tensor<i32>) -> tensor<1xi32>
  %c0 = arith.constant 0 : index
  %4 = tensor.extract %3[%c0] : tensor<1xi32>
  return %4 : i32
}

// -----

// test: scalarize mhlo.gather whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @main(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NOT: mhlo.gather
  // CHECK: return %[[ARG0]] : i32
  %0 = tensor.from_elements %arg0, %arg1, %arg0 : tensor<3xi32>
  %1 = "mhlo.constant"() {value = dense<[0, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "mhlo.gather"(%0, %1) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<[1]> : tensor<1xi64>, indices_are_sorted = false} : (tensor<3xi32>, tensor<2xi32>) -> tensor<2xi32>
  %c1 = arith.constant 1 : index
  %3 = tensor.extract %2[%c1] : tensor<2xi32>
  return %3 : i32
}

// -----

// test: scalarize mhlo.dynamic_gather whenever possible

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @main(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NOT: mhlo.gather
  // CHECK: return %[[ARG0]] : i32
  %0 = tensor.from_elements %arg0, %arg1, %arg0 : tensor<3xi32>
  %1 = "mhlo.constant"() {value = dense<[0, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "mhlo.constant"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "mhlo.dynamic_gather"(%0, %1, %2) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<3xi32>, tensor<2xi32>, tensor<1xi32>) -> tensor<2xi32>
  %c1 = arith.constant 1 : index
  %4 = tensor.extract %3[%c1] : tensor<2xi32>
  return %4 : i32
}

// -----

// test: inject static info for mhlo.real_dynamic_slice

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index) -> tensor<?x?xf32, [@[[S0]], @[[S2:.*]]]>
func.func @main(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?xf32> {
  // CHECK: mhlo.real_dynamic_slice
  // CHECK-SAME: limit_indices = dense<[-1, -2]> : tensor<2xi64>
  // CHECK-SAME: start_indices = dense<[0, -2]> : tensor<2xi64>
  // CHECK-SAME: strides = dense<1> : tensor<2xi64>
  %0 = arith.constant 0 : index
  %1 = tensor.from_elements %0, %arg1 : tensor<2xindex>
  %2 = tensor.dim %arg0, %0 : tensor<?x?xf32>
  %3 = tensor.from_elements %2, %arg2 : tensor<2xindex>
  %4 = arith.constant 1 : index
  %5 = tensor.from_elements %4, %4 : tensor<2xindex>
  %6 = "mhlo.real_dynamic_slice"(%arg0, %1, %3, %5) : (tensor<?x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}

// -----

// test: inject static info for mhlo.dynamic_pad

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@[[S0:.*]], @[[S1:.*]]]>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: tensor<f32>) -> tensor<?x?xf32, [@[[S2:.*]], @[[S3:.*]]]>
func.func @main(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: tensor<f32>) -> tensor<?x?xf32> {
  // CHECK: mhlo.dynamic_pad
  // CHECK-SAME: edge_padding_high = dense<[-2, 0]>
  // CHECK-SAME: edge_padding_low = dense<[0, -2]>
  // CHECK-SAME: interior_padding = dense<0> : tensor<2xi64>
  %0 = arith.constant 0 : index
  %1 = tensor.from_elements %0, %arg1 : tensor<2xindex>
  %2 = tensor.from_elements %arg2, %0 : tensor<2xindex>
  %3 = tensor.from_elements %0, %0 : tensor<2xindex>
  %4 = "mhlo.dynamic_pad"(%arg0, %arg3, %1, %2, %3) : (tensor<?x?xf32>, tensor<f32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// test pattern:
// convert:
//   dyn_reshape(tensor<8xf32>) -> tensor<1x?xf32>
// to:
//   dyn_reshape(tensor<8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32>, %[[ARG1:.*]]: tensor<2xindex>) -> tensor<1x8xf32>
func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<2xindex>) -> (tensor<1x?xf32>) {
  // CHECK: mhlo.reshape
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<8xf32>, tensor<2xindex>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}

// -----

// Test mhlo.compute_reshape_shape

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32, [@S0, @S1]>, %[[ARG1:.*]]: tensor<?x?x?xf32, [@S2, @S3, @S4]>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: tensor<3xindex>)
// CHECK-SAME: tensor<?x?x?xf32, [@S2, @S3, @S4]>
func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: index, %arg3: tensor<3xindex>) -> tensor<?x?x?xf32> {
  %0 = shape.shape_of %arg1 : tensor<?x?x?xf32> -> tensor<3xindex>
  %1 = mhlo.compute_reshape_shape %arg2, %0 : (index, tensor<3xindex>) -> tensor<3xindex>
  %2 = "mhlo.dynamic_reshape"(%arg0, %1)
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %2 : tensor<?x?x?xf32>
}

