// RUN: disc-opt -disc-compare-simplifier -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func @dim_not_neg1_cmp_simplifier
func @dim_not_neg1_cmp_simplifier(%arg0: tensor<?x?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %c-1 = arith.constant -1 : index
  %1 = arith.cmpi eq, %0, %c-1 : index
  %2 = arith.select %1, %c0, %c-1 : index
  return %2: index
  // CHECK: %[[RES:.*]] = arith.constant -1 : index
  // CHECK: return %[[RES]] : index
}

// CHECK-LABEL: func @shape_not_zero_cmp_simplifier
func @shape_not_zero_cmp_simplifier(%arg0: tensor<128x?xf32>) -> index {
  %0 = shape.shape_of %arg0 : tensor<128x?xf32> -> tensor<2xindex>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0] : tensor<2xindex>
  %2 = arith.cmpi eq, %1, %c0 : index
  %c-1 = arith.constant -1 : index
  %3 = arith.select %2, %c0, %c-1 : index
  return %3: index
  // CHECK: %[[RES:.*]] = arith.constant -1 : index
  // CHECK: return %[[RES]] : index
}

// CHECK-LABEL: func @shape_computation_cmp_simplifier
func @shape_computation_cmp_simplifier(%arg0: tensor<?x?xf32>) -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %1 = shape.num_elements %0 : tensor<2xindex> -> index
  %7 = mhlo.compute_reshape_shape %1, %0 : index, tensor<2xindex> -> tensor<2xindex>
  %8 = tensor.extract %7[%c0] : tensor<2xindex>
  // positive * -1 * -1 + 0
  %2 = arith.muli %1, %c-1 : index
  %3 = arith.muli %2, %c-1 : index
  %4 = arith.addi %3, %c0 : index
  %5 = arith.cmpi ult, %4, %c-1 : index
  %6 = arith.select %5, %c0, %c-1 : index
  // positive * -1 + 0
  %tmp = arith.addi %8, %c1 : index
  %9 = arith.muli %tmp, %c-1 : index
  %10 = arith.addi %9, %c0 : index
  %11 = arith.cmpi sgt, %10, %c0 : index
  %12 = arith.select %11, %c0, %c-1 : index
  // positive * 0 + positive * -1 * 0
  %13 = arith.muli %8, %c0 : index
  %14 = arith.muli %8, %c-1 : index
  %15 = arith.muli %14, %c0 : index
  %16 = arith.addi %13, %15 : index
  %17 = arith.cmpi eq, %16, %c0 : index
  %18 = arith.select %17, %c-1, %c0 : index
  // positive * 1 + 1
  %19 = arith.muli %1, %c1 : index
  %20 = arith.addi %19, %c1 : index
  %21 = arith.cmpi sgt, %20, %c0 : index
  %22 = arith.select %21, %c-1, %c0 : index
  // CHECK: %[[RES:.*]] = arith.constant -1 : index
  // CHECK: return %[[RES]], %[[RES]], %[[RES]], %[[RES]]
  return %6, %12, %18, %22: index, index, index, index
}

// CHECK-LABEL: func @select_both_neg_cmp_simplifier
func @select_both_neg_cmp_simplifier(%arg0: tensor<?x?xf32>, %arg1: i1) -> index {
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = arith.addi %0, %c1 : index
  %2 = arith.muli %1, %c-1 : index
  %3 = arith.select %arg1, %2, %c-1 : index
  %4 = arith.cmpi slt, %1, %3 : index
  %5 = arith.select %4, %c1, %c-1 : index
  return %5: index
  // CHECK: %[[RES:.*]] = arith.constant -1 : index
  // CHECK: return %[[RES]] : index
}

// CHECK-LABEL: func @shape_not_zero_unknown_dim_no_simplifier
func @shape_not_zero_unknown_dim_no_simplifier(%arg0: tensor<?x?xf32>) -> index {
  %0 = shape.shape_of %arg0 : tensor<?x?xf32> -> tensor<2xindex>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0] : tensor<2xindex>
  %2 = arith.cmpi eq, %1, %c0 : index
  %c-1 = arith.constant -1 : index
  %3 = arith.select %2, %c0, %c-1 : index
  return %3: index
  // CHECK: %[[CMPI:.*]] = arith.cmpi eq
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMPI]]
  // CHECK: return %[[SELECT]]
}

// CHECK-LABEL: func @unknown_cmp_no_simplifier
func @unknown_cmp_no_simplifier(%arg0: tensor<2xindex>) -> index {
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %arg0[%c0] : tensor<2xindex>
  %2 = arith.cmpi eq, %1, %c0 : index
  %c-1 = arith.constant -1 : index
  %3 = arith.select %2, %c0, %c-1 : index
  // CHECK: %[[CMPI:.*]] = arith.cmpi eq
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMPI]]
  // CHECK: return %[[SELECT]]
  return %3: index
}