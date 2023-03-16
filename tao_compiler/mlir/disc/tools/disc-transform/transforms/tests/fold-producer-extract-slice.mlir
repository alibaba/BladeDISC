// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @fold_extracted_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
func.func @fold_extracted_slice(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = arith.addi %[[ARG2]], %[[ARG1]]
  // CHECK: %[[T1:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME: %[[T0]], %[[T0]]
  // CHECK-SAME: %[[ARG4]], %[[ARG4]]
  // CHECK-SAME: 1, 1
  // CHECK-NOT: tensor.extract_slice
  // CHECK: %[[T2:.*]] = linalg.fill
  // CHECK-SAME: %[[T1]]
  // CHECK: return %[[T2]]
  %0 = tensor.extract_slice %arg0[%arg1, %arg1][%arg3, %arg3][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.extract_slice %0[%arg2, %arg2][%arg4, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %slice = get_producer_of_operand %fill[1] : (!pdl.operation) -> !pdl.operation
  transform.disc.fold_producer_extract_slice %slice {max_repeat_num = 1}
}

// -----

// CHECK-LABEL: @fold_two_extracted_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
func.func @fold_two_extracted_slice(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<?x?xf32> {
  // CHECK: %[[T0:.*]] = arith.addi %[[ARG1]], %[[ARG2]]
  // CHECK: %[[T1:.*]] = arith.addi %[[T0]], %[[ARG1]]
  // CHECK: %[[T2:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME: %[[T1]], %[[T1]]
  // CHECK-SAME: %[[ARG4]], %[[ARG4]]
  // CHECK-SAME: 1, 1
  // CHECK-NOT: tensor.extract_slice
  // CHECK: %[[T3:.*]] = linalg.fill
  // CHECK-SAME: %[[T2]]
  // CHECK: return %[[T3]]
  %0 = tensor.extract_slice %arg0[%arg1, %arg1][%arg3, %arg3][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.extract_slice %0[%arg2, %arg2][%arg4, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %2 = tensor.extract_slice %1[%arg1, %arg1][%arg4, %arg4][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %slice = get_producer_of_operand %fill[1] : (!pdl.operation) -> !pdl.operation
  transform.disc.fold_producer_extract_slice %slice {max_repeat_num = 2}
}
