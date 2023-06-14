// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @promote_dot_operands
// CHECK-SAME: (%[[ARG0:.*]]: tensor<256x128xf16>, %[[ARG1:.*]]: tensor<128x256xf16>, %[[ARG2:.*]]: tensor<256x256xf16>)
func.func @promote_dot_operands(%A: tensor<256x128xf16>, %B: tensor<128x256xf16>, %C: tensor<256x256xf16>) -> tensor<256x256xf16> {
  // CHECK: %[[VAR1:.*]] = bufferization.alloc_tensor() copy(%[[ARG0]]) {bufferization.escape = [false]} : tensor<256x128xf16>
  // CHECK: %[[VAR2:.*]] = bufferization.alloc_tensor() copy(%[[ARG1]]) {bufferization.escape = [false]} : tensor<128x256xf16>
  // CHECK: %[[VAR3:.*]] = linalg.matmul ins(%[[VAR1]], %[[VAR2]] : tensor<256x128xf16>, tensor<128x256xf16>) outs(%[[VAR0:.*]] : tensor<256x256xf16>) -> tensor<256x256xf16>
  // CHECK: return %[[VAR3]]
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0e+00 : f16
  %fill = linalg.fill ins(%cst : f16) outs(%C : tensor<256x256xf16>) -> tensor<256x256xf16>
  %0 = linalg.matmul ins(%A, %B : tensor<256x128xf16>, tensor<128x256xf16>)
      outs(%fill : tensor<256x256xf16>) -> (tensor<256x256xf16>)
  return %0 : tensor<256x256xf16>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %matmul_op = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %matmul_promote, %lhs, %rhs = transform.disc.promote_dot_operands %matmul_op [0, 1] : (!pdl.operation) -> (!pdl.operation, !pdl.operation,!pdl.operation)
}
