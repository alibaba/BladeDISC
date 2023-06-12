// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @split_reduction_serial
// CHECK-SAME: (%[[ARG0:.*]]: tensor<256x128xf16>, %[[ARG1:.*]]: tensor<128x256xf16>, %[[ARG2:.*]]: tensor<256x256xf16>)
func.func @split_reduction_serial(%A: tensor<256x128xf16>, %B: tensor<128x256xf16>, %C: tensor<256x256xf16>) -> tensor<256x256xf16> {
  // CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RES:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[C128]] step %[[C32]] iter_args(%[[ARG4:.*]] = %[[FILL:.*]]) -> (tensor<256x256xf16>)
  // CHECK:   %[[extracted_slice:.*]] = tensor.extract_slice %[[ARG0]][0, %[[ARG3]]] [256, 32] [1, 1] : tensor<256x128xf16> to tensor<256x32xf16>
  // CHECK:   %[[extracted_slice_0:.*]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], 0] [32, 256] [1, 1] : tensor<128x256xf16> to tensor<32x256xf16>
  // CHECK:   %[[MATMUL:.*]] = linalg.matmul ins(%[[extracted_slice]], %[[extracted_slice_0]] : tensor<256x32xf16>, tensor<32x256xf16>) outs(%arg4 : tensor<256x256xf16>) -> tensor<256x256xf16>
  // CHECK:   scf.yield %[[MATMUL]] : tensor<256x256xf16>
  // CHECK: return %[[RES]]
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
  %matmul_split, %foreach_split = transform.disc.split_reduction_serial %matmul_op by tile_sizes = [32]
}
