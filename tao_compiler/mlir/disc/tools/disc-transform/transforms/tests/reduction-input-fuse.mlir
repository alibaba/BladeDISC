// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (512, d0 - d1)>

// CHECK-LABEL: @test_reduction_input_fuse
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
func.func @test_reduction_input_fuse(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  // CHECK: %[[M:.*]] = tensor.dim %[[ARG0]], %c0 : tensor<?x?xf32>
  // CHECK: %[[K:.*]] = tensor.dim %[[ARG0]], %c1 : tensor<?x?xf32>
  // CHECK: %[[N:.*]] = tensor.dim %[[ARG1]], %c1 : tensor<?x?xf32>
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %k = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %n = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %t0 = linalg.fill {disc.transform.name = "fill"} ins(%cst : f32) outs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: scf.for %[[IV:.*]] = %c0 to %[[K]] step %c512 iter_args(%[[V0:.*]] = %[[ARG0]])
  %t1 = scf.for %iv = %c0 to %k step %c512 iter_args(%v0 = %t0) -> (tensor<?x?xf32>) {
    %r0 = affine.min #map1(%k, %iv)
    // CHECK: %[[R1:.*]] = tensor.extract_slice %[[ARG0]]
    // CHECK: %[[R2:.*]] = tensor.extract_slice %[[ARG1]]
    // CHECK: %[[R3:.*]] = tensor.extract_slice %[[V0]]
    // CHECK: %[[R4:.*]] = disc_linalg_ext.conditional_generic
    // CHECK-SAME: outs(%[[R3]] : tensor<?x?xf32>)
    // CHECK: %[[R5:.*]] = linalg.matmul
    // CHECK-SAME: ins(%[[R1]], %[[R2]] : tensor<?x?xf32>, tensor<?x?xf32>)
    // CHECK-SAME: outs(%[[R4]] : tensor<?x?xf32>)
    %r1 = tensor.extract_slice %arg0[0, %iv] [%m, %r0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %r2 = tensor.extract_slice %arg1[%iv, 0] [%r0, %n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %r3 = tensor.extract_slice %v0[0, 0] [%m, %n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %r4 = linalg.matmul {disc.transform.name = "dot_general"} ins(%r1, %r2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%r3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %r5 = tensor.insert_slice %r4 into %v0[0, 0] [%m, %n] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    scf.yield %r5 : tensor<?x?xf32>
  }
  return %t1 : tensor<?x?xf32>
}


transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {disc.transform.name = "fill"} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.match ops{["scf.for"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %2, %3 = transform.disc.reduction_input_fuse %0 into %1
}