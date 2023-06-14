// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<(d0)[s0] -> (-d0 + s0, 288)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x3072xf32>, %[[ARG1:.*]]: tensor<?x768xf32>)
func.func @main(%arg0: tensor<?x3072xf32>, %arg1: tensor<?x768xf32>) -> tensor<?x768xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c288 = arith.constant 288 : index
  %0 = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant"} dense<-8.000000e-01> : tensor<3072x768xf32>
  %1 = disc_linalg_ext.constant_wrapper {disc.transform.name = "constant_1"} dense<-1.000000e-01> : tensor<768xf32>
  %2 = linalg.fill {disc.transform.name = "dot_general"} ins(%cst : f32) outs(%arg1 : tensor<?x768xf32>) -> tensor<?x768xf32>
  %3 = linalg.matmul {disc.transform.name = "dot_general"} ins(%arg0, %0 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%2 : tensor<?x768xf32>) -> tensor<?x768xf32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x3072xf32>
  // CHECK: %[[T0:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[T1:.*]] = %[[ARG1]])
  %4 = scf.for %arg2 = %c0 to %dim step %c288 iter_args(%arg3 = %3) -> (tensor<?x768xf32>) {
    // CHECK: %[[T2:.*]] = tensor.extract_slice %[[T1]]
    // CHECK: %[[T3:.*]] = linalg.fill
    // CHECK-SAME: outs(%[[T2]] : tensor<?x768xf32>)
    // CHECK-NEXT: %[[T4:.*]] = linalg.matmul
    // CHECK-SAME: outs(%[[T3]] : tensor<?x768xf32>)
    // CHECK-NEXT: %[[T5:.*]] = linalg.generic
    // CHECK-SAME: outs(%[[T4]] : tensor<?x768xf32>)
    %5 = affine.min #map(%arg2)[%dim]
    %extracted_slice = tensor.extract_slice %arg3[%arg2, 0] [%5, 768] [1, 1] : tensor<?x768xf32> to tensor<?x768xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<768xf32>) outs(%extracted_slice : tensor<?x768xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "subtract"} {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.subf %out, %in : f32
      linalg.yield %7 : f32
    } -> tensor<?x768xf32>
    %inserted_slice = tensor.insert_slice %6 into %arg3[%arg2, 0] [%5, 768] [1, 1] : tensor<?x768xf32> into tensor<?x768xf32>
    scf.yield %inserted_slice : tensor<?x768xf32>
  }
  return %4 : tensor<?x768xf32>
}
transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.match ops{["linalg.matmul"]}  in %arg0 : (!pdl.operation) -> !pdl.operation
  %2 = transform.structured.match ops{["linalg.fill"]}  in %arg0 : (!pdl.operation) -> !pdl.operation
  %3 = transform.disc.fuse_into_containing_op %1 into %0
  %4 = transform.disc.fuse_into_containing_op %2 into %0
  %5 = transform.disc.apply_patterns %arg0 {canonicalization}
}