// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @tensor.empty
func.func @tensor.empty(%arg0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T0:.*]] = memref.alloc({{.*}})
  // CHECK-NEXT: return %[[T0]]
  %0 = tensor.empty(%arg0, %arg0) : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.bufferize %arg1
}