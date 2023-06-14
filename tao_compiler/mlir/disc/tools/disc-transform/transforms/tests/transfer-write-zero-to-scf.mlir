// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @transfer_write_zero_to_scf
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x128xf16>)
func.func @transfer_write_zero_to_scf(%arg0: memref<128x128xf16>) {
  // CHECK-DAG: %[[cst:.*]] = arith.constant 0
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[c128:.*]] = arith.constant 128 : index
  // CHECK: scf.for %[[arg2:.*]] = %[[c0]] to %[[c128]] step %[[c1]] {
  // CHECK:   scf.for %[[arg3:.*]] = %[[c0]] to %[[c128]] step %[[c1]] {
  // CHECK:     memref.store %[[cst]], %[[ARG0]][%[[arg2]], %[[arg3]]] : memref<128x128xf16>
  // CHECK:   }
  // CHECK: }
  %cst = arith.constant dense<0.0> : vector<128x128xf16>
  %c0 = arith.constant 0 : index
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} :
      vector<128x128xf16>, memref<128x128xf16>
  return
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %arg1: (!pdl.operation) -> !pdl.operation
  transform.disc.transfer_write_zero_to_scf %func : (!pdl.operation) -> ()
}
