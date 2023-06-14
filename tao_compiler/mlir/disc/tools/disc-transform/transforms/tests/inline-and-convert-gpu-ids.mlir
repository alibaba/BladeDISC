// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @foreach_thread_to_gpu_warps
// CHECK-SAME: (%[[arg0:.*]]: memref<2x2xf16>)
func.func @foreach_thread_to_gpu_warps(%arg0: memref<2x2xf16>) {
  // CHECK-DAG: %[[cst:.*]] = arith.constant 0
  // CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[c128:.*]] = arith.constant 128 : index
  // CHECK: scf.parallel (%[[arg1:.*]], %[[arg2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c4]], %[[c128]]) step (%[[c1]], %[[c1]]) {
  // CHECK:   %[[LANE:.*]] = arith.remui %[[arg2]], %[[c32]]
  // CHECK:   memref.store %[[cst]], %[[arg0]][%[[arg1]], %[[LANE]]]
  // CHECK:   scf.yield
  // CHECK: } {mapping = "cta-thread-mapping"}
  %cst = arith.constant 0.0e+00 : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %id = gpu.lane_id
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c4, %c128) step (%c1, %c1) {
    memref.store %cst, %arg0[%arg1, %id] : memref<2x2xf16>
  } {mapping = "cta-thread-mapping"}
  return
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.inline_and_convert_gpu_ids %func : (!pdl.operation) -> ()
}
