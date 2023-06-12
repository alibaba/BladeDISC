// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @gmem_to_smem
// CHECK-SAME: (%[[ARG0:.*]]: memref<2x2xf16>, %[[ARG1:.*]]: memref<2x2xf16, #gpu.address_space<workgroup>>)
func.func @gmem_to_smem(%arg0: memref<2x2xf16>, %arg1: memref<2x2xf16, #gpu.address_space<workgroup>>) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[arg2:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
  // CHECK:   scf.for %[[arg3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
  // CHECK:     %[[VAR0:.*]] = memref.load %[[ARG0]][%[[arg2]], %[[arg3]]] : memref<2x2xf16>
  // CHECK:     memref.store %[[VAR0]], %[[ARG1]][%[[arg2]], %[[arg3]]] : memref<2x2xf16, #gpu.address_space<workgroup>>
  // CHECK:   }
  // CHECK: }
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0: memref<2x2xf16>) outs(%arg1: memref<2x2xf16, #gpu.address_space<workgroup>>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  }
  return
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.gmem_to_smem %generic : (!pdl.operation) -> ()
}
