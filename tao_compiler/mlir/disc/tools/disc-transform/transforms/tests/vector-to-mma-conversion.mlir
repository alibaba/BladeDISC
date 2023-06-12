// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s


// CHECK-LABEL: @vector_to_mma_conversion
// CHECK-SAME: (%[[ARG0:.*]]: memref<16x16xf16, #gpu.address_space<workgroup>>, %[[ARG1:.*]]: memref<16x8xf16, #gpu.address_space<workgroup>>, %[[ARG2:.*]]: memref<16x8xf16>)
func.func @vector_to_mma_conversion(%arg0: memref<16x16xf16, #gpu.address_space<workgroup>>, %arg1: memref<16x8xf16, #gpu.address_space<workgroup>>, %arg2: memref<16x8xf16>) {
  // CHECK: %[[LHS:.*]] = nvgpu.ldmatrix %[[ARG0]]
  // CHECK: %[[RHS:.*]] = nvgpu.ldmatrix %[[ARG1]]
  // CHECK: %[[RES:.*]] = nvgpu.mma.sync(%[[LHS]], %[[RHS]], %[[REGC:.*]]) {mmaShape = [16, 8, 16]}
  %cst = arith.constant 0.0e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
  %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x8xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
  %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
  %3 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<16x16xf16>, vector<16x8xf16> into vector<16x8xf16>
  vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
  return
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.disc.vector.vector_to_mma_conversion %func : (!pdl.operation) -> ()
}
