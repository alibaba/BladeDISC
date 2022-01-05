// RUN: disc-opt %s -map-parallel-loops-to-gpu | FileCheck %s

// CHECK-LABEL: @tile_nested_innermost
func @tile_nested_innermost() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.parallel
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    // CHECK: scf.parallel
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    // CHECK: mapping = [{bound = #map, map = #map, processor = 3 : i64}, {bound = #map, map = #map, processor = 4 : i64}]
    }
  // CHECK: mapping = [{bound = #map, map = #map, processor = 0 : i64}, {bound = #map, map = #map, processor = 1 : i64}]
  }
  // CHECK: scf.parallel
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
  // CHECK: mapping = [{bound = #map, map = #map, processor = 0 : i64}, {bound = #map, map = #map, processor = 1 : i64}]}
  }
  return
}
