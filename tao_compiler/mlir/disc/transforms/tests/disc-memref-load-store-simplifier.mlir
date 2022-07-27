// RUN: disc-opt %s -disc-memref-load-store-simplifier | FileCheck %s

// CHECK-LABEL: @opt_in_the_same_block
// CHECK-SAME: (%[[INPUT:.*]]: f32
func.func @opt_in_the_same_block(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK-NOT: memref.load
  %a = memref.load %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[INPUT]], %[[INPUT]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// -----

// CHECK-LABEL: @opt_in_the_dominant_block
// CHECK-SAME: (%[[INPUT:.*]]: f32
func.func @opt_in_the_dominant_block(%input: f32, %pred: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: scf.if
  %result = scf.if %pred -> (f32) {
    // CHECK-NOT: memref.load
    %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
    // CHECK: scf.yield %[[INPUT]]
    scf.yield %b : f32 
  } else {
    %f0 = arith.constant 0.0 : f32
    scf.yield %f0 : f32
  }
  return %result : f32
}

// -----

// CHECK-LABEL: @should_not_opt_diff_index
func.func @should_not_opt_diff_index(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref[%c1] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// -----

// CHECK-LABEL: @should_not_opt_diff_memref
func.func @should_not_opt_diff_memref(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  %memref_2 = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref_2[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// -----


// CHECK-LABEL: @should_not_opt_not_dominate
// CHECK-SAME: (%[[INPUT:.*]]: f32
func.func @should_not_opt_not_dominate(%input: f32, %pred: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  // CHECK: scf.if
  scf.if %pred -> () {
    // CHECK: memref.store
    memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
    scf.yield
  }
  // CHECK: memref.load
  %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
  return %b: f32
}

// ----

// CHECK-LABEL: @should_not_opt_on_gpu
func.func @should_not_opt_on_gpu(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "gpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "gpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref[%c0] : memref<?xf32, "gpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// CHECK-LABEL: @should_not_opt_multi_store
func.func @should_not_opt_multi_store(%input: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  %f0 = arith.constant 0.0 : f32
  memref.store %f0, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b : f32
}

// CHECK-LABEL: @should_not_opt_cast
func.func @should_not_opt_cast(%input: f32) -> (f32, memref<?xf32, "cpu">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  %cast = memref.reinterpret_cast %memref to offset: [0], sizes: [%c2], strides: [%c1] : memref<?xf32, "cpu"> to memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL:.*]] = memref.load
  %a = memref.load %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: %[[LOAD_VAL]], %[[LOAD_VAL]]
  %b = "arith.addf"(%a, %a) : (f32, f32) -> f32
  return %b, %cast : f32, memref<?xf32, "cpu">
}

// -----

// Three UTs to deal with case like:
//   store a[x]
//   if (%pred) {
//     store a[y]
//   }
//   load a[z]

// CHECK-LABEL: @opt_multi_store_diff_index
// CHECK-SAME: (%[[INPUT:.*]]: f32
func.func @opt_multi_store_diff_index(%input: f32, %pred: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  scf.if %pred -> () {
    memref.store %input, %memref[%c1] : memref<?xf32, "cpu">
    scf.yield
  }
  %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: return %[[INPUT]]
  return %b: f32
}

// CHECK-LABEL: @should_not_opt_multi_store_same_index
func.func @should_not_opt_multi_store_same_index(%input: f32, %pred: i1) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  // CHECK: memref.store
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: scf.if
  scf.if %pred -> () {
    // CHECK: memref.store
    memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
    scf.yield
  }
  // CHECK: memref.load
  %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
  return %b: f32
}

// CHECK-LABEL: @should_not_opt_multi_store_unknown_index
func.func @should_not_opt_multi_store_unknown_index(%input: f32, %pred: i1, %idx: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c2) : memref<?xf32, "cpu">
  // CHECK: memref.store
  memref.store %input, %memref[%c0] : memref<?xf32, "cpu">
  // CHECK: scf.if
  scf.if %pred -> () {
    // CHECK: memref.store
    memref.store %input, %memref[%idx] : memref<?xf32, "cpu">
    scf.yield
  }
  // CHECK: memref.load
  %b = memref.load %memref[%c0] : memref<?xf32, "cpu">
  return %b: f32
}