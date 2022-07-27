// RUN: disc-opt %s -disc-memref-cse | FileCheck %s

// CHECK-LABEL: @cse_in_the_same_block
func.func @cse_in_the_same_block() -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  // CHECK: memref.load
  // CHECK-NOT: memref.load
  %a = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
  %b = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
  return %a, %b : f32, f32
}

// CHECK-LABEL: @cse_in_the_dominant_block
func.func @cse_in_the_dominant_block(%pred : i1) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  // CHECK: memref.load
  %a = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
  // CHECK: scf.if
  %result = scf.if %pred -> (f32) {
    // CHECK-NOT: memref.load
    %b = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
    scf.yield %b : f32 
  } else {
    %f0 = arith.constant 0.0 : f32
    scf.yield %f0 : f32
  }
  return %a, %result : f32, f32
}

// CHECK-LABEL: @should_not_cse_0
func.func @should_not_cse_0(%pred : i1) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  // CHECK: memref.load
  %a = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
  // CHECK: scf.if
  %result = scf.if %pred -> (f32) {
    // CHECK: memref.load
    %b = memref.load %memref[%c1, %c0] : memref<?x?xf32, "gpu">
    scf.yield %b : f32
  } else {
    %f0 = arith.constant 0.0 : f32
    scf.yield %f0 : f32
  }
  return %a, %result : f32, f32
}

// CHECK-LABEL: @should_not_cse_1
func.func @should_not_cse_1(%pred : i1) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref1 = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  %memref2 = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  // CHECK: memref.load
  %a = memref.load %memref1[%c0, %c1] : memref<?x?xf32, "gpu">
  // CHECK: scf.if
  %result = scf.if %pred -> (f32) {
    // CHECK: memref.load
    %b = memref.load %memref2[%c0, %c1] : memref<?x?xf32, "gpu">
    scf.yield %b : f32
  } else {
    %f0 = arith.constant 0.0 : f32
    scf.yield %f0 : f32
  }
  return %a, %result : f32, f32
}

// CHECK-LABEL: @should_not_cse_2
func.func @should_not_cse_2(%pred : i1) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  // CHECK: scf.if
  %result = scf.if %pred -> (f32) {
    // CHECK: memref.load
    %b = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
    scf.yield %b : f32
  } else {
    %f0 = arith.constant 0.0 : f32
    scf.yield %f0 : f32
  }
  // CHECK: memref.load
  %a = memref.load %memref[%c0, %c1] : memref<?x?xf32, "gpu">
  return %a, %result : f32, f32
}

// CHECK-LABEL: @cse_iteratively
func.func @cse_iteratively(%i : index, %j : index) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %memref = memref.alloc(%c1, %c2) : memref<?x?xf32, "gpu">
  // CHECK: memref.load
  // CHECK-NOT: memref.load
  %m = "arith.addi"(%j, %i) : (index, index) -> index
  %n = "arith.addi"(%j, %i) : (index, index) -> index
  %a = memref.load %memref[%c0, %m] : memref<?x?xf32, "gpu">
  %b = memref.load %memref[%c0, %n] : memref<?x?xf32, "gpu">
  return %a, %b : f32, f32
}

