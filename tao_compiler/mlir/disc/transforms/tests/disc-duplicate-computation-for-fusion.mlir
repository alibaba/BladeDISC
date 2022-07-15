// RUN: disc-opt --disc-duplicate-computation-for-fusion -split-input-file %s | FileCheck %s

// CHECK-LABEL: @main
func @main(%arg0 : memref<?x?xf32, "gpu">, %arg1 : memref<f32, "gpu">, %arg2 : memref<2xi32, "cpu">) -> memref<?x?xf32, "gpu"> {
  // make sure dynamic_broadcast_in_dim is duplicated
  // CHECK: lmhlo.dynamic_broadcast_in_dim
  // CHECK: lmhlo.dynamic_broadcast_in_dim
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, "gpu">
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, "gpu">
  %2 = memref.alloc(%0, %1) : memref<?x?xf32, "gpu">
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg2, %2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<f32, "gpu">, memref<2xi32, "cpu">, memref<?x?xf32, "gpu">) -> ()
  %3 = memref.alloc(%0, %1) : memref<?x?xf32, "gpu">
  "lmhlo.abs"(%2, %3) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %4 = memref.alloc(%0, %1) : memref<?x?xf32, "gpu">
  "lmhlo.dot_general"(%3, %3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %5 = memref.alloc(%0, %1) : memref<?x?xf32, "gpu">
  "lmhlo.exponential"(%4, %5) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  %6 = memref.alloc(%0, %1) : memref<?x?xf32, "gpu">
  "lmhlo.add"(%2, %5, %6) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  return %6 :  memref<?x?xf32, "gpu">
}

// -----

// CHECK-LABEL: @main
func @main(%arg0 : memref<?x128xf32, "gpu">, %arg1 : memref<128xf32, "gpu">, %arg2 : memref<1xi32, "cpu">) -> memref<?x128xf32, "gpu"> {
  // make sure dynamic_broadcast_in_dim is duplicated
  // CHECK: lmhlo.dynamic_broadcast_in_dim
  // CHECK: lmhlo.dynamic_broadcast_in_dim
  %c0 = arith.constant 0 : index
  "lmhlo.constant"(%arg1) {value = dense<0.1> : tensor<128xf32>} : (memref<128xf32, "gpu">) -> ()
  %0 = memref.dim %arg0, %c0 : memref<?x128xf32, "gpu">
  %2 = memref.alloc(%0) : memref<?x128xf32, "gpu">
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg2, %2) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (memref<128xf32, "gpu">, memref<1xi32, "cpu">, memref<?x128xf32, "gpu">) -> ()
  %3 = memref.alloc(%0) : memref<?x128xf32, "gpu">
  "lmhlo.abs"(%2, %3) : (memref<?x128xf32, "gpu">, memref<?x128xf32, "gpu">) -> ()
  %4 = memref.alloc(%0) : memref<?x128xf32, "gpu">
  "lmhlo.dot_general"(%3, %3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x128xf32, "gpu">, memref<?x128xf32, "gpu">, memref<?x128xf32, "gpu">) -> ()
  %5 = memref.alloc(%0) : memref<?x128xf32, "gpu">
  "lmhlo.exponential"(%4, %5) : (memref<?x128xf32, "gpu">, memref<?x128xf32, "gpu">) -> ()
  %6 = memref.alloc(%0) : memref<?x128xf32, "gpu">
  "lmhlo.add"(%2, %5, %6) : (memref<?x128xf32, "gpu">, memref<?x128xf32, "gpu">, memref<?x128xf32, "gpu">) -> ()
  return %6 :  memref<?x128xf32, "gpu">
}