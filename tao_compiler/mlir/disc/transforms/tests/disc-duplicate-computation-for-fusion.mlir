// RUN: disc-opt -split-input-file --disc-duplicate-computation-for-fusion %s | FileCheck %s --check-prefix=BASE
// RUN: DISC_ENABLE_TRANSFORM_SCHEDULE=1 disc-opt -split-input-file --disc-duplicate-computation-for-fusion=gpu-enabled=false -cse -loop-invariant-code-motion --canonicalize %s -o - | FileCheck %s --check-prefix=TRANSFORM

// BASE-LABEL: @main
func.func @main(%arg0 : memref<?x?xf32, #gpu.address_space<global>>, %arg1 : memref<f32, #gpu.address_space<global>>, %arg2 : memref<2xi32>) -> memref<?x?xf32, #gpu.address_space<global>> {
  // make sure dynamic_broadcast_in_dim is duplicated
  // BASE: lmhlo.dynamic_broadcast_in_dim
  // BASE: lmhlo.dynamic_broadcast_in_dim
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32, #gpu.address_space<global>>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32, #gpu.address_space<global>>
  %2 = memref.alloc(%0, %1) : memref<?x?xf32, #gpu.address_space<global>>
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg2, %2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<f32, #gpu.address_space<global>>, memref<2xi32>, memref<?x?xf32, #gpu.address_space<global>>) -> ()
  %3 = memref.alloc(%0, %1) : memref<?x?xf32, #gpu.address_space<global>>
  "lmhlo.abs"(%2, %3) : (memref<?x?xf32, #gpu.address_space<global>>, memref<?x?xf32, #gpu.address_space<global>>) -> ()
  %4 = memref.alloc(%0, %1) : memref<?x?xf32, #gpu.address_space<global>>
  "lmhlo.dot_general"(%3, %3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x?xf32, #gpu.address_space<global>>, memref<?x?xf32, #gpu.address_space<global>>, memref<?x?xf32, #gpu.address_space<global>>) -> ()
  %5 = memref.alloc(%0, %1) : memref<?x?xf32, #gpu.address_space<global>>
  "lmhlo.exponential"(%4, %5) : (memref<?x?xf32, #gpu.address_space<global>>, memref<?x?xf32, #gpu.address_space<global>>) -> ()
  %6 = memref.alloc(%0, %1) : memref<?x?xf32, #gpu.address_space<global>>
  "lmhlo.add"(%2, %5, %6) : (memref<?x?xf32, #gpu.address_space<global>>, memref<?x?xf32, #gpu.address_space<global>>, memref<?x?xf32, #gpu.address_space<global>>) -> ()
  return %6 :  memref<?x?xf32, #gpu.address_space<global>>
}

// -----

// BASE-LABEL: @main
func.func @main(%arg0 : memref<?x128xf32, #gpu.address_space<global>>, %arg1 : memref<128xf32, #gpu.address_space<global>>, %arg2 : memref<1xi32>) -> memref<?x128xf32, #gpu.address_space<global>> {
  // make sure dynamic_broadcast_in_dim is duplicated
  // BASE: lmhlo.dynamic_broadcast_in_dim
  // BASE: lmhlo.dynamic_broadcast_in_dim
  %c0 = arith.constant 0 : index
  "lmhlo.constant"(%arg1) {value = dense<0.1> : tensor<128xf32>} : (memref<128xf32, #gpu.address_space<global>>) -> ()
  %0 = memref.dim %arg0, %c0 : memref<?x128xf32, #gpu.address_space<global>>
  %2 = memref.alloc(%0) : memref<?x128xf32, #gpu.address_space<global>>
  "lmhlo.dynamic_broadcast_in_dim"(%arg1, %arg2, %2) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (memref<128xf32, #gpu.address_space<global>>, memref<1xi32>, memref<?x128xf32, #gpu.address_space<global>>) -> ()
  %3 = memref.alloc(%0) : memref<?x128xf32, #gpu.address_space<global>>
  "lmhlo.abs"(%2, %3) : (memref<?x128xf32, #gpu.address_space<global>>, memref<?x128xf32, #gpu.address_space<global>>) -> ()
  %4 = memref.alloc(%0) : memref<?x128xf32, #gpu.address_space<global>>
  "lmhlo.dot_general"(%3, %3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x128xf32, #gpu.address_space<global>>, memref<?x128xf32, #gpu.address_space<global>>, memref<?x128xf32, #gpu.address_space<global>>) -> ()
  %5 = memref.alloc(%0) : memref<?x128xf32, #gpu.address_space<global>>
  "lmhlo.exponential"(%4, %5) : (memref<?x128xf32, #gpu.address_space<global>>, memref<?x128xf32, #gpu.address_space<global>>) -> ()
  %6 = memref.alloc(%0) : memref<?x128xf32, #gpu.address_space<global>>
  "lmhlo.add"(%2, %5, %6) : (memref<?x128xf32, #gpu.address_space<global>>, memref<?x128xf32, #gpu.address_space<global>>, memref<?x128xf32, #gpu.address_space<global>>) -> ()
  return %6 :  memref<?x128xf32, #gpu.address_space<global>>
}

// -----

// TRANSFORM-LABEL: @duplicate_dot_weight
func.func @duplicate_dot_weight(%arg0 : memref<128x128xf32>, %arg1 : memref<?x128xf32>,
                                %arg2 : memref<?x128xf32>, %arg3 : memref<?x128xf32>, %arg4 : memref<?x128xf32>) -> (memref<?x128xf32>, memref<?x128xf32>) {
  // TRANSFORM: lmhlo.constant
  // TRANSFORM-NEXT: memref.alloc
  // TRANSFORM-NEXT: lmhlo.constant
  // TRANSFORM-NEXT: lmhlo.dot_general
  // TRANSFORM-NEXT: lmhlo.dot_general
  "lmhlo.constant"(%arg0) {value = dense<0.1> : tensor<128x128xf32>} : (memref<128x128xf32>) -> ()
  "lmhlo.dot_general"(%arg1, %arg0, %arg2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x128xf32>, memref<128x128xf32>, memref<?x128xf32>) -> ()
  "lmhlo.dot_general"(%arg3, %arg0, %arg4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<?x128xf32>, memref<128x128xf32>, memref<?x128xf32>) -> ()
  return %arg2, %arg4 :  memref<?x128xf32>, memref<?x128xf32>
}