// RUN: disc-opt -disc-dot-merge -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @dot_merging_dynamic
func.func @dot_merging_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m = tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.real_dynamic_slice
  // CHECK:     mhlo.real_dynamic_slice
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dot_merging_dynamic_batch
func.func @dot_merging_dynamic_batch(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>, %b: tensor<index>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m = tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %dim_b = tensor.extract %b[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_b, %dim_m, %dim_k : tensor<3xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %rhs_shape = tensor.from_elements %dim_b, %dim_k, %dim_n : tensor<3xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "mhlo.abs"(%1) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK:     mhlo.concatenate
  // CHECK-SAME: -> tensor<?x?x?xf32>
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.real_dynamic_slice
  // CHECK:     mhlo.real_dynamic_slice
  return %5: tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @dot_merging_static
func.func @dot_merging_static(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %1 = "mhlo.abs"(%arg0) : (tensor<128x256xf32>) -> tensor<128x256xf32>
  %2 = "mhlo.abs"(%arg1) : (tensor<256x512xf32>) -> tensor<256x512xf32>
  %3 = "mhlo.dot_general"(%arg0, %2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %4 = "mhlo.add"(%0, %3) : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
  // CHECK:     mhlo.concatenate
  // CHECK-SAME: -> tensor<256x1024xf32>
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.slice
  // CHECK:     mhlo.slice
  // CHECK-NOT: mhlo.real_dynamic_slice
  return %4: tensor<128x512xf32>
}

// CHECK-LABEL: func.func @dot_merging_static_batch
func.func @dot_merging_static_batch(%arg0: tensor<2x128x256xf32>, %arg1: tensor<2x256x512xf32>) -> tensor<2x128x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<2x128x256xf32>, tensor<2x256x512xf32>) -> tensor<2x128x512xf32>
  %1 = "mhlo.abs"(%arg0) : (tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  %2 = "mhlo.abs"(%arg1) : (tensor<2x256x512xf32>) -> tensor<2x256x512xf32>
  %3 = "mhlo.dot_general"(%arg0, %2) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<2x128x256xf32>, tensor<2x256x512xf32>) -> tensor<2x128x512xf32>
  %4 = "mhlo.add"(%0, %3) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  // CHECK:     mhlo.concatenate
  // CHECK:     -> tensor<2x256x1024xf32>
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.slice
  // CHECK:     mhlo.slice
  // CHECK-NOT: mhlo.real_dynamic_slice
  return %4: tensor<2x128x512xf32>
}

// CHECK-LABEL: func.func @dot_not_merging_diff_batch
func.func @dot_not_merging_diff_batch(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>, %b: tensor<index>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m = tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %dim_b = tensor.extract %b[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_b, %dim_m, %dim_k : tensor<3xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %rhs_shape_1 = tensor.from_elements %dim_b, %dim_k, %dim_n : tensor<3xindex>
  %rhs_shape_2 = tensor.from_elements %dim_k, %dim_n, %dim_b : tensor<3xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape_1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape_2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [2],lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.real_dynamic_slice
  // CHECK-NOT: mhlo.real_dynamic_slice
  return %5: tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @dot_not_merging_cycle
func.func @dot_not_merging_cycle(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.extract %arg2[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%0, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: mhlo.real_dynamic_slice
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dot_batching
// This UT builds the tensor shape explicitly because the current implement of
// ShapeAnalysis relies on them to build DimValue. ShapeAnalysisV2, which is on
// going for development, will solve this problem.
func.func @dot_batching(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m= tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK:     mhlo.dynamic_reshape
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK-DAG: lhs_batching_dimensions = [0]
  // CHECK-DAG: rhs_batching_dimensions = [0]
  // CHECK-DAG: lhs_contracting_dimensions = [2]
  // CHECK-DAG: rhs_contracting_dimensions = [1]
  // CHECK:     -> tensor<2x?x?xf32>
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.real_dynamic_slice
  return %5: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dot_batching_static
func.func @dot_batching_static(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %1 = "mhlo.abs"(%arg0) : (tensor<128x256xf32>) -> tensor<128x256xf32>
  %2 = "mhlo.abs"(%arg1) : (tensor<256x512xf32>) -> tensor<256x512xf32>
  %3 = "mhlo.dot_general"(%1, %2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  %4 = "mhlo.add"(%0, %3) : (tensor<128x512xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>
  // CHECK-NOT: mhlo.dynamic_reshape
  // CHECK:     mhlo.reshape
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK-DAG: lhs_batching_dimensions = [0]
  // CHECK-DAG: rhs_batching_dimensions = [0]
  // CHECK-DAG: lhs_contracting_dimensions = [2]
  // CHECK-DAG: rhs_contracting_dimensions = [1]
  // CHECK:     -> tensor<2x128x512xf32>
  // CHECK-NOT: mhlo.dot_general
  // CHECK:     mhlo.slice
  // CHECK-NOT: mhlo.real_dynamic_slice
  return %4: tensor<128x512xf32>
}

// CHECK-LABEL: func.func @dot_not_batching_diff_dtype
func.func @dot_not_batching_diff_dtype(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %m: tensor<index>, %n: tensor<index>, %k: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim_m = tensor.extract %m[] : tensor<index>
  %dim_n = tensor.extract %n[] : tensor<index>
  %dim_k = tensor.extract %k[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim_m, %dim_k : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim_k, %dim_n : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.convert"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %4 = "mhlo.convert"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %5 = "mhlo.dot_general"(%3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %6 = "mhlo.convert"(%5) : (tensor<?x?xf16>) -> tensor<?x?xf32>
  %7 = "mhlo.add"(%2, %6) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: lhs_batching_dimensions = [0]
  // CHECK-NOT: rhs_batching_dimensions = [0]
  return %7: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dot_not_batching_cycle
func.func @dot_not_batching_cycle(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<index>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.extract %arg2[] : tensor<index>
  %lhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %lhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %rhs_shape = tensor.from_elements %dim, %dim : tensor<2xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %rhs_shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "mhlo.abs"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "mhlo.dot_general"(%3, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "mhlo.add"(%2, %5) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NOT: mhlo.concatenate
  // CHECK:     mhlo.dot_general
  // CHECK:     mhlo.dot_general
  // CHECK-NOT: lhs_batching_dimensions = [0]
  // CHECK-NOT: rhs_batching_dimensions = [0]
  return %5: tensor<?x?xf32>
}


// CHECK-LABEL: func.func @dot_merging_shared_rhs
func.func @dot_merging_shared_rhs(%arg0: tensor<?x64xf32>, %arg1: tensor<?x64xf32>, %cst: tensor<64x8xf32>) -> (tensor<?x8xf32>, tensor<?x8xf32>) {
  // CHECK: mhlo.concatenate
  // CHECK-SAME: -> tensor<?x64xf32>
  // CHECK: mhlo.dot_general
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [0]
  %a = "mhlo.dot_general"(%arg0, %cst) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x64xf32>, tensor<64x8xf32>) -> tensor<?x8xf32>
  %b = "mhlo.dot_general"(%arg1, %cst) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x64xf32>, tensor<64x8xf32>) -> tensor<?x8xf32>
  // CHECK: mhlo.real_dynamic_slice
  // CHECK: mhlo.real_dynamic_slice
  // CHECK: return
  // CHECK-SAME: tensor<?x8xf32>, tensor<?x8xf32>
  return %a, %b : tensor<?x8xf32>, tensor<?x8xf32>
}


// CHECK-LABEL: func.func @dot_merging_shared_lhs
func.func @dot_merging_shared_lhs(%arg0: tensor<?x64xf32>, %arg1: tensor<?x64xf32>, %cst: tensor<64x8xf32>) -> (tensor<?x8xf32>, tensor<?x8xf32>) {
  // CHECK: mhlo.concatenate
  // CHECK-SAME: -> tensor<?x64xf32>
  // CHECK: mhlo.dot_general
  // CHECK-SAME: lhs_contracting_dimensions = [0]
  // CHECK-SAME: rhs_contracting_dimensions = [1]
  %a = "mhlo.dot_general"(%cst, %arg0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<64x8xf32>, tensor<?x64xf32>) -> tensor<?x8xf32>
  %b = "mhlo.dot_general"(%cst, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<64x8xf32>, tensor<?x64xf32>) -> tensor<?x8xf32>
  // CHECK: mhlo.real_dynamic_slice
  // CHECK: mhlo.real_dynamic_slice
  // CHECK: return
  // CHECK-SAME: tensor<?x8xf32>, tensor<?x8xf32>
  return %a, %b : tensor<?x8xf32>, tensor<?x8xf32>
}



