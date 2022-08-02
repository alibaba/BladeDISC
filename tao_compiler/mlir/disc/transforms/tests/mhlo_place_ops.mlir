// RUN: disc-opt --mhlo-place-ops -split-input-file %s | FileCheck %s

func.func @main(%arg : tensor<i64>) -> tensor<i64> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}}  {
  // CHECK: "mhlo.tuple"({{.*}}) {disc.device = ["cpu"]} : (tensor<i64>) -> tuple<tensor<i64>>
  // CHECK: "mhlo.get_tuple_element"({{.*}}) {disc.device = "cpu", index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
  %tuple = "mhlo.tuple"(%arg) : (tensor<i64>) -> tuple<tensor<i64>>
  %element = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
  return %element : tensor<i64>
}

// -----

func.func @main(%arg0: tensor<?x8xf32>) -> tensor<?x24xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}} {
  // CHECK: mhlo.constant
  // CHECK: disc.device = "cpu"
  // CHECK: mhlo_disc.h2d
  // CHECK-NOT: disc.device
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-SAME: disc.device = "gpu"
  // CHECK: tensor.from_elements
  // CHECK-NOT: disc.device
  // CHECK: mhlo.dynamic_reshape
  // CHECK-SAME: disc.device = "gpu"
  // CHECK: mhlo_disc.d2h
  // CHECK-SAME: disc.device = "cpu"
  %0 = mhlo.constant dense<[7, 3]> : tensor<2xi32>
  %c0 = arith.constant 0 : index
  %1 = tensor.dim %arg0, %c0 : tensor<?x8xf32>
  %c8 = arith.constant 8 : index
  %c0_0 = arith.constant 0 : index
  %2 = tensor.extract %0[%c0_0] : tensor<2xi32>
  %3 = arith.index_cast %2 : i32 to index
  %c1 = arith.constant 1 : index
  %4 = tensor.extract %0[%c1] : tensor<2xi32>
  %5 = arith.index_cast %4 : i32 to index
  %6 = tensor.from_elements %3, %1, %5, %c8 {disc.shape_op = true} : tensor<4xindex>
  %7 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %6) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  %8 = arith.muli %3, %1 : index
  %9 = arith.muli %5, %c8 : index
  %10 = tensor.from_elements %8, %9 {disc.shape_op = true} : tensor<2xindex>
  %11 = "mhlo.dynamic_reshape"(%7, %10) : (tensor<?x?x?x?xf32>, tensor<2xindex>) -> tensor<?x24xf32>
  return %11 : tensor<?x24xf32>
}


// -----

func.func @main(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x6x2xi64>) -> tensor<?x6x?xi32> attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "input0, input1", output_placements = "cpu", outputs = "output0"}} {
  // CHECK: tensor.from_elements
  // CHECK-NOT: disc.device = "gpu"
  // CHECK: mhlo.dynamic_gather
  // CHECK-NOT: disc.device = "gpu"
  // CHECK: ->
  %c1_i64 = arith.constant 1 : i64
  %c1_i64_0 = arith.constant 1 : i64
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c2 : tensor<?x?x?xi32>
  %1 = arith.index_cast %0 : index to i64
  %2 = tensor.from_elements %c1_i64, %c1_i64_0, %1 {disc.shape_op = true} : tensor<3xi64>
  %3 = "mhlo.dynamic_gather"(%arg0, %arg1, %2) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0, 1], index_vector_dim = 2, offset_dims = [2], start_index_map = [0, 1]>, disc.shape_op = true, indices_are_sorted = false} : (tensor<?x?x?xi32>, tensor<?x6x2xi64>, tensor<3xi64>) -> tensor<?x6x?xi32>
  return %3 : tensor<?x6x?xi32>
}


