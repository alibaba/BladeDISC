// RUN: disc-opt -split-input-file -disc-shape-propagate -canonicalize %s | FileCheck %s

// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101xi64>, %arg1: tensor<4x101xi64>) -> tensor<4x101xi1> attributes{tf.entry_function = {input_dynamic_dims = "0:1|1:1"}}{
  // CHECK: %0 = mhlo.compare  LT, %arg0, %arg1 : (tensor<4x?xi64>, tensor<4x?xi64>) -> tensor<4x?xi1>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<4x101xi64>, tensor<4x101xi64>) -> tensor<4x101xi1>
  // CHECK: return %0 : tensor<4x?xi1>
  return %0 : tensor<4x101xi1>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101xi64>) -> tensor<4x101xi1> attributes{tf.entry_function = {input_dynamic_dims = "0:1"}}{
  // CHECK: %1 = shape.shape_of %arg0 : tensor<4x?xi64> -> tensor<2xindex>
  // CHECK: %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>, tensor<2xindex>) -> tensor<4x?xi64>
  %0 = mhlo.constant dense<0> : tensor<4x101xi64>
  %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<4x101xi64>, tensor<4x101xi64>) -> tensor<4x101xi1>
  return %1 : tensor<4x101xi1>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101x32x128xbf16>) -> tensor<4x32x101x128xbf16> attributes{tf.entry_function = {input_dynamic_dims = "0:0,1"}}{
  // CHECK: %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[4,32,101,128]{3,1,2,0}"} : (tensor<?x?x32x128xbf16>) -> tensor<?x32x?x128xbf16>
  %1 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>, result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[4,32,101,128]{3,1,2,0}"} : (tensor<4x101x32x128xbf16>) -> tensor<4x32x101x128xbf16>
  return %1 : tensor<4x32x101x128xbf16>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101x4096xf32>) -> tensor<4x101xf32> attributes{tf.entry_function = {input_dynamic_dims = "0:0,1"}}{
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [2] : (tensor<?x?x4096xf32>, tensor<f32>) -> tensor<?x?xf32>                                                                                                                                                           
  // CHECK:  reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {                                                                                                                                                                                                                         
  // CHECK:  %2 = mhlo.add %arg1, %arg2 : tensor<f32>                                                                                                                                                                                                                                  
  // CHECK:  mhlo.return %2 : tensor<f32>                                                                                                                                                                                                                                              
  // CHECK: }
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [2] : (tensor<4x101x4096xf32>, tensor<f32>) -> tensor<4x101xf32>
        reducer(%arg216: tensor<f32>, %arg217: tensor<f32>)  {
          %2869 = mhlo.add %arg216, %arg217 : tensor<f32>
          mhlo.return %2869 : tensor<f32>
       }
  return %2 : tensor<4x101xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101x4096xf32>) -> tensor<4x101xf32> attributes{tf.entry_function = {input_dynamic_dims = "0:1"}}{
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [2] : (tensor<4x?x4096xf32>, tensor<f32>) -> tensor<4x?xf32>                                                                                                                                                           
  // CHECK:  reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {                                                                                                                                                                                                                         
  // CHECK:  %2 = mhlo.add %arg1, %arg2 : tensor<f32>                                                                                                                                                                                                                                  
  // CHECK:  mhlo.return %2 : tensor<f32>                                                                                                                                                                                                                                              
  // CHECK: }
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [2] : (tensor<4x101x4096xf32>, tensor<f32>) -> tensor<4x101xf32>
        reducer(%arg216: tensor<f32>, %arg217: tensor<f32>)  {
          %2869 = mhlo.add %arg216, %arg217 : tensor<f32>
          mhlo.return %2869 : tensor<f32>
       }
  return %2 : tensor<4x101xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x101x4096xf32>) -> tensor<4x101xf32> attributes{tf.entry_function = {input_dynamic_dims = "0:2"}}{
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [2] : (tensor<4x101x?xf32>, tensor<f32>) -> tensor<4x101xf32>                                                                                                                                                          
  // CHECK:   reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {                                                                                                                                                                                                                         
  // CHECK:   %2 = mhlo.add %arg1, %arg2 : tensor<f32>
  // CHECK:   mhlo.return %2 : tensor<f32>
  // CHECK: }
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [2] : (tensor<4x101x4096xf32>, tensor<f32>) -> tensor<4x101xf32>
        reducer(%arg216: tensor<f32>, %arg217: tensor<f32>)  {
          %2869 = mhlo.add %arg216, %arg217 : tensor<f32>
          mhlo.return %2869 : tensor<f32>
       }
  return %2 : tensor<4x101xf32>

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x32x101x64xbf16>, %arg1: tensor<4x32x101x64xbf16>) -> tensor<4x32x101x128xbf16> attributes{tf.entry_function = {input_dynamic_dims = "0:3"}}{
  // CHECK: %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<4x32x101x?xbf16>, tensor<4x32x101x64xbf16>) -> tensor<4x32x101x?xbf16>
  %1 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<4x32x101x64xbf16>, tensor<4x32x101x64xbf16>) -> tensor<4x32x101x128xbf16>
  return %1 : tensor<4x32x101x128xbf16>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x32x101x64xbf16>, %arg1: tensor<4x32x101x64xbf16>) -> tensor<4x32x101x128xbf16> attributes{tf.entry_function = {input_dynamic_dims = "1:1"}}{
  // CHECK: %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<4x32x101x64xbf16>, tensor<4x?x101x64xbf16>) -> tensor<4x32x101x128xbf16>
  %1 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<4x32x101x64xbf16>, tensor<4x32x101x64xbf16>) -> tensor<4x32x101x128xbf16>
  return %1 : tensor<4x32x101x128xbf16>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x32x101x64xbf16>, %arg1: tensor<4x32x101x64xbf16>) -> tensor<4x32x101x128xbf16> attributes{tf.entry_function = {input_dynamic_dims = "1:1|0:1"}}{
  // CHECK: %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<4x?x101x64xbf16>, tensor<4x?x101x64xbf16>) -> tensor<4x?x101x128xbf16>
  %1 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<4x32x101x64xbf16>, tensor<4x32x101x64xbf16>) -> tensor<4x32x101x128xbf16>
  return %1 : tensor<4x32x101x128xbf16>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<32001x4096xf32>, %arg1: tensor<404x1xi64>, %arg2: tensor<404x4096xf32>) -> tensor<32001x4096xf32> attributes{tf.entry_function = {input_dynamic_dims = "1:0|2:0"}}{
  
  // CHECK: %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK:   ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:     %1 = mhlo.add %arg3, %arg4 : tensor<f32>
  // CHECK:     mhlo.return %1 : tensor<f32>
  // CHECK:   }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<32001x4096xf32>, tensor<?x1xi64>, tensor<?x4096
  // CHECK: xf32>) -> tensor<?x4096xf32>
  %1 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
      ^bb0(%arg216: tensor<f32>, %arg217: tensor<f32>):
        %2869 = mhlo.add %arg216, %arg217 : tensor<f32>
        mhlo.return %2869 : tensor<f32>
      }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<32001x4096xf32>, tensor<404x1xi64>, tensor<404x4096xf32>) -> tensor<32001x4096xf32>
  return %1 : tensor<32001x4096xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<32001x4096xf32>, %arg1: tensor<4x101x1xi64>) -> tensor<4x101x4096xf32> attributes{tf.entry_function = {input_dynamic_dims = "1:0|0:1"}}{
  // CHECK: %c1_i64 = arith.constant 1 : i64
  // CHECK: %c1 = arith.constant 1 : index
  // CHECK: %dim = tensor.dim %arg0, %c1 : tensor<32001x?xf32>
  // CHECK: %0 = arith.index_cast %dim : index to i64
  // CHECK: %from_elements = tensor.from_elements %c1_i64, %0 : tensor<2xi64>
  // CHECK: %1 = "mhlo.dynamic_gather"(%arg0, %arg1, %from_elements) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<32001x?xf32>, tensor<?x101x1xi64>, tensor<2xi64>) -> tensor<?x101x?xf32>
  %1 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 4096]> : tensor<2xi64>} : (tensor<32001x4096xf32>, tensor<4x101x1xi64>) -> tensor<4x101x4096xf32>
  return %1 : tensor<4x101x4096xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<32001x4096xf32>, %arg1: tensor<4x101x1xi64>) -> tensor<4x101x4096xf32> attributes{tf.entry_function = {input_dynamic_dims = "1:0,1"}}{
  // CHECK: %cst = arith.constant dense<[1, 4096]> : tensor<2xi64>
  // CHECK: %0 = "mhlo.dynamic_gather"(%arg0, %arg1, %cst) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<32001x4096xf32>, tensor<?x?x1xi64>, tensor<2xi64>) -> tensor<?x?x?xf32>
  %1 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 4096]> : tensor<2xi64>} : (tensor<32001x4096xf32>, tensor<4x101x1xi64>) -> tensor<4x101x4096xf32>
  return %1 : tensor<4x101x4096xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<32001x4096xf32>, %arg1: tensor<4x101x1xi64>) -> tensor<4x101x4096xf32> attributes{tf.entry_function = {input_dynamic_dims = "1:0"}}{
  // CHECK: %cst = arith.constant dense<[1, 4096]> : tensor<2xi64>
  // CHECK: %0 = "mhlo.dynamic_gather"(%arg0, %arg1, %cst) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<32001x4096xf32>, tensor<?x101x1xi64>, tensor<2xi64>) -> tensor<?x101x?xf32>
  %1 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 4096]> : tensor<2xi64>} : (tensor<32001x4096xf32>, tensor<4x101x1xi64>) -> tensor<4x101x4096xf32>
  return %1 : tensor<4x101x4096xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<32001x4096xf32>, %arg1: tensor<4x101x1xi64>) -> tensor<4x101x2048xf32> attributes{tf.entry_function = {input_dynamic_dims = "1:0"}}{
  // CHECK: %cst = arith.constant dense<[1, 2048]> : tensor<2xi64>
  // CHECK: %0 = "mhlo.dynamic_gather"(%arg0, %arg1, %cst) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<32001x4096xf32>, tensor<?x101x1xi64>, tensor<2xi64>) -> tensor<?x101x2048xf32>
  %1 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 2048]> : tensor<2xi64>} : (tensor<32001x4096xf32>, tensor<4x101x1xi64>) -> tensor<4x101x2048xf32>
  return %1 : tensor<4x101x2048xf32>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<32001x4096xf32>, %arg1: tensor<4x101x1xi64>) -> tensor<4x101x2048xf32> attributes{tf.entry_function = {input_dynamic_dims = "1:0|0:1"}}{
  // CHECK: %cst = arith.constant dense<[1, 2048]> : tensor<2xi64>
  // CHECK: %0 = "mhlo.dynamic_gather"(%arg0, %arg1, %cst) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<32001x?xf32>, tensor<?x101x1xi64>, tensor<2xi64>) -> tensor<?x101x2048xf32>
  %1 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 2048]> : tensor<2xi64>} : (tensor<32001x4096xf32>, tensor<4x101x1xi64>) -> tensor<4x101x2048xf32>
  return %1 : tensor<4x101x2048xf32>
}


// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<4x32x101x128xbf16>) -> tensor<4x32x101x64xbf16> attributes{tf.entry_function = {input_dynamic_dims = "0:2"}}{
  // %0 = mhlo.real_dynamic_slice %arg0, %from_elements, %from_elements_7, %from_elements_6 : (tensor<4x32x?x128xbf16>, tensor<4xindex>, tensor<4xindex>, tensor<4xindex>) -> tensor<4x32x?x64xbf16
  %140 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 32, 101, 64]> : tensor<4xi64>, start_indices = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<4x32x101x128xbf16>) -> tensor<4x32x101x64xbf16>
  return %140 : tensor<4x32x101x64xbf16>
}

// -----
// CHECK-LABEL: main
func.func @main(%arg0: tensor<1x101x128xbf16>) -> tensor<101x128xbf16> attributes{tf.entry_function = {input_dynamic_dims = "0:1"}}{
  // CHECK: %0 = mhlo.dynamic_reshape %arg0, %from_elements : (tensor<1x?x128xbf16>, tensor<2xindex>) -> tensor<?x128xbf16>
  %0 = mhlo.reshape %arg0: (tensor<1x101x128xbf16>) -> tensor<101x128xbf16>
  return %0: tensor<101x128xbf16>
}