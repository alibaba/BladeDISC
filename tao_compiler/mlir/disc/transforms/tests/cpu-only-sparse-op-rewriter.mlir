// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt %s -disc-sparse-op-rewriter -split-input-file | FileCheck %s

// CHECK-LABEL: @sparse_reshape_elimination
func.func @sparse_reshape_elimination(
    %arg24: tensor<2xi64>,
    %arg86: tensor<?x?xi64>,
    %1273:  tensor<?xi64>,
    %1274:  tensor<?xi64>
    ) -> (
    tensor<?x2xi64>,
    tensor<?xi64>
  ) {
  // CHECK-NOT: mhlo_disc.sparse_reshape
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant dense<1> : tensor<1xi64>
  %cst_1 = arith.constant dense<1> : tensor<1xi32>
  %cst_2 = arith.constant dense<[1, 2]> : tensor<2xi64>
  %2 = mhlo.constant dense<1> : tensor<i64>
  %4 = mhlo.constant dense<0> : tensor<i64>
  %5 = mhlo.constant dense<1> : tensor<i32>
  %dim_31 = tensor.dim %arg24, %c0 : tensor<2xi64>
  %1275 = "mhlo.dynamic_gather"(%arg24, %5, %cst_1) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = false} : (tensor<2xi64>, tensor<i32>, tensor<1xi32>) -> tensor<i64>
  %1276 = arith.index_cast %dim_31 : index to i64
  %1277 = arith.subi %1276, %c1_i64 : i64
  %1278 = arith.minsi %1277, %c0_i64 : i64
  %1279 = arith.addi %1278, %c1_i64 : i64
  %from_elements_779 = tensor.from_elements %1278 : tensor<1xi64>
  %from_elements_780 = tensor.from_elements %1279 : tensor<1xi64>
  %1280 = mhlo.real_dynamic_slice %arg24, %from_elements_779, %from_elements_780, %cst {kDiscSliceOpStaticKnownInfo = {limit_indices = dense<-2> : tensor<1xi64>, start_indices = dense<-2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}} : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %1281 = mhlo.reduce(%1280 init: %2) applies mhlo.multiply across dimensions = [0] : (tensor<1xi64>, tensor<i64>) -> tensor<i64>
  %1282 = mhlo.reshape %1281 : (tensor<i64>) -> tensor<1xi64>
  %1283 = mhlo.reshape %1275 : (tensor<i64>) -> tensor<1xi64>
  %1284 = "mhlo.concatenate"(%1282, %1283) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %output_indices_781, %output_shape_782 = "mhlo_disc.sparse_reshape"(%arg86, %arg24, %1284) : (tensor<?x?xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<?x2xi64>, tensor<2xi64>)
  %1285 = "mhlo.dynamic_gather"(%output_indices_781, %1273, %cst_2) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<?x2xi64>, tensor<?xi64>, tensor<2xi64>) -> tensor<?x2xi64>
  %output_indices_783, %output_values_784, %empty_row_indicator_785, %reverse_index_map_786, %output_elements_787 = "mhlo_disc.sparse_fill_empty_rows"(%1285, %1274, %output_shape_782, %4) : (tensor<?x2xi64>, tensor<?xi64>, tensor<2xi64>, tensor<i64>) -> (tensor<?x2xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<1xi64>)
  return %output_indices_783, %output_values_784 : tensor<?x2xi64>, tensor<?xi64>
}

// -----

// CHECK-LABEL: @sparse_segment_reduction_rewrite
func.func @sparse_segment_reduction_rewrite(%arg0: tensor<?x2xi64>, %arg1: tensor<?xi64>, %arg2: tensor<2xi64>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: index, %arg6: tensor<?x?xf32>, %arg7: tensor<2xindex>, %arg8: tensor<3xindex>, %arg9: tensor<4xindex>) -> (tensor<?x?xf32>, tensor<?x1x?xf32>) {
  // CHECK: mhlo_disc.sparse_segment_reduction_with_empty_rows
  // CHECK-NOT: mhlo_disc.sparse_fill_empty_rows
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %cst = arith.constant dense<0> : tensor<2xi32>
  %cst_0 = arith.constant dense<1> : tensor<2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2xindex>
  %cst_2 = arith.constant dense<1> : tensor<2xindex>
  %cst_3 = arith.constant dense<0> : tensor<1xindex>
  %cst_4 = arith.constant dense<1> : tensor<1xindex>
  %false = arith.constant false
  %dim = tensor.dim %arg4, %c1 : tensor<?x?xf32>
  %output_indices, %output_values, %empty_row_indicator, %reverse_index_map, %output_elements = "mhlo_disc.sparse_fill_empty_rows"(%arg0, %arg1, %arg2, %1) : (tensor<?x2xi64>, tensor<?xi64>, tensor<2xi64>, tensor<i64>) -> (tensor<?x2xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>, tensor<1xi64>)
  %extracted = tensor.extract %output_elements[%c0] : tensor<1xi64>
  %2 = arith.index_cast %extracted : i64 to index
  %from_elements = tensor.from_elements %2, %c2 : tensor<2xindex>
  %3 = mhlo.real_dynamic_slice %output_indices, %cst_1, %from_elements, %cst_2 {kDiscSliceOpStaticKnownInfo = {limit_indices = dense<[-2, -1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}} : (tensor<?x2xi64>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x2xi64>
  %from_elements_5 = tensor.from_elements %2 : tensor<1xindex>
  %4 = mhlo.real_dynamic_slice %output_values, %cst_3, %from_elements_5, %cst_4 {kDiscSliceOpStaticKnownInfo = {limit_indices = dense<-2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}} : (tensor<?xi64>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xi64>
  %5 = arith.index_cast %2 : index to i32
  %6 = arith.cmpi slt, %5, %c0_i32 : i32
  %7 = arith.cmpi eq, %6, %false : i1
  %8 = arith.cmpi ne, %5, %c0_i32 : i32
  %9 = arith.andi %8, %7 : i1
  %10 = arith.select %9, %5, %c0_i32 : i32
  %11 = arith.select %6, %c0_i32, %5 : i32
  %12 = arith.cmpi slt, %11, %5 : i32
  %13 = arith.select %12, %11, %5 : i32
  %from_elements_6 = tensor.from_elements %13, %c1_i32 : tensor<2xi32>
  %14 = mhlo.real_dynamic_slice %3, %cst, %from_elements_6, %cst_0 {kDiscSliceOpStaticKnownInfo = {limit_indices = dense<[-2, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}} : (tensor<?x2xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
  %from_elements_7 = tensor.from_elements %10 : tensor<1xi32>
  %15 = mhlo.dynamic_reshape %14, %from_elements_7 : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64>
  %16 = "mhlo_disc.sparse_segment_reduction"(%arg4, %4, %15) {reduction_mode = 1 : i64} : (tensor<?x?xf32>, tensor<?xi64>, tensor<?xi64>) -> tensor<?x?xf32>
  %from_elements_8 = tensor.from_elements %arg5, %dim : tensor<2xindex>
  %from_elements_9 = tensor.from_elements %c1, %arg5, %dim, %c1 : tensor<4xindex>
  %17 = "mhlo.dynamic_broadcast_in_dim"(%empty_row_indicator, %from_elements_9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi1>, tensor<4xindex>) -> tensor<1x?x?x1xi1>
  %18 = mhlo.dynamic_reshape %17, %from_elements_8 : (tensor<1x?x?x1xi1>, tensor<2xindex>) -> tensor<?x?xi1>
  %19 = "mhlo.dynamic_broadcast_in_dim"(%0, %from_elements_8) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  %20 = mhlo.select %18, %19, %16 : tensor<?x?xi1>, tensor<?x?xf32>
  %21 = "mhlo_disc.sparse_segment_reduction"(%arg3, %4, %15) {reduction_mode = 0 : i64} : (tensor<?x?xf32>, tensor<?xi64>, tensor<?xi64>) -> tensor<?x?xf32>
  %22 = "mhlo.dynamic_broadcast_in_dim"(%empty_row_indicator, %arg9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi1>, tensor<4xindex>) -> tensor<1x?x?x1xi1>
  %23 = mhlo.dynamic_reshape %22, %arg7 : (tensor<1x?x?x1xi1>, tensor<2xindex>) -> tensor<?x?xi1>
  %24 = mhlo.select %23, %arg6, %21 : tensor<?x?xi1>, tensor<?x?xf32>
  %25 = mhlo.dynamic_reshape %24, %arg8 : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x1x?xf32>
  return %20, %25 : tensor<?x?xf32>, tensor<?x1x?xf32>
}
