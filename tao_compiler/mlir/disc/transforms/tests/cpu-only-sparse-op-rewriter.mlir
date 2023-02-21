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
