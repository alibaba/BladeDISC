// RUN: disc-opt -disc-real-dynamic-slice-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<?x?x?xi1>, %arg1: tensor<?x?x?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant dense<1> : tensor<3xi64>
  %cst_4 = arith.constant dense<0> : tensor<2xindex>
  %cst_5 = arith.constant dense<1> : tensor<2xindex>
  %index, %num_output_elements = "mhlo_disc.where"(%arg0) : (tensor<?x?x?xi1>) -> (tensor<?x3xi64>, tensor<1xi64>)
  %extracted = tensor.extract %num_output_elements[%c0] : tensor<1xi64>
  %27 = arith.index_cast %extracted : i64 to index
  %from_elements = tensor.from_elements %27, %c0 : tensor<2xindex>
  %28 = mhlo.real_dynamic_slice %index, %cst_4, %from_elements, %cst_5 : (tensor<?x3xi64>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x3xi64>
  %29 = "mhlo.dynamic_gather"(%arg1, %28, %cst_0) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false} : (tensor<?x?x?xi64>, tensor<?x3xi64>, tensor<3xi64>) -> tensor<?xi64>
  return %29 : tensor<?xi64>
}
