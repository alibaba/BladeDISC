// RUN: disc-opt -split-input-file hlo-legalize-to-lhlo \
// RUN:  -canonicalize %s -o - | FileCheck %s


// CHECK-LABEL: @test_scatterop_lowering(
// CHECK-SAME: %[[ARG0:.*]]: memref<32000x4096xf32>, %[[ARG1:.*]]: memref<8192x1xi64>, %[[ARG2:.*]]: memref<8192x4096xf32>) -> memref<32000x4096xf32> {
func.func @test_scatterop_lowering(%arg0: tensor<32000x4096xf32>, %arg1: tensor<8192x1xi64>, %arg2: tensor<8192x4096xf32>) -> tensor<32000x4096xf32>
{
    // CHECK: %alloc = memref.alloc() : memref<32000x4096xf32>
    %2 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
      // CHECK: "lmhlo.scatter"(%arg0, %arg1, %arg2, %alloc) ({
      ^bb0(%arg143: tensor<f32>, %arg144: tensor<f32>):
      // CHECK-NEXT:^bb0(%arg3: memref<f32>, %arg4: memref<f32>, %arg5: memref<f32>):
        %1 = mhlo.add %arg143, %arg144 : tensor<f32>
        // CHECK-NEXT:  %alloc_0 = memref.alloc() : memref<f32>
        // CHECK-NEXT:  "lmhlo.add"(%arg3, %arg4, %alloc_0) : (memref<f32>, memref<f32>, memref<f32>) -> ()
        // CHECK-NEXT:  "lmhlo.copy"(%alloc_0, %arg5) : (memref<f32>, memref<f32>) -> ()
        mhlo.return %1 : tensor<f32>
        // CHECK-NEXT:  "lmhlo.terminator"() : () -> ()
      }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<32000x4096xf32>, tensor<8192x1xi64>, tensor<8192x4096xf32>) -> tensor<32000x4096xf32>
      // CHECK-NEXT:}) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<32000x4096xf32>, memref<8192x1xi64>, memref<8192x4096xf32>, memref<32000x4096xf32>) -> ()
    return %2 : tensor<32000x4096xf32>
    // CHECK: return %alloc : memref<32000x4096xf32>
}