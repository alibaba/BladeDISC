// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<()[s0] -> (0, s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 6)>
#map3 = affine_map<(d0) -> (-d0 + 6)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map5 = affine_map<(d0) -> (-d0 + 16)>
#map6 = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
#map7 = affine_map<(d0) -> (-d0 + 2)>

module {
  // CHECK-LABEL: @matmul_nn
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>)
  func.func @matmul_nn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c6 = arith.constant 6 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    // CHECK: %[[PACKED:.*]] = disc_linalg_ext.multi_level_pack %[[ARG1]]
    // CHECK-SAME: tile_levels = [1, 1] tile_sizes = [2, 16] permutation = [0, 2, 3, 1]
    // CHECK-SAME: (tensor<?x?xf32> tensor<?x?x16x2xf32>) -> tensor<?x?x16x2xf32>
    // CHECK: %[[RES:.*]] = scf.foreach_thread
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: %[[SLICE_FROM_PACKED:.*]] = tensor.extract_slice %[[PACKED]]
    // CHECK: %[[TRANSPOSE:.*]] = linalg.generic
    // CHECK-SAME: %[[SLICE_FROM_PACKED]]
    // CHECK: linalg.matmul
    // CHECK-SAME: %[[TRANSPOSE]]
    %0 = scf.foreach_thread (%arg3, %arg4) in (%c1, %c1) shared_outs(%arg5 = %arg2) -> (tensor<?x?xf32>) {
      %1 = affine.max #map()[%dim]
      %2 = affine.max #map()[%dim_1]
      %3 = affine.apply #map1(%arg3)[%dim]
      %4 = affine.apply #map1(%arg4)[%dim_1]
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [%1, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_2 = tensor.extract_slice %arg5[%3, %4] [%1, %2] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %5 = scf.for %arg6 = %c0 to %1 step %c6 iter_args(%arg7 = %extracted_slice_2) -> (tensor<?x?xf32>) {
        %6 = affine.min #map2(%arg6)[%1]
        %7 = affine.apply #map3(%6)
        %8 = scf.for %arg8 = %c0 to %2 step %c16 iter_args(%arg9 = %arg7) -> (tensor<?x?xf32>) {
          %9 = affine.min #map4(%arg8)[%2]
          %extracted_slice_3 = tensor.extract_slice %arg9[%arg6, %arg8] [%6, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %10 = linalg.fill ins(%cst : f32) outs(%extracted_slice_3 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %11 = arith.addi %arg8, %4 : index
          %12 = affine.apply #map5(%9)
          %13 = scf.for %arg10 = %c0 to %dim_0 step %c2 iter_args(%arg11 = %10) -> (tensor<?x?xf32>) {
            %14 = affine.min #map6(%arg10)[%dim_0]
            %extracted_slice_4 = tensor.extract_slice %extracted_slice[%arg6, %arg10] [%6, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_5 = tensor.extract_slice %arg1[%arg10, %11] [%14, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_6 = tensor.extract_slice %arg11[0, 0] [%6, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %15 = affine.apply #map7(%14)
            %padded = tensor.pad %extracted_slice_4 nofold low[%c0, %c0] high[%7, %15] {
            ^bb0(%arg12: index, %arg13: index):
              tensor.yield %cst : f32
            } : tensor<?x?xf32> to tensor<6x2xf32>
            %padded_7 = tensor.pad %extracted_slice_5 low[%c0, %c0] high[%15, %12] {
            ^bb0(%arg12: index, %arg13: index):
              tensor.yield %cst : f32
            } : tensor<?x?xf32> to tensor<2x16xf32>
            %padded_8 = tensor.pad %extracted_slice_6 low[%c0, %c0] high[%7, %12] {
            ^bb0(%arg12: index, %arg13: index):
              tensor.yield %cst : f32
            } : tensor<?x?xf32> to tensor<6x16xf32>
            %16 = linalg.matmul ins(%padded, %padded_7 : tensor<6x2xf32>, tensor<2x16xf32>) outs(%padded_8 : tensor<6x16xf32>) -> tensor<6x16xf32>
            %extracted_slice_9 = tensor.extract_slice %16[0, 0] [%6, %9] [1, 1] : tensor<6x16xf32> to tensor<?x?xf32>
            %inserted_slice_10 = tensor.insert_slice %extracted_slice_9 into %arg11[0, 0] [%6, %9] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %inserted_slice_10 : tensor<?x?xf32>
          }
          %inserted_slice = tensor.insert_slice %13 into %arg9[%arg6, %arg8] [%6, %9] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice : tensor<?x?xf32>
        }
        scf.yield %8 : tensor<?x?xf32>
      }
      scf.foreach_thread.perform_concurrently {
        tensor.parallel_insert_slice %5 into %arg5[%3, %4] [%1, %2] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      }
    } {thread_dim_mapping = []}
    return %0 : tensor<?x?xf32>
  }
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %foreach_op = transform.structured.match ops{["scf.foreach_thread"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %pad_for_weight = get_producer_of_operand %matmul[1] : (!pdl.operation) -> !pdl.operation
  transform.disc.cache_read {padded} %pad_for_weight at %foreach_op with tile_levels = [1, 1] tile_sizes = [2, 16] permutation = [0, 2, 3, 1]
}