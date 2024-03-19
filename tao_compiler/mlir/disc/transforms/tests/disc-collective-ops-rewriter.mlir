// RUN: disc-opt -disc-collecitve-ops-rewriter %s | FileCheck %s

func.func @main(%arg0: tensor<8x3xf32>) -> (tensor<2x3xf32>) attributes {tf.entry_function = {input_output_alias_outputs = "", input_output_alias_params = "", input_placements = "gpu", output_placements = "gpu"}} {
  // CHECK: %[[T0:.*]] = "mhlo_disc.custom_call_v2"(%arg0) {call_target_name = "ral_reduce_scatter", custom_attrs = {reduction_kind = "sum", replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, scatter_dimension = 0 : i64}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d"} : (tensor<8x3xf32>) -> tensor<2x3xf32>
  %3 = "mhlo.reduce_scatter"(%arg0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %5 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %5 : tensor<f32>
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, scatter_dimension = 0 : i64} : (tensor<8x3xf32>) -> tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}