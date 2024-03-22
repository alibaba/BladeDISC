// RUN: disc-opt --disc-collective-ops-rewriter %s | FileCheck %s

// CHECK-LABEL: @reduce_scatter
func.func @reduce_scatter(%arg0: tensor<8x3xf32>) -> (tensor<2x3xf32>) attributes {tf.entry_function = {input_output_alias_outputs = "", input_output_alias_params = "", input_placements = "gpu", output_placements = "gpu"}} {
  // CHECK: %0 = "mhlo_disc.custom_call_v2"(%arg0)
  %3 = "mhlo.reduce_scatter"(%arg0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %5 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %5 : tensor<f32>
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, scatter_dimension = 0 : i64} : (tensor<8x3xf32>) -> tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}

func.func @all_reduce(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<f32>) {
  // CHECK: %0 = "mhlo_disc.custom_call_v2"(%arg1) {call_target_name = "ral_all_reduce", custom_attrs = {is_async = false, reduction_kind = "sum"}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d", replica_groups = dense<> : tensor<0x0xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = "mhlo_disc.custom_call_v2"(%arg0) {call_target_name = "ral_all_reduce", custom_attrs = {is_async = false, reduction_kind = "sum"}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d", replica_groups = dense<> : tensor<0x0xi64>} : (tensor<f32>) -> tensor<f32>
  %0:2 = "mhlo.all_reduce"(%arg1, %arg0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %1 : tensor<f32>
  }) {replica_groups = dense<> : tensor<0x0xi64>} : (tensor<4xf32>, tensor<f32>) -> (tensor<4xf32>, tensor<f32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<f32>
}
