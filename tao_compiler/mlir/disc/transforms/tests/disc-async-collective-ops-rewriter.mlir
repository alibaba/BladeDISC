// RUN: ENABLE_ASYNC_COLLECTIVE=true disc-opt --disc-collective-ops-rewriter -split-input-file %s -o - | FileCheck %s
func.func @async_all_reduce(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<f32>) {
  // CHECK: %0 = "mhlo_disc.custom_call_v2"(%arg1) {call_target_name = "ral_all_reduce", custom_attrs = {async_token_key = {{.*}} : i64, is_async = true, reduction_kind = "sum"}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d", replica_groups = dense<> : tensor<0x0xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = "mhlo_disc.custom_call_v2"(%0) {call_target_name = "ral_async_collective_done", custom_attrs = {async_token_key = {{.*}} : i64, is_async = true, reduction_kind = "sum"}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d", replica_groups = dense<> : tensor<0x0xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = "mhlo_disc.custom_call_v2"(%arg0) {call_target_name = "ral_all_reduce", custom_attrs = {async_token_key = {{.*}} : i64, is_async = true, reduction_kind = "sum"}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d", replica_groups = dense<> : tensor<0x0xi64>} : (tensor<f32>) -> tensor<f32>
  // CHECK: %3 = "mhlo_disc.custom_call_v2"(%2) {call_target_name = "ral_async_collective_done", custom_attrs = {async_token_key = {{.*}} : i64, is_async = true, reduction_kind = "sum"}, device = "d", expected_input_layouts = "*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*", output_placements = "d", replica_groups = dense<> : tensor<0x0xi64>} : (tensor<f32>) -> tensor<f32>
  // CHECK: return %1, %3 : tensor<4xf32>, tensor<f32>
  %0:2 = "mhlo.all_reduce"(%arg1, %arg0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %1 : tensor<f32>
  }) {replica_groups = dense<> : tensor<0x0xi64>} : (tensor<4xf32>, tensor<f32>) -> (tensor<4xf32>, tensor<f32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<f32>
}
