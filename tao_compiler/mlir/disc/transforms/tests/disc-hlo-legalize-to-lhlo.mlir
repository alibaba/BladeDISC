// RUN: disc-opt -disc-hlo-legalize-to-lhlo -hlo-legalize-to-lhlo \
// RUN:  -canonicalize -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: test_disc_mhlo_only_static_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<2x2xf32>, %[[ARG1:.*]]: memref<2x2xf32>)
func.func @test_disc_mhlo_only_static_shape(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  // %[[T0:.*]] = memref.alloc() : memref<2x2xf32>
  // "lmhlo_disc.h2d"(%[[ARG0]], %[[T0]]) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  %0 = "mhlo_disc.h2d"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // %[[T1:.*]] = memref.alloc() : memref<2x2xf32>
  // "lmhlo_disc.d2h"(%[[ARG1]], %[[T1]]) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  %1 = "mhlo_disc.d2h"(%arg1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0, %1 : tensor<2x2xf32>, tensor<2x2xf32>
}

// -----

// CHECK-LABEL: test_disc_mhlo_only_dynamic_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<?x?xf32>)
func.func @test_disc_mhlo_only_dynamic_shape(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[T0:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x?xf32>
  // CHECK: %[[T1:.*]] = memref.dim %[[ARG0]], %[[C1]] : memref<?x?xf32>
  // CHECK: %[[T2:.*]] = memref.alloc(%[[T0]], %[[T1]]) : memref<?x?xf32>
  // CHECK: lmhlo_disc.h2d
  // CHECK-SAME: (%[[ARG0]], %[[T2]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  %0 = "mhlo_disc.h2d"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[T3:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?x?xf32>
  // CHECK: %[[T4:.*]] = memref.dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
  // CHECK: %[[T5:.*]] = memref.alloc(%[[T3]], %[[T4]]) : memref<?x?xf32>
  // CHECK: lmhlo_disc.d2h
  // CHECK-SAME: (%[[ARG1]], %[[T5]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  %1 = "mhlo_disc.d2h"(%arg1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: test_mixed_disc_mhlo_and_mhlo
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>)
func.func @test_mixed_disc_mhlo_and_mhlo(%arg0: tensor<?x?xf32>) -> (tensor<100x100xf32>, tensor<?x?xf32>) {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[T0:.*]] = memref.alloc() : memref<100x100xf32>
  // CHECK: "lmhlo.constant"(%[[T0]]) {value = dense<0.000000e+00> : tensor<100x100xf32>} : (memref<100x100xf32>) -> ()
  // CHECK: %[[T1:.*]] = memref.alloc() : memref<100x100xf32>
  // CHECK: "lmhlo_disc.h2d"(%[[T0]], %[[T1]]) : (memref<100x100xf32>, memref<100x100xf32>) -> ()
  %0 = mhlo.constant dense<0.000000e+00> : tensor<100x100xf32>
  %1 = "mhlo_disc.h2d"(%0) : (tensor<100x100xf32>) -> tensor<100x100xf32>
  // CHECK: %[[T2:.*]] = memref.dim %[[ARG0]], %c0 : memref<?x?xf32>
  // CHECK: %[[T3:.*]] = memref.dim %[[ARG0]], %c1 : memref<?x?xf32>
  // CHECK: %[[T4:.*]] = memref.alloc(%[[T2]], %[[T3]]) : memref<?x?xf32>
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[T4]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: %[[T5:.*]] = memref.alloc(%[[T2]], %[[T3]]) : memref<?x?xf32>
  // CHECK: "lmhlo_disc.d2h"(%[[T4]], %[[T5]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  %2 = "mhlo.abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo_disc.d2h"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0, %3 : tensor<100x100xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: test_topk_custom_call
func.func @test_topk_custom_call(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi64>, %arg2: tensor<index>) -> (tensor<?x?xf32>, tensor<?x?xi64>) {
  // CHECK: lmhlo_disc.custom_call
  %1, %2 = "mhlo_disc.custom_call"(%arg0, %arg1, %arg2) { backend_config = "{\"dimension\": 5}", call_target_name = "topk" } : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<index>) -> (tensor<?x?xf32>, tensor<?x?xi64>)
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xi64>
}

// -----

// CHECK-LABEL: @per_tensor_quantized_dynamic_conv
// CHECK-SAME: %[[INPUT:.*]]: memref<?x?x?x?xi8>, %[[WEIGHT:.*]]: memref<?x?x?x?xi8>,
// CHECK-SAME: %[[PADDING:.*]]: memref<4xi32>,
func.func @per_tensor_quantized_dynamic_conv(
              %input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>, %padding : tensor<4xi32>,
              %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
              %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
              %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8> {
  // CHECK: %[[OUT:.*]] = memref.alloc({{.*}})
  // CHECK-SAME: memref<?x?x?x?xi8>
  // CHECK: "lmhlo_disc.quantized_dynamic_conv"
  // CHECK-SAME: %[[INPUT]], %[[WEIGHT]], %[[PADDING]]
  // CHECK-SAME: %[[OUT]]
  %out = "mhlo_disc.quantized_dynamic_conv"(%input, %weight, %padding,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      dimension_numbers = #mhlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 1,
        input_spatial_dimensions = [2, 3],
        kernel_input_feature_dimension = 1,
        kernel_output_feature_dimension = 0,
        kernel_spatial_dimensions = [2, 3],
        output_batch_dimension = 0,
        output_feature_dimension = 1,
        output_spatial_dimensions = [2, 3]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<3> : tensor<2xi64>
  } : (tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, tensor<4xi32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  // CHECK: return %[[OUT]] : memref<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @per_tensor_quantized_dot_general
// CHECK-SAME: %[[INPUT:.*]]: memref<?x?xi8>, %[[WEIGHT:.*]]: memref<?x?xi8>
func.func @per_tensor_quantized_dot_general(%input: tensor<?x?xi8>, %weight: tensor<?x?xi8>,
                                            %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                            %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                                            %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?xi8> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[M:.*]] = memref.dim %[[INPUT]], %[[C1]] : memref<?x?xi8>
  // CHECK: %[[N:.*]] = memref.dim %[[WEIGHT]], %[[C1]] : memref<?x?xi8>
  // CHECK: %[[RESULT:.*]] = memref.alloc(%[[M]], %[[N]]) :  memref<?x?xi8>
  // CHECK: lmhlo_disc.quantized_dot_general
  // CHECK-SAME: %[[INPUT]], %[[WEIGHT]]
  // CHECK-SAME: %[[RESULT]]
  %out = "mhlo_disc.quantized_dot_general"(%input, %weight,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[]> : tensor<0xi64>,
      use_dynamic = false,
      dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>
  } : (tensor<?x?xi8>, tensor<?x?xi8>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?xi8>
  return %out : tensor<?x?xi8>
}

// -----

// CHECK-LABEL: @per_channel_quantized_dot_general
// CHECK-SAME: %[[INPUT:.*]]: memref<?x?x?x?xi8>, %[[WEIGHT:.*]]: memref<?x?x?x?xi8>
func.func @per_channel_quantized_dot_general(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>,
                                             %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                                             %weight_scale: tensor<?xf32>, %weight_zero_point: tensor<?xi32>,
                                             %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8> {
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index

  // CHECK: %[[T0:.*]] = memref.dim %[[INPUT]], %[[C0]] : memref<?x?x?x?xi8>
  // CHECK: %[[T1:.*]] = memref.dim %[[INPUT]], %[[C1]] : memref<?x?x?x?xi8>
  // CHECK: %[[T2:.*]] = memref.dim %[[INPUT]], %[[C2]] : memref<?x?x?x?xi8>
  // CHECK: %[[T3:.*]] = memref.dim %[[WEIGHT]], %[[C3]] : memref<?x?x?x?xi8>

  // CHECK: %[[RESULT:.*]] = memref.alloc(%[[T0]], %[[T1]], %[[T2]], %[[T3]]) : memref<?x?x?x?xi8>
  // CHECK: lmhlo_disc.quantized_dot_general
  // CHECK-SAME: %[[INPUT]], %[[WEIGHT]]
  // CHECK-SAME: %[[RESULT]]
  %out = "mhlo_disc.quantized_dot_general"(%input, %weight,
                                           %input_scale, %input_zero_point,
                                           %weight_scale, %weight_zero_point,
                                           %result_scale, %result_zero_point) {
      use_symmetric = true,
      axis = dense<[3]> : tensor<1xi64>,
      use_dynamic = false,
      dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1],
                                        lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>
  } : (tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>,
       tensor<f32>, tensor<i32>,
       tensor<?xf32>, tensor<?xi32>,
       tensor<f32>, tensor<i32>) -> tensor<?x?x?x?xi8>
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @custom_call_v2_op
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<2xi32>
func.func @custom_call_v2_op(
    %arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32>
        attributes {
            tf.entry_function = {
                input_placements = "cpu,cpu",
                inputs = "input0,input1",
                output_placements = "cpu", outputs = "output0"}} {
  // CHECK: %[[T0:.*]] = "lmhlo_disc.custom_call_v2"(%[[ARG0]], %[[ARG1]])
  // return %[[T0]]
  %1 = "mhlo_disc.custom_call_v2"(%arg0, %arg1) {
    call_target_name = "foo",
    custom_attrs = {},
    has_side_effect = false,
    device = "h",
    input_placements = "h,h",
    output_placements = "h",
    expected_input_layouts = "AB,A",
    expected_output_layouts = "AB",
    input_layouts = "AB,A",
    output_layouts = "AB"
  } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}