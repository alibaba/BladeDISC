// RUN: disc-opt --disc-mhlo-mark-shape-calc -split-input-file %s | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x8xf32>) -> tensor<?x24xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}} {
  // CHECK: mhlo.constant
  // CHECK: tensor.from_elements
  // CHECK-SAME: disc.shape_op = true
  // CHECK: mhlo.dynamic_broadcast_in_dim
  // CHECK-NOT: disc.shape_op
  // CHECK: tensor.from_elements
  // CHECK-SAME: disc.shape_op = true
  // CHECK: mhlo.dynamic_reshape
  %0 = mhlo.constant dense<[7, 3]> : tensor<2xi32>
  %c0 = arith.constant 0 : index
  %1 = tensor.dim %arg0, %c0 : tensor<?x8xf32>
  %c8 = arith.constant 8 : index
  %c0_0 = arith.constant 0 : index
  %2 = tensor.extract %0[%c0_0] : tensor<2xi32>
  %3 = arith.index_cast %2 : i32 to index
  %c1 = arith.constant 1 : index
  %4 = tensor.extract %0[%c1] : tensor<2xi32>
  %5 = arith.index_cast %4 : i32 to index
  %6 = tensor.from_elements %3, %1, %5, %c8 : tensor<4xindex>
  %7 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %6) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  %8 = arith.muli %3, %1 : index
  %9 = arith.muli %5, %c8 : index
  %10 = tensor.from_elements %8, %9 : tensor<2xindex>
  %11 = "mhlo.dynamic_reshape"(%7, %10) : (tensor<?x?x?x?xf32>, tensor<2xindex>) -> tensor<?x24xf32>
  return %11 : tensor<?x24xf32>
}

// -----

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x6x2xi64>) -> tensor<?x6x?xi32> attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "input0, input1", output_placements = "cpu", outputs = "output0"}} {
  // CHECK: tensor.from_elements
  // CHECK-SAME: disc.shape_op = true
  // CHECK: mhlo.dynamic_gather
  // CHECK-NOT: disc.shape_op
  // CHECK-SAME: ->
  %c1_i64 = arith.constant 1 : i64
  %c1_i64_0 = arith.constant 1 : i64
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c2 : tensor<?x?x?xi32>
  %1 = arith.index_cast %0 : index to i64
  %2 = tensor.from_elements %c1_i64, %c1_i64_0, %1 : tensor<3xi64>
  %3 = "mhlo.dynamic_gather"(%arg0, %arg1, %2) {dimension_numbers = #mhlo.gather<collapsed_slice_dims = [0, 1], index_vector_dim = 2, offset_dims = [2], start_index_map = [0, 1]>, indices_are_sorted = false} : (tensor<?x?x?xi32>, tensor<?x6x2xi64>, tensor<3xi64>) -> tensor<?x6x?xi32>
  return %3 : tensor<?x6x?xi32>
}

// -----

// Test for per_tensor_quantized_dynamic_conv

// CHECK-LABEL: @main
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x?x?x?xi8>, %[[WEIGHT:.*]]: tensor<?x?x?x?xi8>,
// CHECK-SAME: %[[PADDING:.*]]: tensor<4xf32>,
func.func @main(%input: tensor<?x?x?x?xi8>, %weight: tensor<?x?x?x?xi8>, %padding : tensor<4xf32>,
                %input_scale: tensor<f32>, %input_zero_point: tensor<i32>,
                %weight_scale: tensor<f32>, %weight_zero_point: tensor<i32>,
                %result_scale: tensor<f32>, %result_zero_point: tensor<i32>) -> tensor<?x?x?x?xi8>
                    attributes {tf.entry_function = {
                                  input_placements = "cpu,cpu,gpu,cpu,cpu,cpu,cpu,cpu,cpu",
                                  inputs = "input0,input1,input2,input3,input4,input5,input6,input7,input8",
                                  output_placements = "cpu", outputs = "output0"}} {
  // CHECK: %[[RefinedPadding:.*]] = mhlo.convert %[[PADDING]]
  // CHECK-SAME: disc.shape_op = true
  // CHECK: %[[OUT:.*]] = "mhlo_disc.quantized_dynamic_conv"
  // CHECK-SAME: %[[INPUT]], %[[WEIGHT]], %[[RefinedPadding]]
  %refined_padding = mhlo.convert %padding : (tensor<4xf32>) -> tensor<4xi32>
  %out = "mhlo_disc.quantized_dynamic_conv"(%input, %weight, %refined_padding,
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
  return %out : tensor<?x?x?x?xi8>
}

// -----

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK: mhlo.add
  // CHECK-SAME: disc.shape_op = true
  %0 = mhlo.add %arg1, %arg1 : tensor<2xi32>
  %1 = "mhlo_disc.custom_call_v2"(%arg0, %0) {
    call_target_name = "foo",
    custom_attrs = {},
    has_side_effect = false,
    device = "d",
    input_placements = "d,s",
    output_placements = "d",
    expected_input_layouts = "*,*",
    expected_output_layouts = "*",
    input_layouts = "*,*",
    output_layouts = "*"
  } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK-NOT: disc.shape_op = true
  %0 = mhlo.add %arg1, %arg1 : tensor<2xi32>
  %1 = "mhlo_disc.custom_call_v2"(%arg0, %0) {
    call_target_name = "foo",
    custom_attrs = {},
    has_side_effect = false,
    device = "d",
    input_placements = "d,h",
    output_placements = "d",
    expected_input_layouts = "*,*",
    expected_output_layouts = "*",
    input_layouts = "*,*",
    output_layouts = "*"
  } : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x?x?xi64>) -> tensor<?x3xi64> {
  // CHECK: %[[V1:.*]], %[[V1:.*]] = "mhlo_disc.where"(%{{.*}}) : (tensor<?x?x?xi1>) -> (tensor<?x3xi64>, tensor<1xi64>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = mhlo.constant dense<0> : tensor<i64>
  %cst = arith.constant dense<1> : tensor<2xindex>
  %cst_0 = arith.constant dense<0> : tensor<2xindex>
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xi64>
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?xi64>
  %dim_2 = tensor.dim %arg0, %c2 : tensor<?x?x?xi64>
  %from_elements = tensor.from_elements %dim, %dim_1, %dim_2 : tensor<3xindex>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %from_elements) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>, tensor<3xindex>) -> tensor<?x?x?xi64>
  %2 = mhlo.compare  NE, %arg0, %1 : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi1>
  %index, %num_output_elements = "mhlo_disc.where"(%2) : (tensor<?x?x?xi1>) -> (tensor<?x3xi64>, tensor<1xi64>)
  %extracted = tensor.extract %num_output_elements[%c0] : tensor<1xi64>
  %3 = arith.index_cast %extracted : i64 to index
  %from_elements_3 = tensor.from_elements %3, %c3 : tensor<2xindex>
  %4 = mhlo.real_dynamic_slice %index, %cst_0, %from_elements_3, %cst : (tensor<?x3xi64>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x3xi64>
  return %4 : tensor<?x3xi64>
}
