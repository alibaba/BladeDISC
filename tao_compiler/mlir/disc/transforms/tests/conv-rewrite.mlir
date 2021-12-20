// RUN: disc-opt -disc-conv-rewriter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @dynamic_conv
func @dynamic_conv(%arg0: tensor<?x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<?x8x7x16xf32> {
  %c1 = constant 1 : index
  %0 = tensor.dim %arg0, %c1 : tensor<?x32x32x6xf32>
  %1 = index_cast %0 : index to i32
  %c0 = constant 0 : index
  %2 = tensor.dim %arg1, %c0 : tensor<3x3x3x16xf32>
  %3 = index_cast %2 : index to i32
  %c0_i32 = constant 0 : i32
  %c1_i32 = constant 1 : i32
  %c2_i32 = constant 2 : i32
  %c4_i32 = constant 4 : i32
  %c2_i32_0 = constant 2 : i32
  %4 = subi %3, %c1_i32 : i32
  %5 = muli %c2_i32_0, %4 : i32
  %6 = addi %c1_i32, %5 : i32
  %7 = subi %c4_i32, %c1_i32 : i32
  %8 = addi %1, %7 : i32
  %9 = divi_unsigned %8, %c4_i32 : i32
  %10 = subi %9, %c1_i32 : i32
  %11 = muli %c4_i32, %10 : i32
  %12 = addi %6, %11 : i32
  %13 = subi %12, %1 : i32
  %14 = cmpi sge, %13, %c0_i32 : i32
  %15 = select %14, %13, %c0_i32 : i32
  %16 = divi_unsigned %15, %c2_i32 : i32
  %17 = subi %15, %16 : i32
  %c2 = constant 2 : index
  %18 = tensor.dim %arg0, %c2 : tensor<?x32x32x6xf32>
  %19 = index_cast %18 : index to i32
  %c1_1 = constant 1 : index
  %20 = tensor.dim %arg1, %c1_1 : tensor<3x3x3x16xf32>
  %21 = index_cast %20 : index to i32
  %c0_i32_2 = constant 0 : i32
  %c1_i32_3 = constant 1 : i32
  %c2_i32_4 = constant 2 : i32
  %c5_i32 = constant 5 : i32
  %c3_i32 = constant 3 : i32
  %22 = subi %21, %c1_i32_3 : i32
  %23 = muli %c3_i32, %22 : i32
  %24 = addi %c1_i32_3, %23 : i32
  %25 = subi %c5_i32, %c1_i32_3 : i32
  %26 = addi %19, %25 : i32
  %27 = divi_unsigned %26, %c5_i32 : i32
  %28 = subi %27, %c1_i32_3 : i32
  %29 = muli %c5_i32, %28 : i32
  %30 = addi %24, %29 : i32
  %31 = subi %30, %19 : i32
  %32 = cmpi sge, %31, %c0_i32_2 : i32
  %33 = select %32, %31, %c0_i32_2 : i32
  %34 = divi_unsigned %33, %c2_i32_4 : i32
  %35 = subi %33, %34 : i32
  // CHECK: "mhlo.transpose"(%arg0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x32x32x6xf32>) -> tensor<?x6x32x32xf32>
  // CHECK: "mhlo.transpose"(%arg1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x3x16xf32>) -> tensor<16x3x3x3xf32>
  // CHECK: "mhlo.dynamic_conv"({{.*}}) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 1 : i64, input_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>, kernel_input_feature_dimension = 1 : i64, kernel_output_feature_dimension = 0 : i64, kernel_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 1 : i64, output_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>}, feature_group_count = 2 : i64, rhs_dilation = dense<[2, 3]> : tensor<2xi64>, window_strides = dense<[4, 5]> : tensor<2xi64>} : (tensor<?x6x32x32xf32>, tensor<16x3x3x3xf32>, tensor<4xi32>) -> tensor<?x16x8x7xf32>
  // CHECK: "mhlo.transpose"({{.*}}) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<?x16x8x7xf32>) -> tensor<?x8x7x16xf32>
  %36 = tensor.from_elements %16, %17, %34, %35 : tensor<4xi32>
  %37 = "mhlo.dynamic_conv"(%arg0, %arg1, %36) {batch_group_count = 1 : i64,
    dimension_numbers = {input_batch_dimension = 0 : i64,
    input_feature_dimension = 3 : i64,
    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
    kernel_input_feature_dimension = 2 : i64,
    kernel_output_feature_dimension = 3 : i64,
    kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
    output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64,
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
    feature_group_count = 2 : i64,
    rhs_dilation = dense<[2, 3]> : tensor<2xi64>,
    window_strides = dense<[4, 5]> : tensor<2xi64>}
    : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<?x8x7x16xf32>
  return %37 : tensor<?x8x7x16xf32>
}

// -----

// CHECK-LABEL: @conv
func @conv(%arg0: tensor<?x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<?x8x7x16xf32> {
  // CHECK: "mhlo.transpose"(%arg0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<?x32x32x6xf32>) -> tensor<?x6x32x32xf32>
  // CHECK: "mhlo.transpose"(%arg1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x3x16xf32>) -> tensor<16x3x3x3xf32>
  // CHECK: "mhlo.dynamic_conv"({{.*}})
  // CHECK: "mhlo.transpose"({{.*}}) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<?x16x8x7xf32>) -> tensor<?x8x7x16xf32>
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64,
    dimension_numbers = {input_batch_dimension = 0 : i64,
    input_feature_dimension = 3 : i64,
    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
    kernel_input_feature_dimension = 2 : i64,
    kernel_output_feature_dimension = 3 : i64,
    kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
    output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64,
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
    feature_group_count = 2 : i64,
    padding = dense<[[1, 2], [3, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 3]> : tensor<2xi64>,
    window_strides = dense<[4, 5]> : tensor<2xi64>}
    : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<?x8x7x16xf32>
  return %0 : tensor<?x8x7x16xf32>
}
