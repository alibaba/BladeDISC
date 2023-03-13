// RUN: disc-opt -pass-pipeline='builtin.module(func.func(disc-conv-rewriter{gpu-sm-cc-major=8}))' -split-input-file %s | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<?x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<?x8x7x16xf32> {
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c1 : tensor<?x32x32x6xf32>
  %1 = arith.index_cast %0 : index to i32
  %c0 = arith.constant 0 : index
  %2 = tensor.dim %arg1, %c0 : tensor<3x3x3x16xf32>
  %3 = arith.index_cast %2 : index to i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c2_i32_0 = arith.constant 2 : i32
  %4 = arith.subi %3, %c1_i32 : i32
  %5 = arith.muli %c2_i32_0, %4 : i32
  %6 = arith.addi %c1_i32, %5 : i32
  %7 = arith.subi %c4_i32, %c1_i32 : i32
  %8 = arith.addi %1, %7 : i32
  %9 = arith.divui %8, %c4_i32 : i32
  %10 = arith.subi %9, %c1_i32 : i32
  %11 = arith.muli %c4_i32, %10 : i32
  %12 = arith.addi %6, %11 : i32
  %13 = arith.subi %12, %1 : i32
  %14 = arith.cmpi sge, %13, %c0_i32 : i32
  %15 = arith.select %14, %13, %c0_i32 : i32
  %16 = arith.divui %15, %c2_i32 : i32
  %17 = arith.subi %15, %16 : i32
  %c2 = arith.constant 2 : index
  %18 = tensor.dim %arg0, %c2 : tensor<?x32x32x6xf32>
  %19 = arith.index_cast %18 : index to i32
  %c1_1 = arith.constant 1 : index
  %20 = tensor.dim %arg1, %c1_1 : tensor<3x3x3x16xf32>
  %21 = arith.index_cast %20 : index to i32
  %c0_i32_2 = arith.constant 0 : i32
  %c1_i32_3 = arith.constant 1 : i32
  %c2_i32_4 = arith.constant 2 : i32
  %c5_i32 = arith.constant 5 : i32
  %c3_i32 = arith.constant 3 : i32
  %22 = arith.subi %21, %c1_i32_3 : i32
  %23 = arith.muli %c3_i32, %22 : i32
  %24 = arith.addi %c1_i32_3, %23 : i32
  %25 = arith.subi %c5_i32, %c1_i32_3 : i32
  %26 = arith.addi %19, %25 : i32
  %27 = arith.divui %26, %c5_i32 : i32
  %28 = arith.subi %27, %c1_i32_3 : i32
  %29 = arith.muli %c5_i32, %28 : i32
  %30 = arith.addi %24, %29 : i32
  %31 = arith.subi %30, %19 : i32
  %32 = arith.cmpi sge, %31, %c0_i32_2 : i32
  %33 = arith.select %32, %31, %c0_i32_2 : i32
  %34 = arith.divui %33, %c2_i32_4 : i32
  %35 = arith.subi %33, %34 : i32
  // CHECK: mhlo.transpose
  // CHECK-SAME: permutation = dense<[3, 0, 1, 2]>
  // CHECK: mhlo.dynamic_conv
  // CHECK-SAME: batch_group_count = 1
  // CHECK-SAME: dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]>
  // CHECK-SAME: feature_group_count = 2
  // CHECK-SAME: rhs_dilation = dense<[2, 3]>
  // CHECK-SAME: window_strides = dense<[4, 5]>
  %36 = tensor.from_elements %16, %17, %34, %35 : tensor<4xi32>
  %37 = "mhlo.dynamic_conv"(%arg0, %arg1, %36) {batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 3,
    input_spatial_dimensions = [1, 2],
    kernel_input_feature_dimension = 2,
    kernel_output_feature_dimension = 3,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 0, output_feature_dimension = 3,
    output_spatial_dimensions = [1, 2]>, feature_group_count = 2 : i64, rhs_dilation = dense<[2, 3]> : tensor<2xi64>, window_strides = dense<[4, 5]> : tensor<2xi64>}
    : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<?x8x7x16xf32>
  return %37 : tensor<?x8x7x16xf32>
}

// -----

// CHECK-LABEL: @conv
func.func @conv(%arg0: tensor<?x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<?x8x6x16xf32> {
  // CHECK: mhlo.transpose
  // CHECK-SAME: ermutation = dense<[3, 0, 1, 2]>
  // CHECK: mhlo.dynamic_conv
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 3,
    input_spatial_dimensions = [1, 2],
    kernel_input_feature_dimension = 2,
    kernel_output_feature_dimension = 3,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 0, output_feature_dimension = 3,
    output_spatial_dimensions = [1, 2]>,
    feature_group_count = 2,
    padding = dense<[[1, 2], [3, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 3]> : tensor<2xi64>,
    window_strides = dense<[4, 5]> : tensor<2xi64>}
    : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<?x8x6x16xf32>
  return %0 : tensor<?x8x6x16xf32>
}
