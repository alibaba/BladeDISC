// RUN: disc-opt -disc-reduction-rewriter -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @two_side_reduction
func.func @two_side_reduction(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  // CHECK: mhlo.reduce
  // CHECK: mhlo.reduce
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  return %1 : tensor<?xf16>
}

// CHECK-LABEL: func.func @two_side_reduction_check_dims
func.func @two_side_reduction_check_dims(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  // CHECK: dimensions = [2, 3]
  // CHECK: dimensions = [0]
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  return %1 : tensor<?xf16>
}
