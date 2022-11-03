// RUN: disc-opt -disc-element-type-converter=enable-fp16-gemm=false %s | FileCheck %s --check-prefix=BASIC
// RUN: disc-opt -disc-element-type-converter=enable-fp16-gemm=true %s | FileCheck %s --check-prefix=FP16

// CHECK-LABEL: @dot_fp32

// Test with `enable_fp16_gemm=false`
// BASIC: mhlo.dot_general
// BASIC-NOT: f16

// Test with `enable_fp16_gemm=true`
// FP16: mhlo.dot_general
// FP16-SAME: f16
func.func @dot_fp32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_batching_dimensions = [], rhs_contracting_dimensions = [1]>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}


