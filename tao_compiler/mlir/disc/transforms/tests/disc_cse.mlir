// RUN: disc-opt -disc-mhlo-cse -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: func.func @multiple_reduce
func.func @multiple_reduce(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  // CHECK: mhlo.reduce
  // CHECK-NOT: mhlo.reduce
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %2 = mhlo.add %1, %1 {disc.device = "gpu"} : tensor<?xf16>
  %12 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %13 = mhlo.multiply %2, %12 {disc.device = "gpu"} : tensor<?xf16>
  return %13 : tensor<?xf16>
}
