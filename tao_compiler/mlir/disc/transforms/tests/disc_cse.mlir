// RUN: disc-opt -cse -split-input-file %s -o - | FileCheck %s

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

// CHECK-LABEL: func.func @multiple_reduce_with_cst0
func.func @multiple_reduce_with_cst0(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  // CHECK: mhlo.reduce
  %11 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %2 = mhlo.multiply %arg1, %1 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %12 = mhlo.add %11, %11 {disc.device = "gpu"} : tensor<?xf16>
  // CHECK-NOT: mhlo.reduce
  %13 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %2 = mhlo.multiply %arg1, %1 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %14 = mhlo.multiply %12, %13 {disc.device = "gpu"} : tensor<?xf16>
  return %14 : tensor<?xf16>
}

// CHECK-LABEL: func.func @multiple_reduce_with_cst1
func.func @multiple_reduce_with_cst1(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  // CHECK: mhlo.reduce
  %11 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %2 = mhlo.multiply %arg1, %1 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %12 = mhlo.add %11, %11 {disc.device = "gpu"} : tensor<?xf16>
  // CHECK: mhlo.reduce
  %13 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %1 = mhlo.constant dense<3.000000e+00> : tensor<f16>
    %2 = mhlo.multiply %arg1, %1 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %14 = mhlo.multiply %12, %13 {disc.device = "gpu"} : tensor<?xf16>
  return %14 : tensor<?xf16>
}


// CHECK-LABEL: func.func @multiple_reduce_with_cst2
func.func @multiple_reduce_with_cst2(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  %cst1 = mhlo.constant dense<1.000000e+00> : tensor<f16>
  // CHECK: mhlo.reduce
  %11 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %2 = mhlo.multiply %arg1, %cst1 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %12 = mhlo.add %11, %11 {disc.device = "gpu"} : tensor<?xf16>
  // CHECK: mhlo.reduce
  %cst3 = mhlo.constant dense<3.000000e+00> : tensor<f16>
  %13 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %2 = mhlo.multiply %arg1, %cst3 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %14 = mhlo.multiply %12, %13 {disc.device = "gpu"} : tensor<?xf16>
  return %14 : tensor<?xf16>
}

// CHECK-LABEL: func.func @multiple_different_reduce
func.func @multiple_different_reduce(%arg0: tensor<?x?x?x?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant {disc.device = "gpu"} dense<0.000000e+00> : tensor<f16>
  // CHECK: mhlo.reduce
  %11 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %12 = mhlo.add %11, %11 {disc.device = "gpu"} : tensor<?xf16>
  // CHECK: mhlo.reduce
  %13 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
    %1 = mhlo.constant dense<3.000000e+00> : tensor<f16>
    %2 = mhlo.multiply %arg1, %1 : tensor<f16>
    %3 = mhlo.add %2, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<?x?x?x?xf16>, tensor<f16>) -> tensor<?xf16>
  %14 = mhlo.multiply %12, %13 {disc.device = "gpu"} : tensor<?xf16>
  return %14 : tensor<?xf16>
}
