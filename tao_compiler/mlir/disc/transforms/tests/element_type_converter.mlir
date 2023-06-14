// RUN: disc-opt -split-input-file -disc-element-type-converter %s | FileCheck %s
// RUN: disc-opt -split-input-file -disc-element-type-converter=promote-fp16-sensitive-ops-to-f32=true %s | FileCheck %s --check-prefix=F16232_CHECK

// CHECK-LABEL: @rank2_colunm_reduction_i1
func.func @rank2_colunm_reduction_i1(%arg0: tensor<?x?xi1>) -> tensor<?xi1> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<i1>, %arg2: tensor<i1>):
    %2 = mhlo.add %arg1, %arg2 : tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?x?xi1>, tensor<i1>) -> tensor<?xi1>
  // CHECK: mhlo.convert {{.*}} : (tensor<?x?xi1>) -> tensor<?x?xi32>
  // CHECK-NEXT: mhlo.reduce
  // CHECK-NOT: i1
  // CHECK: mhlo.convert {{.*}} : (tensor<?xi32>) -> tensor<?xi1>
  return %1 : tensor<?xi1>
}

// -----

// F16232_CHECK-LABEL: @f16_tanh
// F16232_CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf16>)
func.func @f16_tanh(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
  // F16232_CHECK: %[[T0:.*]] = mhlo.convert %[[ARG0]] : (tensor<?x?xf16>) -> tensor<?x?xf32>
  // F16232_CHECK: %[[T1:.*]] = mhlo.tanh %[[T0]] : tensor<?x?xf32>
  // F16232_CHECK: %[[T2:.*]] = mhlo.convert %[[T1:.*]] : (tensor<?x?xf32>) -> tensor<?x?xf16>
  // F16232_CHECK: return %[[T2]]
  %0 = mhlo.tanh %arg0 : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// F16232_CHECK-LABEL: @f16_rsqrt
// F16232_CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf16>)
func.func @f16_rsqrt(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
  // F16232_CHECK: %[[T0:.*]] = mhlo.convert %[[ARG0]] : (tensor<?x?xf16>) -> tensor<?x?xf32>
  // F16232_CHECK: %[[T1:.*]] = mhlo.rsqrt %[[T0]] : tensor<?x?xf32>
  // F16232_CHECK: %[[T2:.*]] = mhlo.convert %[[T1:.*]] : (tensor<?x?xf32>) -> tensor<?x?xf16>
  // F16232_CHECK: return %[[T2]]
  %0 = mhlo.rsqrt %arg0 : tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// F16232_CHECK-LABEL: @f16_reduce
// F16232_CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf16>, %[[ARG1:.*]]: tensor<f16>)
func.func @f16_reduce(%arg0: tensor<?x?xf16>, %arg1: tensor<f16>) -> tensor<?xf16> {
  // F16232_CHECK: %[[T0:.*]] = mhlo.convert %[[ARG0]] : (tensor<?x?xf16>) -> tensor<?x?xf32>
  // F16232_CHECK: %[[T1:.*]] = mhlo.convert %[[ARG1]] : (tensor<f16>) -> tensor<f32>
  // F16232_CHECK: %[[T2:.*]] = mhlo.reduce
  // F16232_CHECK-SAME: %[[T0]]
  // F16232_CHECK-SAME: %[[T1]]
  // F16232_CHECK: %[[T3:.*]] = mhlo.convert %[[T2]] : (tensor<?xf32>) -> tensor<?xf16>
  // F16232_CHECK: return %[[T3]]
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f16>
    "mhlo.return"(%1) : (tensor<f16>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?x?xf16>, tensor<f16>) -> tensor<?xf16>
  return %0 : tensor<?xf16>
}

