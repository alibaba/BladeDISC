// RUN: disc-opt -disc-hlo-legalize-to-lhlo -hlo-legalize-to-lhlo \
// RUN:  -canonicalize -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: test_disc_mhlo_only_static_shape
// CHECK-SAME: (%[[ARG0:.*]]: memref<2x2xf32>, %[[ARG1:.*]]: memref<2x2xf32>)
func @test_disc_mhlo_only_static_shape(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
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
func @test_disc_mhlo_only_dynamic_shape(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK: %[[T0:.*]] = memref.tensor_load %[[ARG0]] : memref<?x?xf32>
  // CHECK: %[[T1:.*]] = shape.shape_of %[[T0]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[T2:.*]] = tensor.extract %[[T1]][%c0] : tensor<2xindex>
  // CHECK: %[[T3:.*]] = tensor.extract %[[T1]][%c1] : tensor<2xindex>
  // CHECK: %[[T4:.*]] = memref.alloc(%[[T2]], %[[T3]]) : memref<?x?xf32>
  // CHECK: "lmhlo_disc.h2d"(%[[ARG0]], %[[T4]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  %0 = "mhlo_disc.h2d"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: %[[T5:.*]] = memref.tensor_load %[[ARG1]] : memref<?x?xf32>
  // CHECK: %[[T6:.*]] = shape.shape_of %[[T5]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[T7:.*]] = tensor.extract %[[T6]][%c0] : tensor<2xindex>
  // CHECK: %[[T8:.*]] = tensor.extract %[[T6]][%c1] : tensor<2xindex>
  // CHECK: %[[T9:.*]] = memref.alloc(%[[T7]], %[[T8]]) : memref<?x?xf32>
  // CHECK: "lmhlo_disc.d2h"(%[[ARG1]], %[[T9]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  %1 = "mhlo_disc.d2h"(%arg1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: test_mixed_disc_mhlo_and_mhlo
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>)
func @test_mixed_disc_mhlo_and_mhlo(%arg0: tensor<?x?xf32>) -> (tensor<100x100xf32>, tensor<?x?xf32>) {
  // CHECK: %[[T0:.*]] = memref.alloc() : memref<100x100xf32>
  // CHECK: "lmhlo.constant"(%[[T0]]) {value = dense<0.000000e+00> : tensor<100x100xf32>} : (memref<100x100xf32>) -> ()
  // CHECK: %[[T1:.*]] = memref.alloc() : memref<100x100xf32>
  // CHECK: "lmhlo_disc.h2d"(%[[T0]], %[[T1]]) : (memref<100x100xf32>, memref<100x100xf32>) -> ()
  %0 = mhlo.constant dense<0.000000e+00> : tensor<100x100xf32>
  %1 = "mhlo_disc.h2d"(%0) : (tensor<100x100xf32>) -> tensor<100x100xf32>

  // CHECK: %[[T2:.*]] = memref.tensor_load %[[ARG0]] : memref<?x?xf32>
  // CHECK: %[[T3:.*]] = shape.shape_of %[[T2]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[T4:.*]] = tensor.extract %[[T3]][%c0] : tensor<2xindex>
  // CHECK: %[[T5:.*]] = tensor.extract %[[T3]][%c1] : tensor<2xindex>
  // CHECK: %[[T6:.*]] = memref.alloc(%[[T4]], %[[T5]]) : memref<?x?xf32>
  // CHECK: "lmhlo.abs"(%[[ARG0]], %[[T6]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK: %[[T7:.*]] = memref.tensor_load %[[T6]] : memref<?x?xf32>
  // CHECK: %[[T8:.*]] = shape.shape_of %[[T7]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[T9:.*]] = tensor.extract %[[T8]][%c0] : tensor<2xindex>
  // CHECK: %[[T10:.*]] = tensor.extract %[[T8]][%c1] : tensor<2xindex>
  // CHECK: %[[T11:.*]] = memref.alloc(%[[T9]], %[[T10]]) : memref<?x?xf32>
  // CHECK: "lmhlo_disc.d2h"(%[[T6]], %[[T11]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  %2 = "mhlo.abs"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "mhlo_disc.d2h"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0, %3 : tensor<100x100xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: test_topk_custom_call
func @test_topk_custom_call(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi64>, %arg2: tensor<index>) -> (tensor<?x?xf32>, tensor<?x?xi64>) {
  // CHECK: lmhlo_disc.custom_call
  %1, %2 = "mhlo_disc.custom_call"(%arg0, %arg1, %arg2) { backend_config = "{\"dimension\": 5}", call_target_name = "topk" } : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<index>) -> (tensor<?x?xf32>, tensor<?x?xi64>)
  return %1, %2 : tensor<?x?xf32>, tensor<?x?xi64>
}
