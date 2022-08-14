// RUN: disc-opt %s -disc-shape-simplifier='insert-tie-shape=true' | FileCheck %s

// CHECK-LABEL: main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
func.func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1> {
  // CHECK: %[[T4:.*]] = "disc_shape.tie_shape"(%[[ARG0:.*]], %[[T0:.*]], %[[T1:.*]], %[[T2:.*]], %[[T3:.*]]) :
  // CHECK: %[[T5:.*]] = "disc_shape.tie_shape"(%[[ARG1:.*]], %[[T0]], %[[T1]], %[[T2]], %[[T3]])
  // CHECK: %[[RESULT:.*]] = mhlo.compare
  // CHECK: %[[T6:.*]] = "disc_shape.tie_shape"(%[[RESULT]], %[[T0]], %[[T1]], %[[T2]], %[[T3]])
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
  return %0 : tensor<?x?x?x?xi1>
}
