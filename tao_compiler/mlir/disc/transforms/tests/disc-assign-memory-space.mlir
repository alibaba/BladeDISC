// RUN: disc-opt -split-input-file --disc-assign-memory-space  %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32, "cpu">) -> memref<?xf32, "cpu">
func.func @main(%arg0 : memref<?xf32>) -> memref<?xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}}  {
  // CHECK: %[[T0:.*]] = memref.dim %[[ARG0]], %c0 : memref<?xf32, "cpu">
  // CHECK: %[[T1:.*]] = memref.alloc(%[[T0]]) : memref<?xf32, "cpu">
  // CHECK:  return %[[T1]] : memref<?xf32, "cpu">
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf32>
  %1 = memref.alloc(%0) : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32, "cpu">) -> memref<?xf32, "gpu">
func.func @main(%arg0 : memref<?xf32>) -> memref<?xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "gpu", outputs = "output0"}}  {
  // CHECK: %[[T0:.*]] = memref.dim %[[ARG0]], %c0 : memref<?xf32, "cpu">
  // CHECK: %[[T1:.*]] = memref.alloc(%[[T0]]) : memref<?xf32, "gpu">
  // CHECK:  return %[[T1]] : memref<?xf32, "gpu">
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf32>
  %1 = memref.alloc(%0) : memref<?xf32>
  return %1 : memref<?xf32>
}

// -----

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x8xf32, "cpu">) -> memref<?x24xf32, "cpu">
func.func @main(%arg0: memref<?x8xf32>) -> memref<?x24xf32> attributes {tf.entry_function = {input_placements = "cpu", inputs = "input0", output_placements = "cpu", outputs = "output0"}} {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index

  // CHECK: %[[T0:.*]] = memref.alloc() : memref<2xi32, "cpu">
  %0 = memref.alloc() : memref<2xi32>

  // CHECK: "lmhlo.constant"(%[[T0]]) {disc.device = "cpu", value = dense<[7, 3]> : tensor<2xi32>} : (memref<2xi32, "cpu">) -> ()
  "lmhlo.constant"(%0) {disc.device = "cpu", value = dense<[7, 3]> : tensor<2xi32>} : (memref<2xi32>) -> ()

  %1 = memref.dim %arg0, %c0 : memref<?x8xf32>
  %2 = memref.load %0[%c0] : memref<2xi32>
  %3 = arith.index_cast %2 : i32 to index
  %4 = memref.load %0[%c1] : memref<2xi32>
  %5 = arith.index_cast %4 : i32 to index

  // CHECK: %[[T6:.*]] = memref.alloc() : memref<4xindex, "cpu">
  %6 = memref.alloc() : memref<4xindex>

  memref.store %3, %6[%c0] : memref<4xindex>
  memref.store %1, %6[%c1] : memref<4xindex>
  memref.store %5, %6[%c2] : memref<4xindex>
  memref.store %c8, %6[%c3] : memref<4xindex>

  // CHECK: %[[T7:.*]] = memref.alloc({{.*}}) : memref<?x8xf32, "gpu">
  %7 = memref.alloc(%1) : memref<?x8xf32>

  // CHECK: "lmhlo_disc.h2d"(%[[ARG0]], %[[T7]]) : (memref<?x8xf32, "cpu">, memref<?x8xf32, "gpu">) -> ()
  "lmhlo_disc.h2d"(%arg0, %7) : (memref<?x8xf32>, memref<?x8xf32>) -> ()

  // CHECK: %[[T8:.*]] = memref.alloc({{.*}}) : memref<?x?x?x8xf32, "gpu">
  // CHECK: %[[T9:.*]] = memref.cast %[[T8]] : memref<?x?x?x8xf32, "gpu"> to memref<?x?x?x?xf32, "gpu">
  // CHECK: "lmhlo.dynamic_broadcast_in_dim"(%[[T7]], %[[T6]], %[[T9]])
  %8 = memref.alloc(%3, %1, %5) : memref<?x?x?x8xf32>
  %9 = memref.cast %8 : memref<?x?x?x8xf32> to memref<?x?x?x?xf32>
  "lmhlo.dynamic_broadcast_in_dim"(%7, %6, %9) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x8xf32>, memref<4xindex>, memref<?x?x?x?xf32>) -> ()

  %10 = arith.muli %3, %1 : index
  %11 = arith.muli %5, %c8 : index

  // CHECK: %[[T12:.*]] = memref.alloc() : memref<2xindex, "cpu">
  %12 = memref.alloc() : memref<2xindex>
  memref.store %10, %12[%c0] : memref<2xindex>
  memref.store %11, %12[%c1] : memref<2xindex>

  // CHECK: %[[T13:.*]] = memref.alloc({{.*}}) : memref<?x24xf32, "gpu">
  // CHECK: "lmhlo.dynamic_reshape"(%[[T9]], %[[T12]], %[[T13]])
  // CHECK: %[[T14:.*]] = memref.alloc({{.*}}) : memref<?x24xf32, "cpu">
  // CHECK: "lmhlo_disc.d2h"(%[[T13]], %[[T14]]) {disc.device = "cpu"} : (memref<?x24xf32, "gpu">, memref<?x24xf32, "cpu">) -> ()
  %13 = memref.alloc(%10) : memref<?x24xf32>
  "lmhlo.dynamic_reshape"(%9, %12, %13) {disc.device = "gpu"} : (memref<?x?x?x?xf32>, memref<2xindex>, memref<?x24xf32>) -> ()
  %14 = memref.alloc(%10) : memref<?x24xf32>
  "lmhlo_disc.d2h"(%13, %14) {disc.device = "cpu"} : (memref<?x24xf32>, memref<?x24xf32>) -> ()
  return %14 : memref<?x24xf32>
}

// -----

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x10xi32, "cpu">, %[[ARG1:.*]]: memref<10x10xi32, "gpu">) -> (memref<10xi32, "cpu">, memref<10xi32, "gpu">)
func.func @main(%arg0: memref<10x10xi32>, %arg1: memref<10x10xi32>) -> (memref<10xi32>, memref<10xi32>) attributes {tf.entry_function = {input_placements = "cpu,gpu", inputs = "input0,input1", output_placements = "cpu,gpu", outputs = "output0,output1"}} {
  // CHECK: %[[T0:.*]] = memref.alloc({{.*}}) : memref<i32, "cpu">
  %0 = memref.alloc() : memref<i32>
  // CHECK: "lmhlo.constant"(%[[T0]]) {value = dense<1> : tensor<i32>} : (memref<i32, "cpu">) -> ()
  "lmhlo.constant"(%0) {value = dense<1> : tensor<i32>} : (memref<i32>) -> ()

  // CHECK: %[[T1:.*]] = memref.alloc({{.*}}) : memref<10xi32, "cpu">
  // CHECK: "lmhlo.reduce"(%[[ARG0]], %[[T0]], %[[T1]])
  %1 = memref.alloc() : memref<10xi32>
  "lmhlo.reduce"(%arg0, %0, %1) ( {
  ^bb0(%arg43: memref<i32>, %arg44: memref<i32>, %arg45: memref<i32>):  // no predecessors
    "lmhlo.multiply"(%arg43, %arg44, %arg45) {disc.device = "cpu"} : (memref<i32>, memref<i32>, memref<i32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<0> : tensor<1xi64>, disc.device = "cpu"} : (memref<10x10xi32>, memref<i32>, memref<10xi32>) -> ()

  // CHECK: %[[T2:.*]] = memref.alloc({{.*}}) : memref<10xi32, "gpu">
  // CHECK: %[[T3:.*]] = memref.alloc({{.*}}) : memref<i32, "gpu">
  // CHECK: "lmhlo.constant"(%[[T3]]) {value = dense<1> : tensor<i32>} : (memref<i32, "gpu">) -> ()
  // CHECK: "lmhlo.reduce"(%[[ARG1]], %[[T3]], %[[T2]])
  %2 = memref.alloc() : memref<10xi32>
  "lmhlo.reduce"(%arg1, %0, %2) ( {
  ^bb0(%arg43: memref<i32>, %arg44: memref<i32>, %arg45: memref<i32>):  // no predecessors
    "lmhlo.multiply"(%arg43, %arg44, %arg45) {disc.device = "gpu"} : (memref<i32>, memref<i32>, memref<i32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {dimensions = dense<0> : tensor<1xi64>, disc.device = "gpu"} : (memref<10x10xi32>, memref<i32>, memref<10xi32>) -> ()
  return %1, %2 : memref<10xi32>, memref<10xi32>
}

// -----

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32, "gpu">)
// CHECK-SAME: memref<?xf32, "gpu">
func.func @main(%arg0: memref<?xf32>) -> (memref<?xf32>) attributes {tf.entry_function = {input_placements = "gpu", inputs = "input0", output_placements = "gpu", outputs = "output0"}} {
  // CHECK: lmhlo_disc.custom_call_v2
  // CHECK-SAME: %[[ARG0]]
  // CHECK-SAME: (memref<?xf32, "gpu">) -> (memref<?xf32, "gpu">, memref<?xf32, "cpu">, memref<?xf32, "gpu">, memref<?xi32, "cpu">)
  // CHECK-NEXT: %[[T0:.*]] = "lmhlo_disc.custom_call_v2"
  // CHECK-SAME: (memref<?xf32, "gpu">, memref<?xf32, "cpu">, memref<?xf32, "gpu">, memref<?xi32, "cpu">) -> memref<?xf32, "gpu">
  // CHECK: return %[[T0]]
  %0:4 = "lmhlo_disc.custom_call_v2"(%arg0) {call_target_name = "test1", custom_attrs = {}, device = "d", disc.device = "gpu", expected_input_layouts = "*", expected_output_layouts = "*,*,*,*", has_side_effect = false, input_layouts = "*", input_placements = "d", output_layouts = "*,*,*,*", output_placements = "d,h,x,h"} : (memref<?xf32>) -> (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xi32>)
  %1 = "lmhlo_disc.custom_call_v2"(%0#0, %0#1, %0#2, %0#3) {call_target_name = "test2", custom_attrs = {}, device = "d", disc.device = "gpu", expected_input_layouts = "*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*", input_placements = "x,h,d,s", output_layouts = "*", output_placements = "d"} : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xi32>) -> memref<?xf32>
  return %1 : memref<?xf32>
}
