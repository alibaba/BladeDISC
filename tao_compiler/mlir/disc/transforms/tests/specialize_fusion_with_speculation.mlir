// RUN: DISC_ENABLE_SHAPE_CONSTRAINT_IR=0 DISC_ENABLE_HORIZONTAL_FUSION=0 disc-opt \
// RUN:   -pass-pipeline='builtin.module(func.func(disc-specialize-fusion-with-speculation{core-count=72 max-threads-per-core=1536}))' \
// RUN:   %s --split-input-file | FileCheck %s

// CHECK-LABEL: simple_broadcast_specialization
func.func @simple_broadcast_specialization(%arg0: !disc_ral.context) {
  %c0 = arith.constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32>
  %c0_0 = arith.constant 0 : index
  %1 = memref.dim %0, %c0_0 : memref<?x?xf32>
  %c1 = arith.constant 1 : index
  %2 = memref.dim %0, %c1 : memref<?x?xf32>
  %3 = memref.alloc() : memref<f32>
  %4 = tensor.from_elements %1, %2 : tensor<2xindex>
  %5 = bufferization.to_memref %4 : memref<2xindex>
  %6 = memref.alloc(%1, %2) : memref<?x?xf32>
  %7 = memref.alloc(%1, %2) : memref<?x?xf32>
  %8 = memref.alloc(%1, %2) : memref<?x?xf32>
  // CHECK: %[[T0:.*]] = "disc_ral.recv_input"
  // CHECK: %[[T6:.*]] = memref.alloc{{.*}} : memref<?x?xf32>
  // CHECK: %[[T7:.*]] = memref.alloc{{.*}} : memref<?x?xf32>
  // CHECK: %[[T8:.*]] = memref.alloc{{.*}} : memref<?x?xf32>
  // CHECK: %[[c0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[T9:.*]] = memref.dim %[[T0]], %[[c0_1]] : memref<?x?xf32>
  // CHECK: %[[c0_2:.*]] = arith.constant 0 : index
  // CHECK: %[[T10:.*]] = memref.dim %[[T7]], %[[c0_2]] : memref<?x?xf32>
  // CHECK: %[[T11:.*]] = arith.cmpi eq, %[[T9]], %[[T10]] : index
  // CHECK: %[[c1_3:.*]] = arith.constant 1 : index
  // CHECK: %[[T12:.*]] = memref.dim %[[T0]], %[[c1_3]] : memref<?x?xf32>
  // CHECK: %[[c1_4:.*]] = arith.constant 1 : index
  // CHECK: %[[T13:.*]] = memref.dim %[[T7]], %[[c1_4]] : memref<?x?xf32>
  // CHECK: %[[T14:.*]] = arith.cmpi eq, %[[T12]], %[[T13]] : index
  // CHECK: %[[T15:.*]] = arith.andi %[[T14]], %[[T11]] : i1
  // CHECK: scf.if %[[T15]] {
  // CHECK:   %[[CastedT0:.*]] = memref.reinterpret_cast %[[T0]]
  // CHECK:   %[[CastedT8:.*]] = memref.reinterpret_cast %[[T8]]
  // CHECK:   "lmhlo.fusion"() ({
  // CHECK:     "lmhlo.constant"
  // CHECK:     "lmhlo.dynamic_broadcast_in_dim"(%[[T3:.*]], %[[T5:.*]], %[[T6]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (memref<f32>, memref<2xindex>, memref<?x?xf32>) -> ()
  // CHECK:     "lmhlo.add"(%[[T6]], %[[CastedT0]], %[[CastedT8]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK:     "lmhlo.terminator"() : () -> ()
  // CHECK:   })
  // CHECK: } else {
  // CHECK:   "lmhlo.fusion"() ({
  // CHECK:     "lmhlo.constant"
  // CHECK:     "lmhlo.dynamic_broadcast_in_dim"(%[[T3]], %[[T5]], %[[T6]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (memref<f32>, memref<2xindex>, memref<?x?xf32>) -> ()
  // CHECK:     "lmhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[T5]], %[[T7]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<?x?xf32>, memref<2xindex>, memref<?x?xf32>) -> ()
  // CHECK:     "lmhlo.add"(%[[T6]], %[[T7]], %[[T8]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // CHECK:     "lmhlo.terminator"() : () -> ()
  // CHECK:   })
  // CHECK: }

  "lmhlo.fusion"() ({
    "lmhlo.constant"(%3) {value = dense<1.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%3, %5, %6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (memref<f32>, memref<2xindex>, memref<?x?xf32>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%0, %5, %7) {broadcast_dimensions = dense<[0,1]> : tensor<2xi64>} : (memref<?x?xf32>, memref<2xindex>, memref<?x?xf32>) -> ()
    "lmhlo.add"(%6, %7, %8) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test1", disc_vectorize_or_tile_hint = 2, disc.fusion_type = "kLoop", disc.device = "gpu"} : () -> ()
  %c0_1 = arith.constant 0 : index
  "disc_ral.send_output"(%arg0, %c0_1, %8) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: simple_row_reduction_vectorization_specialization
func.func @simple_row_reduction_vectorization_specialization(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: memref<f32>) -> (memref<?x?xf32>, memref<?xf32>) {
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: %[[T0:.*]] = memref.dim %arg1, %[[c1]] : memref<?x?xf32>
  // CHECK: %[[c512:.*]] = arith.constant 512 : index
  // CHECK: %[[T4:.*]] = arith.cmpi sgt, %[[T0]], %[[c512]] : index
  // Schedule 1
  // CHECK: scf.if %[[T4]] {
  // Vectorize with width 2.
  // CHECK:   scf.if %[[V2_1:.*]] {
  // CHECK:     "lmhlo.fusion"() ({
  // CHECK:       "lmhlo.abs"
  // CHECK:       "lmhlo.reduce"
  // CHECK:       "lmhlo.terminator"
  // CHECK:     }) {disc.device = "gpu", disc.fusion.name = "test2"
  // CHECK-SAME: disc_row_reduction_schedule_hint = 1 : i32
  // CHECK-SMAE: disc_thread_per_block_hint = 256 : i32
  // CHECK-SMAE: disc_vectorize_or_tile_hint = 2 : i32
  // CHECK:   } else {
  // No vectorization.
  // CHECK:     "lmhlo.fusion"() ({
  // CHECK:       "lmhlo.abs"
  // CHECK:       "lmhlo.reduce"
  // CHECK:       "lmhlo.terminator"
  // CHECK:     }) {disc.device = "gpu", disc.fusion.name = "test2"
  // CHECK-SAME: disc_row_reduction_schedule_hint = 1 : i32
  // CHECK-SMAE: disc_thread_per_block_hint = 256 : i32
  // CHECK-SMAE: disc_vectorize_or_tile_hint = 1 : i32
  // CHECK:   }
  // Schedule 2
  // CHECK: } else {
  // Vectorize with width 2.
  // CHECK:   scf.if %[[V2_2:.*]] {
  // CHECK:     "lmhlo.fusion"() ({
  // CHECK:       "lmhlo.abs"
  // CHECK:       "lmhlo.reduce"
  // CHECK:       "lmhlo.terminator"
  // CHECK:     }) {disc.device = "gpu", disc.fusion.name = "test2"
  // CHECK-SAME: disc_row_reduction_schedule_hint = 2 : i32
  // CHECK-SMAE: disc_thread_per_block_hint = 256 : i32
  // CHECK-SMAE: disc_vectorize_or_tile_hint = 2 : i32
  // CHECK:   } else {
  // No vectorization.
  // CHECK:     "lmhlo.fusion"() ({
  // CHECK:       "lmhlo.abs"
  // CHECK:       "lmhlo.reduce"
  // CHECK:       "lmhlo.terminator"
  // CHECK:     }) {disc.device = "gpu", disc.fusion.name = "test2"
  // CHECK-SAME: disc_row_reduction_schedule_hint = 2 : i32
  // CHECK-SMAE: disc_thread_per_block_hint = 256 : i32
  // CHECK-SMAE: disc_vectorize_or_tile_hint = 1 : i32
  // CHECK:   }
  // CHECK: }

  "lmhlo.fusion"() ({
    "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    "lmhlo.reduce"(%arg1, %arg3, %arg2) ( {
    ^bb0(%arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg4, %arg5, %arg6) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<?x?xf32>, memref<f32>, memref<?xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "test2", disc.device = "gpu", disc.fusion_type = "kRowReduction"} : () -> ()
  return %arg1, %arg2 : memref<?x?xf32>, memref<?xf32>
}
