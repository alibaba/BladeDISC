// RUN: disc-opt %s -disc-for-loop-unroll-interleave -split-input-file | FileCheck %s

// CHECK-LABEL: @mean_stitch_fusion
func.func @mean_stitch_fusion(%arg0: !disc_ral.context) attributes {tf.entry_function = {input_placements = "gpu", inputs = "input0", output_placements = "gpu", outputs = "output0"}} {
  %cst = arith.constant -0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c4_i32 = arith.constant 4 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %1 = "disc_ral.dispatch"(%arg0, %c0) {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} : (!disc_ral.context, index) -> memref<?x?x?xf32, "gpu">
  %2 = memref.dim %1, %c2 : memref<?x?x?xf32, "gpu">
  %3 = memref.dim %1, %c0 : memref<?x?x?xf32, "gpu">
  %4 = memref.dim %1, %c1 : memref<?x?x?xf32, "gpu">
  %6 = arith.muli %2, %4 : index
  %7 = memref.reinterpret_cast %1 to offset: [0], sizes: [%3, %4, %2], strides: [%6, %2, 1] {kDiscSymbolicDimAttr = [@S0, @S1, @S2]} : memref<?x?x?xf32, "gpu"> to memref<?x?x?xf32, "gpu">
  %9 = arith.index_cast %3 : index to i32
  %10 = arith.index_cast %4 : index to i32
  %11 = arith.muli %9, %10 : i32
  %13 = arith.index_cast %11 : i32 to index
  %18 = arith.index_cast %2 : index to i64
  %19 = memref.alloca() : memref<i64, "cpu">
  memref.store %18, %19[] : memref<i64, "cpu">
  %20 = memref.alloc() : memref<i64, "gpu">
  %21 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
  "disc_ral.dispatch"(%arg0, %21, %19, %20) {backend_config = "gpu", call_target_name = "h2d", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<i64, "cpu">, memref<i64, "gpu">) -> ()
  "disc_ral.dispatch"(%arg0, %21) {backend_config = "gpu", call_target_name = "sync_on_stream", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>) -> ()
  %25 = memref.alloc(%3, %4) {kDiscSymbolicDimAttr = [@S0, @S1]} : memref<?x?xf32, "gpu">
  "lmhlo.fusion"() ({
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%13, %c256) step (%c1, %c1) {
      %31 = memref.alloc() : memref<1xf32, 3>
      %32 = arith.divui %arg2, %c32 : index
      %33 = arith.remui %arg2, %c32 : index
      %34 = arith.cmpi eq, %33, %c0 : index
// CHECK-COUNT-4: disc_shape.linearize
// CHECK-COUNT-4: disc_shape.delinearize
// CHECK-COUNT-4: memref.load
// CHECK-COUNT-4: arith.addf
      %35 = scf.for %arg3 = %arg2 to %2 step %c256 iter_args(%arg4 = %cst) -> (f32) {
        %44 = "disc_shape.linearize"(%arg1, %arg3, %13, %2) {operand_segment_sizes = array<i32: 2, 2>} : (index, index, index, index) -> index
        %45:3 = "disc_shape.delinearize"(%44, %3, %4, %2) : (index, index, index, index) -> (index, index, index)
        %46 = memref.load %7[%45#0, %45#1, %45#2] : memref<?x?x?xf32, "gpu">
        %47 = arith.addf %arg4, %46 : f32
        scf.yield %47 : f32
      }
      %36 = memref.alloc() : memref<32xf32, 3>
      %result, %valid = gpu.shuffle  xor %35, %c1_i32, %c32_i32 : f32
      %37 = arith.addf %35, %result : f32
      %result_0, %valid_1 = gpu.shuffle  xor %37, %c2_i32, %c32_i32 : f32
      %38 = arith.addf %37, %result_0 : f32
      %result_2, %valid_3 = gpu.shuffle  xor %38, %c4_i32, %c32_i32 : f32
      %39 = arith.addf %38, %result_2 : f32
      %result_4, %valid_5 = gpu.shuffle  xor %39, %c8_i32, %c32_i32 : f32
      %40 = arith.addf %39, %result_4 : f32
      %result_6, %valid_7 = gpu.shuffle  xor %40, %c16_i32, %c32_i32 : f32
      %41 = arith.addf %40, %result_6 : f32
      scf.if %34 {
        memref.store %41, %36[%32] : memref<32xf32, 3>
      }
      gpu.barrier
      %42 = arith.cmpi eq, %arg2, %c0 : index
      %43 = arith.cmpi eq, %32, %c0 : index
      scf.if %43 {
        %44 = arith.cmpi slt, %33, %c8 : index
        %45 = scf.if %44 -> (f32) {
          %49 = memref.load %36[%33] : memref<32xf32, 3>
          scf.yield %49 : f32
        } else {
          scf.yield %cst : f32
        }
        %result_8, %valid_9 = gpu.shuffle  xor %45, %c1_i32, %c8_i32 : f32
        %46 = arith.addf %45, %result_8 : f32
        %result_10, %valid_11 = gpu.shuffle  xor %46, %c2_i32, %c8_i32 : f32
        %47 = arith.addf %46, %result_10 : f32
        %result_12, %valid_13 = gpu.shuffle  xor %47, %c4_i32, %c8_i32 : f32
        %48 = arith.addf %47, %result_12 : f32
        scf.if %42 {
          memref.assume_alignment %31, 4 : memref<1xf32, 3>
          memref.store %48, %31[%c0] : memref<1xf32, 3>
        }
      }
      gpu.barrier
// CHECK-COUNT-4: arith.addi
// CHECK-COUNT-4: arith.muli
// CHECK-COUNT-4: memref.reinterpret_cast
// CHECK-COUNT-4: memref.assume_alignment
// CHECK-COUNT-4: memref.load
// CHECK-COUNT-4: memref.load
// CHECK-COUNT-4: arith.divf
// CHECK-COUNT-4: memref.store
      scf.for %arg3 = %arg2 to %c1 step %c256 {
        %44 = arith.addi %arg1, %arg3 : index
        %45 = arith.muli %3, %4 : index
        %46 = memref.reinterpret_cast %25 to offset: [%c0], sizes: [%45], strides: [%c1] : memref<?x?xf32, "gpu"> to memref<?xf32, "gpu">
        memref.assume_alignment %46, 4 : memref<?xf32, "gpu">
        %47 = memref.load %31[%c0] : memref<1xf32, 3>
        %48 = memref.load %20[] : memref<i64, "gpu">
        %49 = arith.sitofp %48 : i64 to f32
        %50 = arith.divf %47, %49 : f32
        memref.store %50, %46[%44] : memref<?xf32, "gpu">
      }
      scf.yield
    }
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "gpu", disc.fusion.name = "main_kStitch_divide__7_1_0", disc.fusion.tag = "1b1rX_vectile2X_no_vectile", disc.fusion_type = "kStitch", disc_row_reduction_schedule_hint = 1 : i32, disc_thread_per_block_hint = 256 : i32, disc_vectorize_or_tile_hint = 1 : i32} : () -> ()
  memref.dealloc %20 : memref<i64, "gpu">
  "disc_ral.dispatch"(%arg0, %c0, %25) {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} : (!disc_ral.context, index, memref<?x?xf32, "gpu">) -> ()
  return
}

// CHECK-LABEL: @sigmoid_grad
func.func @sigmoid_grad(%arg0: !disc_ral.context) attributes {tf.entry_function = {input_placements = "gpu,gpu", inputs = "input0,input1", output_placements = "gpu", outputs = "output0"}} {
  %cst = arith.constant 1.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = "disc_ral.dispatch"(%arg0, %c0) {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} : (!disc_ral.context, index) -> memref<?x?xf32, "gpu">
  %1 = "disc_ral.dispatch"(%arg0, %c1) {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} : (!disc_ral.context, index) -> memref<?x?xf32, "gpu">
  %2 = memref.dim %1, %c0 : memref<?x?xf32, "gpu">
  %3 = memref.dim %0, %c0 : memref<?x?xf32, "gpu">
  %4 = memref.dim %1, %c1 : memref<?x?xf32, "gpu">
  %5 = memref.dim %0, %c1 : memref<?x?xf32, "gpu">
  %6 = arith.cmpi eq, %3, %c1 : index
  %7 = arith.select %6, %2, %3 : index
  %8 = arith.cmpi eq, %5, %c1 : index
  %9 = arith.select %8, %4, %5 : index
  %10 = arith.select %6, %7, %3 : index
  %11 = arith.select %8, %9, %5 : index
  %12 = memref.alloc(%10, %11) : memref<?x?xf32, "gpu">
  %13 = memref.reinterpret_cast %1 to offset: [0], sizes: [%3, %5], strides: [%5, 1] : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  %14 = memref.reinterpret_cast %0 to offset: [0], sizes: [%3, %5], strides: [%5, 1] : memref<?x?xf32, "gpu"> to memref<?x?xf32, "gpu">
  // CHECK: %[[T13:.*]] = memref.reinterpret_cast
  // CHECK: %[[T14:.*]] = memref.reinterpret_cast
  "lmhlo.fusion"() ({
    %15 = arith.muli %3, %5 : index
    %16 = arith.divui %15, %c4 : index
    scf.parallel (%arg1) = (%c0) to (%16) step (%c1) {
      scf.for %arg2 = %c0 to %c4 step %c1 {
        %17 = arith.muli %arg1, %c4 : index
        %18 = arith.addi %17, %arg2 : index
        %19 = arith.muli %3, %5 : index
        %20 = memref.reinterpret_cast %12 to offset: [%c0], sizes: [%19], strides: [%c1] : memref<?x?xf32, "gpu"> to memref<?xf32, "gpu">
        %21:2 = "disc_shape.delinearize"(%18, %3, %5) : (index, index, index) -> (index, index)
        %22 = memref.load %13[%21#0, %21#1] : memref<?x?xf32, "gpu">
        %23 = memref.load %14[%21#0, %21#1] : memref<?x?xf32, "gpu">
        %24 = arith.mulf %22, %23 : f32
        %25 = memref.load %14[%21#0, %21#1] : memref<?x?xf32, "gpu">
        %26 = arith.subf %cst, %25 : f32
        %27 = arith.mulf %24, %26 : f32
        memref.store %27, %20[%18] : memref<?xf32, "gpu">
      }
      scf.yield
    }
    // CHECK-NOT: scf.for
    // CHECK-DAG: memref.assume_alignment %[[T13]], 16
    // CHECK-DAG: memref.assume_alignment %[[T14]], 16
    // CHECK-COUNT-4: memref.load %[[T13]]
    // CHECK-COUNT-4: memref.load %[[T14]]
    // CHECK-COUNT-4: arith.mulf
    // CHECK-COUNT-4: arith.subf
    // CHECK-COUNT-4: arith.mulf
    // CHECK-COUNT-4: memref.store
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "gpu", disc.fusion.name = "main_kLoop_multiply__9_1_0", disc.fusion.tag = "no_ibXVec4", disc.fusion_type = "kLoop", disc_vectorize_or_tile_hint = 4 : i32} : () -> ()
  "disc_ral.dispatch"(%arg0, %c0, %12) {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} : (!disc_ral.context, index, memref<?x?xf32, "gpu">) -> ()
  return
}
