// RUN: disc-opt -disc-lower-to-library-call %s -o - | FileCheck %s

// CHECK-LABEL: func @test_recv_input_and_send_output
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context) {
func @test_recv_input_and_send_output(%arg0: !disc_ral.context) {
  // CHECK: %[[T0:.*]] = "disc_ral.dispatch"(%[[CTX]], %c0)
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index) -> memref<?x?xf32>

  // CHECK: %[[T1:.*]] = "disc_ral.dispatch"(%[[CTX]], %c1)
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index) -> memref<?x?xf32>

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %c0, %[[T0]])
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index, memref<?x?xf32>) -> ()

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %c1, %[[T1]])
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index, memref<?x?xf32>) -> ()
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32>
  %1 = "disc_ral.recv_input"(%arg0, %c1) : (!disc_ral.context, index) -> memref<?x?xf32>
  "disc_ral.send_output"(%arg0, %c0, %0) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  "disc_ral.send_output"(%arg0, %c1, %1) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  return
}

// -----


// CHECK-LABEL: h2d_dynamic_shape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @h2d_dynamic_shape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32>
  %1 = memref.dim %0, %c0 : memref<?x?xf32>
  %2 = memref.dim %0, %c1 : memref<?x?xf32>
  %3 = memref.alloc(%1, %2) : memref<?x?xf32>

  // CHECK: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[STREAM:.*]] = llvm.inttoptr %[[T0:.*]] : i32 to !llvm.ptr<i8>
  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]]
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "h2d", has_side_effect = false}

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]])
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "sync_on_stream", has_side_effect = false}
  "lmhlo_disc.h2d"(%0, %3) : (memref<?x?xf32>, memref<?x?xf32>) -> ()

  "disc_ral.send_output"(%arg0, %c0, %3) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: d2h_dynamic_shape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @d2h_dynamic_shape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32>
  %1 = memref.dim %0, %c0 : memref<?x?xf32>
  %2 = memref.dim %0, %c1 : memref<?x?xf32>
  %3 = memref.alloc(%1, %2) : memref<?x?xf32>

  // CHECK: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[STREAM:.*]] = llvm.inttoptr %[[T0:.*]] : i32 to !llvm.ptr<i8>
  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]]
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "d2h", has_side_effect = false}

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]])
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "sync_on_stream", has_side_effect = false}
  "lmhlo_disc.d2h"(%0, %3) : (memref<?x?xf32>, memref<?x?xf32>) -> ()

  "disc_ral.send_output"(%arg0, %c0, %3) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: h2d_static_shape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @h2d_static_shape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<2x2xf32>
  %1 = memref.alloc() : memref<2x2xf32>

  // CHECK: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[STREAM:.*]] = llvm.inttoptr %[[T0:.*]] : i32 to !llvm.ptr<i8>
  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]]
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "h2d", has_side_effect = false}

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]])
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "sync_on_stream", has_side_effect = false}
  "lmhlo_disc.h2d"(%0, %1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()

  "disc_ral.send_output"(%arg0, %c0, %1) : (!disc_ral.context, index, memref<2x2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: d2h_static_shape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @d2h_static_shape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<2x2xf32>
  %1 = memref.alloc() : memref<2x2xf32>

  // CHECK: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[STREAM:.*]] = llvm.inttoptr %[[T0:.*]] : i32 to !llvm.ptr<i8>
  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]]
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "d2h", has_side_effect = false}

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]])
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "sync_on_stream", has_side_effect = false}
  "lmhlo_disc.d2h"(%0, %1) : (memref<2x2xf32>, memref<2x2xf32>) -> ()

  "disc_ral.send_output"(%arg0, %c0, %1) : (!disc_ral.context, index, memref<2x2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: removable_reshape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @removable_reshape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<2x2xf32, "gpu">
  %1 = memref.alloc() : memref<4x1xf32, "gpu">

  // CHECK: %[[R0:.*]] = memref.alloca() : memref<2xindex, "cpu">
  // CHECK: memref.store %c4, %[[R0]][%c0] : memref<2xindex, "cpu">
  // CHECK: memref.store %c1, %[[R0]][%c1] : memref<2xindex, "cpu">
  // CHECK: %[[R1:.*]] = "disc_ral.dispatch"
  // CHECK-SAME: %[[R0]]
  // CHECK-SAME: call_target_name = "inc_ref"
  // CHECK-NOT: lmhlo.reshape
  "lmhlo.reshape"(%0, %1) : (memref<2x2xf32, "gpu">, memref<4x1xf32, "gpu">) -> ()

  "disc_ral.send_output"(%arg0, %c0, %1) : (!disc_ral.context, index, memref<4x1xf32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: removable_copy
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @removable_copy(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<2x2xf32, "gpu">
  %1 = memref.alloc() : memref<2x2xf32, "gpu">

  // CHECK: %[[R0:.*]] = memref.alloca() : memref<2xindex, "cpu">
  // CHECK: memref.store %c2, %[[R0]][%c0] : memref<2xindex, "cpu">
  // CHECK: memref.store %c2, %[[R0]][%c1] : memref<2xindex, "cpu">
  // CHECK: %[[R1:.*]] = "disc_ral.dispatch"
  // CHECK-SAME: %[[R0]]
  // CHECK-SAME: call_target_name = "inc_ref"
  // CHECK-NOT: lmhlo.copy
  "lmhlo.copy"(%0, %1) : (memref<2x2xf32, "gpu">, memref<2x2xf32, "gpu">) -> ()

  "disc_ral.send_output"(%arg0, %c0, %1) : (!disc_ral.context, index, memref<2x2xf32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: removable_dynamic_reshape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @removable_dynamic_reshape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?xf32, "gpu">
  %d0 = memref.dim %0, %c0 : memref<?xf32, "gpu">
  %1 = memref.alloc(%d0) : memref<?x1xf32, "gpu">
  %2 = memref.alloc() : memref<2xindex, "cpu">
  memref.store %d0, %2[%c0] : memref<2xindex, "cpu">
  memref.store %c1, %2[%c1] : memref<2xindex, "cpu">

  // CHECK: %[[R0:.*]] = memref.alloca() : memref<2xindex, "cpu">
  // CHECK: memref.store %{{.*}}, %[[R0]][%c0] : memref<2xindex, "cpu">
  // CHECK: memref.store %c1, %[[R0]][%c1] : memref<2xindex, "cpu">
  // CHECK: %[[R1:.*]] = "disc_ral.dispatch"
  // CHECK-SAME: %[[R0]]
  // CHECK-SAME: call_target_name = "inc_ref"
  // CHECK-NOT: lmhlo.dynamic_reshape
  "lmhlo.dynamic_reshape"(%0, %2, %1) : (memref<?xf32, "gpu">, memref<2xindex, "cpu">, memref<?x1xf32, "gpu">) -> ()

  "disc_ral.send_output"(%arg0, %c0, %1) : (!disc_ral.context, index, memref<?x1xf32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: removable_copy_with_safe_memref_cast
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @removable_copy_with_safe_memref_cast(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32, "gpu">
  %1 = memref.alloc() : memref<2x2xf32, "gpu">
  %2 = memref.cast %1 : memref<2x2xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.alloc() : memref<2x2xf32, "gpu">
  %4 = memref.cast %3 : memref<2x2xf32, "gpu"> to memref<?x?xf32, "gpu">

  // CHECK: %[[R0:.*]] = memref.alloca() : memref<2xindex, "cpu">
  // CHECK: memref.store %[[D0:.*]], %[[R0]][%c0] : memref<2xindex, "cpu">
  // CHECK: memref.store %[[D0:.*]], %[[R0]][%c1] : memref<2xindex, "cpu">
  // CHECK: %[[R1:.*]] = "disc_ral.dispatch"
  // CHECK-SAME: %[[R0]]
  // CHECK-SAME: call_target_name = "inc_ref"
  // CHECK-NOT: lmhlo.copy
  "lmhlo.copy"(%0, %2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.copy"(%2, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  // CHECK: dealloc
  // CHECK: dealloc
  memref.dealloc %1 : memref<2x2xf32, "gpu">

  "disc_ral.send_output"(%arg0, %c0, %4) : (!disc_ral.context, index, memref<?x?xf32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: removable_copy_with_unsafe_memref_cast
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @removable_copy_with_unsafe_memref_cast(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32, "gpu">
  %1 = memref.alloc() : memref<2x2xf32, "gpu">
  %2 = memref.cast %1 : memref<2x2xf32, "gpu"> to memref<?x?xf32, "gpu">
  %3 = memref.alloc() : memref<2x2xf32, "gpu">
  %4 = memref.cast %3 : memref<2x2xf32, "gpu"> to memref<?x?xf32, "gpu">

  // CHECK: "lmhlo.copy"
  // CHECK: %[[R0:.*]] = memref.alloca() : memref<2xindex, "cpu">
  // CHECK: memref.store %[[D0:.*]], %[[R0]][%c0] : memref<2xindex, "cpu">
  // CHECK: memref.store %[[D0:.*]], %[[R0]][%c1] : memref<2xindex, "cpu">
  // CHECK: %[[R1:.*]] = "disc_ral.dispatch"
  // CHECK-SAME: %[[R0]]
  // CHECK-SAME: call_target_name = "inc_ref"
  // CHECK: "lmhlo.abs"

  "lmhlo.copy"(%0, %2) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.copy"(%2, %4) : (memref<?x?xf32, "gpu">, memref<?x?xf32, "gpu">) -> ()
  "lmhlo.abs"(%1, %1) : (memref<2x2xf32, "gpu">, memref<2x2xf32, "gpu">) -> ()
  // CHECK: dealloc
  // CHECK-NOT: dealloc
  memref.dealloc %1 : memref<2x2xf32, "gpu">

  "disc_ral.send_output"(%arg0, %c0, %4) : (!disc_ral.context, index, memref<?x?xf32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: topk_dynamic_shape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @topk_dynamic_shape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32, "gpu">
  %c1 = constant 1 : index
  %1 = "disc_ral.recv_input"(%arg0, %c1) : (!disc_ral.context, index) -> memref<?x?xi32, "gpu">
  %c2 = constant 2 : index
  %2 = "disc_ral.recv_input"(%arg0, %c2) : (!disc_ral.context, index) -> memref<i32, "cpu">
  %3 = memref.dim %0, %c0 : memref<?x?xf32, "gpu">
  %4 = memref.load %2[] : memref<i32, "cpu">
  %5 = index_cast %4 : i32 to index
  %6 = memref.alloc(%3, %5) : memref<?x?xf32, "gpu">
  %7 = memref.alloc(%3, %5) : memref<?x?xi32, "gpu">
  // CHECK: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[STREAM:.*]] = llvm.inttoptr %[[T0:.*]] : i32 to !llvm.ptr<i8>
  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]]
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "ral_dsort", has_side_effect = false}
  "lmhlo_disc.custom_call"(%0, %1, %2, %6, %7) {backend_config = "{\22dimension\22:1}", call_target_name = "topk", disc.device = "gpu", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<?x?xf32, "gpu">, memref<?x?xi32, "gpu">, memref<i32, "cpu">, memref<?x?xf32, "gpu">, memref<?x?xi32, "gpu">) -> ()
  %c0_0 = constant 0 : index
  "disc_ral.send_output"(%arg0, %c0_0, %6) : (!disc_ral.context, index, memref<?x?xf32, "gpu">) -> ()
  %c1_1 = constant 1 : index
  "disc_ral.send_output"(%arg0, %c1_1, %7) : (!disc_ral.context, index, memref<?x?xi32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: topk_static_shape
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context)
func @topk_static_shape(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<2x16xf32, "gpu">
  %c1 = constant 1 : index
  %1 = "disc_ral.recv_input"(%arg0, %c1) : (!disc_ral.context, index) -> memref<2x16xi32, "gpu">
  %c2 = constant 2 : index
  %2 = "disc_ral.recv_input"(%arg0, %c2) : (!disc_ral.context, index) -> memref<i32, "cpu">
  %3 = memref.load %2[] : memref<i32, "cpu">
  %4 = index_cast %3 : i32 to index
  %5 = memref.alloc(%4) : memref<2x?xf32, "gpu">
  %6 = memref.alloc(%4) : memref<2x?xi32, "gpu">
  // CHECK: %[[T0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[STREAM:.*]] = llvm.inttoptr %[[T0:.*]] : i32 to !llvm.ptr<i8>
  // CHECK: "disc_ral.dispatch"(%[[CTX]], %[[STREAM]]
  // CHECK-SAME: {backend_config = "gpu", call_target_name = "ral_dsort", has_side_effect = false}
  "lmhlo_disc.custom_call"(%0, %1, %2, %5, %6) {backend_config = "{\22dimension\22:1}", call_target_name = "topk", disc.device = "gpu", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x16xf32, "gpu">, memref<2x16xi32, "gpu">, memref<i32, "cpu">, memref<2x?xf32, "gpu">, memref<2x?xi32, "gpu">) -> ()
  %c0_0 = constant 0 : index
  "disc_ral.send_output"(%arg0, %c0_0, %5) : (!disc_ral.context, index, memref<2x?xf32, "gpu">) -> ()
  %c1_1 = constant 1 : index
  "disc_ral.send_output"(%arg0, %c1_1, %6) : (!disc_ral.context, index, memref<2x?xi32, "gpu">) -> ()
  return
}

// -----

// CHECK-LABEL: dynamic_conv
func @dynamic_conv(%arg0: !disc_ral.context) {
  %c0 = constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?x?x?xf32, "gpu">
  %c1 = constant 1 : index
  %1 = "disc_ral.recv_input"(%arg0, %c1) : (!disc_ral.context, index) -> memref<?x?x?x?xf32, "gpu">
  %c1_i32 = constant 1 : i32
  %c3_i32 = constant 3 : i32
  %c2_i32 = constant 2 : i32
  %c3 = constant 3 : index
  %2 = memref.dim %1, %c3 : memref<?x?x?x?xf32, "gpu">
  %3 = memref.dim %0, %c3 : memref<?x?x?x?xf32, "gpu">
  %c2 = constant 2 : index
  %4 = memref.dim %1, %c2 : memref<?x?x?x?xf32, "gpu">
  %5 = memref.dim %0, %c2 : memref<?x?x?x?xf32, "gpu">
  %c1_0 = constant 1 : index
  %6 = memref.dim %1, %c1_0 : memref<?x?x?x?xf32, "gpu">
  %7 = memref.dim %0, %c1_0 : memref<?x?x?x?xf32, "gpu">
  %c0_i32 = constant 0 : i32
  %c0_1 = constant 0 : index
  %8 = memref.dim %0, %c0_1 : memref<?x?x?x?xf32, "gpu">
  %9 = memref.dim %1, %c0_1 : memref<?x?x?x?xf32, "gpu">
  %10 = memref.alloca() : memref<4xi32, "cpu">
  memref.store %c0_i32, %10[%c0_1] : memref<4xi32, "cpu">
  memref.store %c0_i32, %10[%c1_0] : memref<4xi32, "cpu">
  memref.store %c0_i32, %10[%c2] : memref<4xi32, "cpu">
  memref.store %c0_i32, %10[%c3] : memref<4xi32, "cpu">
  %11 = memref.alloc() : memref<f32, "gpu">
  %12 = index_cast %7 : index to i32
  %13 = index_cast %9 : index to i32
  %14 = addi %12, %c2_i32 : i32
  %15 = divi_unsigned %14, %c3_i32 : i32
  %16 = subi %15, %c1_i32 : i32
  %17 = muli %16, %c3_i32 : i32
  %18 = addi %13, %17 : i32
  %19 = subi %18, %12 : i32
  %20 = cmpi sge, %19, %c0_i32 : i32
  %21 = select %20, %19, %c0_i32 : i32
  %22 = divi_unsigned %21, %c2_i32 : i32
  %23 = subi %21, %22 : i32
  %24 = index_cast %5 : index to i32
  %25 = index_cast %6 : index to i32
  %26 = addi %24, %c2_i32 : i32
  %27 = divi_unsigned %26, %c3_i32 : i32
  %28 = subi %27, %c1_i32 : i32
  %29 = muli %28, %c3_i32 : i32
  %30 = addi %25, %29 : i32
  %31 = subi %30, %24 : i32
  %32 = cmpi sge, %31, %c0_i32 : i32
  %33 = select %32, %31, %c0_i32 : i32
  %34 = divi_unsigned %33, %c2_i32 : i32
  %35 = subi %33, %34 : i32
  %36 = memref.alloc(%8, %3, %7, %5) : memref<?x?x?x?xf32, "gpu">
  %37 = memref.alloc(%2, %4, %9, %6) : memref<?x?x?x?xf32, "gpu">
  "lmhlo.transpose"(%1, %37) {disc.device = "gpu", permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<?x?x?x?xf32, "gpu">, memref<?x?x?x?xf32, "gpu">) -> ()
  %38 = cmpi sle, %22, %23 : i32
  %39 = select %38, %22, %23 : i32
  %40 = subi %22, %39 : i32
  %41 = subi %23, %39 : i32
  %42 = cmpi sle, %34, %35 : i32
  %43 = select %42, %34, %35 : i32
  %44 = subi %34, %43 : i32
  %45 = subi %35, %43 : i32
  %46 = memref.alloca() : memref<4xi32, "cpu">
  memref.store %c0_i32, %46[%c0_1] : memref<4xi32, "cpu">
  memref.store %c0_i32, %46[%c1_0] : memref<4xi32, "cpu">
  memref.store %40, %46[%c2] : memref<4xi32, "cpu">
  memref.store %44, %46[%c3] : memref<4xi32, "cpu">
  %47 = memref.alloca() : memref<4xi32, "cpu">
  memref.store %c0_i32, %47[%c0_1] : memref<4xi32, "cpu">
  memref.store %c0_i32, %47[%c1_0] : memref<4xi32, "cpu">
  memref.store %41, %47[%c2] : memref<4xi32, "cpu">
  memref.store %45, %47[%c3] : memref<4xi32, "cpu">
  %48 = memref.alloca() : memref<4xi32, "cpu">
  memref.store %39, %48[%c0_1] : memref<4xi32, "cpu">
  memref.store %39, %48[%c1_0] : memref<4xi32, "cpu">
  memref.store %43, %48[%c2] : memref<4xi32, "cpu">
  memref.store %43, %48[%c3] : memref<4xi32, "cpu">
  %49 = addi %12, %40 : i32
  %50 = addi %49, %41 : i32
  %51 = addi %24, %44 : i32
  %52 = addi %51, %45 : i32
  %53 = index_cast %50 : i32 to index
  %54 = index_cast %52 : i32 to index
  %55 = memref.alloc(%8, %3, %53, %54) : memref<?x?x?x?xf32, "gpu">
  "lmhlo.fusion"() ( {
    "lmhlo.constant"(%11) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, "gpu">) -> ()
    "lmhlo.transpose"(%0, %36) {disc.device = "gpu", permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (memref<?x?x?x?xf32, "gpu">, memref<?x?x?x?xf32, "gpu">) -> ()
    "lmhlo.dynamic_pad"(%36, %11, %46, %47, %10, %55) {disc.device = "gpu"} : (memref<?x?x?x?xf32, "gpu">, memref<f32, "gpu">, memref<4xi32, "cpu">, memref<4xi32, "cpu">, memref<4xi32, "cpu">, memref<?x?x?x?xf32, "gpu">) -> ()
    "lmhlo.terminator"() : () -> ()
  }) {disc.fusion.name = "main_kLoop_dynamic_pad__3_1_0"} : () -> ()
  memref.dealloc %36 : memref<?x?x?x?xf32, "gpu">
  memref.dealloc %11 : memref<f32, "gpu">
  %56 = addi %39, %39 : i32
  %57 = addi %50, %56 : i32
  %58 = subi %57, %13 : i32
  %59 = divi_signed %58, %c3_i32 : i32
  %60 = addi %59, %c1_i32 : i32
  %61 = addi %43, %43 : i32
  %62 = addi %52, %61 : i32
  %63 = subi %62, %25 : i32
  %64 = divi_signed %63, %c3_i32 : i32
  %65 = addi %64, %c1_i32 : i32
  %66 = index_cast %60 : i32 to index
  %67 = index_cast %65 : i32 to index
  %68 = memref.alloc(%8, %2, %66, %67) : memref<?x?x?x?xf32, "gpu">
  // CHECK: memref.store {{.*}} : memref<16xi32, "cpu">
  // CHECK: "disc_ral.dispatch"
  // CHECK-SAME: call_target_name = "ral_conv"
  // CHECK-NOT: lmhlo.dynamic_conv
  "lmhlo.dynamic_conv"(%55, %37, %48, %68) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 1 : i64, input_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>, kernel_input_feature_dimension = 1 : i64, kernel_output_feature_dimension = 0 : i64, kernel_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 1 : i64, output_spatial_dimensions = dense<[2, 3]> : tensor<2xi64>}, disc.device = "gpu", feature_group_count = 1 : i64, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<3> : tensor<2xi64>} : (memref<?x?x?x?xf32, "gpu">, memref<?x?x?x?xf32, "gpu">, memref<4xi32, "cpu">, memref<?x?x?x?xf32, "gpu">) -> ()
  memref.dealloc %55 : memref<?x?x?x?xf32, "gpu">
  memref.dealloc %37 : memref<?x?x?x?xf32, "gpu">
  %69 = memref.alloc(%8, %66, %67, %2) : memref<?x?x?x?xf32, "gpu">
  "lmhlo.transpose"(%68, %69) {disc.device = "gpu", permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (memref<?x?x?x?xf32, "gpu">, memref<?x?x?x?xf32, "gpu">) -> ()
  memref.dealloc %68 : memref<?x?x?x?xf32, "gpu">
  %c0_2 = constant 0 : index
  "disc_ral.send_output"(%arg0, %c0_2, %69) : (!disc_ral.context, index, memref<?x?x?x?xf32, "gpu">) -> ()
  return
}
