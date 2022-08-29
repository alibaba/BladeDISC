// RUN: disc-opt -disc-const-to-ral %s | FileCheck %s

// CHECK-DAG: llvm.mlir.global internal constant @__global_const_1
// CHECK-DAG: llvm.mlir.global internal constant @__global_const_0
// CHECK-LABEL: simple_test
func.func @simple_test(%arg0: !disc_ral.context) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc() : memref<100x100xf32, "gpu">
  %1 = memref.alloc() : memref<100x100xf32, "cpu">
  // CHECK: %[[C0_CPU:.*]] = arith.constant 0 : i32
  // CHECK: %[[N0:.*]] = "disc_ral.dispatch"(%[[CTX:.*]], %[[T0:.*]], %[[T1:.*]], %[[C0_CPU]]) {backend_config = "gpu", call_target_name = "ral_const", has_side_effect = false}
  // CHECK: %[[C0_GPU:.*]] = arith.constant 0 : i32
  // CHECK: %[[N1:.*]] = "disc_ral.dispatch"(%[[CTX]], %[[T0:.*]], %[[T1:.*]], %[[C0_GPU]]) {backend_config = "cpu", call_target_name = "ral_const", has_side_effect = false}
  // CHECK: "disc_ral.send_output"(%[[CTX]], %c0, %[[N0]])
  // CHECK: "disc_ral.send_output"(%[[CTX]], %c1, %[[N1]])
  "lmhlo.constant"(%0) {disc.device = "gpu", value = dense<1.000000e+00> : tensor<100x100xf32>} : (memref<100x100xf32, "gpu">) -> ()
  "lmhlo.constant"(%1) {disc.device = "cpu", value = dense<1.000000e+00> : tensor<100x100xf32>} : (memref<100x100xf32, "cpu">) -> ()
  "disc_ral.send_output"(%arg0, %c0, %0) : (!disc_ral.context, index, memref<100x100xf32, "gpu">) -> ()
  "disc_ral.send_output"(%arg0, %c1, %1) : (!disc_ral.context, index, memref<100x100xf32, "cpu">) -> ()
  return
}
