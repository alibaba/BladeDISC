// RUN: disc-opt -disc-to-llvm -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: test_ral_alloc_free
module @main attributes {gpu.container_module}  {
  func.func @test_ral_alloc_free(%arg0: !disc_ral.context) {
    // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<array<32 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @disc_ral_call({{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()
    // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<array<35 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @disc_ral_call({{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()
    // CHECK-NOT: llvm.call @malloc
    // CHECK-NOT: llvm.call @free
    %c1 = arith.constant 1024 : index
    %c2 = arith.constant 1024 : index
    %c3 = arith.constant 1024 : index
    %arg1 = memref.alloc(%c2, %c3) : memref<?x?xf32, "gpu">
    memref.dealloc %arg1 : memref<?x?xf32, "gpu">
    return
  }
}

// -----

// CHECK-LABEL: test_llvm_alloc_free
module @main attributes {gpu.container_module}  {
  func.func @test_llvm_alloc_free(%arg0: !disc_ral.context) {
    // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<array<32 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @disc_ral_call({{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()
    // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<array<35 x i8>>, i64, i64) -> !llvm.ptr<i8>
    // CHECK: llvm.call @disc_ral_call({{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>) -> ()
    // CHECK-NOT: llvm.call @malloc
    // CHECK-NOT: llvm.call @free
    %c2 = arith.constant 1024 : index
    %c3 = arith.constant 1024 : index
    %arg1 = memref.alloc(%c2, %c3) : memref<?x?xf32, "cpu">
    memref.dealloc %arg1 : memref<?x?xf32, "cpu">
    return
  }
}