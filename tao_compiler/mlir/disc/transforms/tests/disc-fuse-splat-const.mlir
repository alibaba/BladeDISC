// RUN: disc-opt -split-input-file -disc-fuse-splat-const %s | FileCheck %s

// CHECK-LABEL: @fuse_without_erase
func.func @fuse_without_erase(%arg0: memref<2x3xf32, #gpu.address_space<global>>) -> memref<2x3xf32, #gpu.address_space<global>> {
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  %0 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  %1 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  %2 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  %3 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  // CHECK: lmhlo.constant
  "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<2x3xf32>} : (memref<2x3xf32, #gpu.address_space<global>>) -> ()
  // CHECK: lmhlo.fusion
  "lmhlo.fusion"() ({
    // CHECK: lmhlo.constant
    "lmhlo.abs"(%arg0, %1) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.add"(%1, %0, %2) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.abs"(%0, %3) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
  return %3 : memref<2x3xf32, #gpu.address_space<global>>
}

// CHECK-LABEL: @fuse_and_duplicate_with_erase
func.func @fuse_and_duplicate_with_erase(%arg0: memref<2x3xf32, #gpu.address_space<global>>) -> (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) {
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  %0 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  %1 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  %2 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  %3 = memref.alloc() : memref<2x3xf32, #gpu.address_space<global>>
  // CHECK-NOT: lmhlo.constant
  "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<2x3xf32>} : (memref<2x3xf32, #gpu.address_space<global>>) -> ()
  // CHECK: lmhlo.fusion
  "lmhlo.fusion"() ({
    // CHECK: lmhlo.constant
    "lmhlo.abs"(%arg0, %1) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.add"(%1, %0, %2) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: lmhlo.fusion
  "lmhlo.fusion"() ({
    // CHECK: lmhlo.constant
    "lmhlo.abs"(%0, %1) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.add"(%1, %arg0, %3) : (memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  return %2, %3 : memref<2x3xf32, #gpu.address_space<global>>, memref<2x3xf32, #gpu.address_space<global>>
}
