// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @tensor.empty
func.func @tensor.empty(%arg0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T0:.*]] = memref.alloc({{.*}})
  // CHECK-NEXT: return %[[T0]]
  %0 = tensor.empty(%arg0, %arg0) : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.bufferize %arg1
}

// -----

// CHECK-LABEL: @use_alloca
func.func @use_alloca() -> tensor<20x20xf32> {
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<10x10xf32>
  // CHECK: linalg.fill ins({{.*}}) outs(%[[ALLOCA]] : memref<10x10xf32>)
  // CHECK: %[[RET:.*]] = memref.alloc() {alignment = 64 : i64} : memref<20x20xf32>
  // CHECK-NOT: dealloc
  // CHECK: return %[[RET]]
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = tensor.empty() : tensor<10x10xf32>
  %1 = linalg.fill ins(%cst1 : f32) outs(%0 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = tensor.pad %1 low[%c0, %c0] high[%c10, %c10] {
  ^bb0(%arg12: index, %arg13: index):
    tensor.yield %cst0 : f32
  } : tensor<10x10xf32> to tensor<20x20xf32>
  return %2 : tensor<20x20xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.bufferize %arg1
}

// -----

// CHECK-LABEL: @not_use_alloca_due_to_too_large
func.func @not_use_alloca_due_to_too_large() -> tensor<2000x2000xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1000x1000xf32>
  // CHECK: %[[RET:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2000x2000xf32>
  // CHECK: dealloc %[[ALLOC]]
  // CHECK: return %[[RET]]
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %0 = tensor.empty() : tensor<1000x1000xf32>
  %1 = linalg.fill ins(%cst1 : f32) outs(%0 : tensor<1000x1000xf32>) -> tensor<1000x1000xf32>
  %2 = tensor.pad %1 low[%c0, %c0] high[%c1000, %c1000] {
  ^bb0(%arg12: index, %arg13: index):
    tensor.yield %cst0 : f32
  } : tensor<1000x1000xf32> to tensor<2000x2000xf32>
  return %2 : tensor<2000x2000xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.bufferize %arg1
}

// -----

// CHECK-LABEL: @not_use_alloca_due_to_dynamic_shape
func.func @not_use_alloca_due_to_dynamic_shape(%arg0: index) -> tensor<?x?xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc({{.*}}) {alignment = 64 : i64} : memref<?x?xf32>
  // CHECK: %[[RET:.*]] = memref.alloc({{.*}}) {alignment = 64 : i64} : memref<?x?xf32>
  // CHECK: dealloc %[[ALLOC]]
  // CHECK: return %[[RET]]
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %0 = tensor.empty(%arg0, %arg0) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst1 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[%c0, %c0] high[%c1000, %c1000] {
  ^bb0(%arg12: index, %arg13: index):
    tensor.yield %cst0 : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  transform.disc.bufferize %arg1
}