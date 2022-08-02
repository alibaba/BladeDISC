// RUN: disc-opt -disc-math-approximation %s -o - | FileCheck %s

// CHECK-LABEL: @exp
func.func @exp(%arg0 : f32) -> f32 {
  // CHECK-NOT: math.exp
  %0 = math.exp %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @log
func.func @log(%arg0 : f32) -> f32 {
  // CHECK-NOT: math.log
  %0 = math.log %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @tanh
func.func @tanh(%arg0 : f32) -> f32 {
  // CHECK-NOT: math.tanh
  %0 = math.tanh %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @sin
func.func @sin(%arg0 : f32) -> f32 {
  // CHECK-NOT: math.sin
  %0 = math.sin %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @cos
func.func @cos(%arg0 : f32) -> f32 {
  // CHECK-NOT: math.cos
  %0 = math.cos %arg0 : f32
  return %0 : f32
}