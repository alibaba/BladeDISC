// RUN: disc-opt -split-input-file --disc-bf16-expansion  %s | FileCheck %s

// CHECK-LABEL: @truncf_f32
// CHECK-NOT: arith.truncf
func.func @truncf_f32(%arg0 : f32) -> bf16 {
  %0 = arith.truncf %arg0 : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @truncf_vector_f32
// CHECK-NOT: arith.truncf
func.func @truncf_vector_f32(%arg0 : vector<4xf32>) -> vector<4xbf16> {
  %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xbf16>
  return %0 : vector<4xbf16>
}

// CHECK-LABEL: @extf_bf16
// CHECK-NOT: arith.extf
func.func @extf_bf16(%arg0 : bf16) -> f32 {
  %0 = arith.extf %arg0 : bf16 to f32
  return %0 : f32
}

// CHECK-LABEL: @extf_vector_bf16
// CHECK-NOT: arith.extf
func.func @extf_vector_bf16(%arg0 : vector<4xbf16>) -> vector<4xf32> {
  %0 = arith.extf %arg0 : vector<4xbf16> to vector<4xf32>
  return %0 : vector<4xf32>
}
