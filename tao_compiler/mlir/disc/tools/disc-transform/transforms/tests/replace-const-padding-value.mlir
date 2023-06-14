// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @replace_const_padding_value
func.func @replace_const_padding_value(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[T0:.*]] = disc_linalg_ext.padding_value_placeholder padding_mode(kAny), value(0.000000e+00 : f32)
  // CHECK-NEXT: tensor.pad
  // CHECK-NEXT: ^bb0
  // CHECK-NEXT:   tensor.yield %[[T0]] : f32
  %out = tensor.pad %arg0 low[%c1] high[%c1] {
  ^bb0(%arg1: index):
    tensor.yield %cst : f32
  } : tensor<?xf32> to tensor<?xf32>
  return %out : tensor<?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %pad = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.disc.replace_const_padding_value %pad mode("kAny")
}