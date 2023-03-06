// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

// CHECK-LABEL: @convert_padding_placeholder_to_const
func.func @convert_padding_placeholder_to_const() -> f32 {
  // CHECK-NOT: disc_linalg_ext.padding_value_placeholder
  // CHECK: %[[T0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[T0]] : f32
  %0 = disc_linalg_ext.padding_value_placeholder padding_mode(kAny), value(0.000000e+00 : f32)
  return %0 : f32
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %placeholder = transform.structured.match ops{["disc_linalg_ext.padding_value_placeholder"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.disc.convert_padding_placeholder_to_const %placeholder
}