// RUN: disc-opt -disc-lower-tf -split-input-file %s | FileCheck %s

// CHECK-LABEL: main
func.func @main(%input: tensor<?x?xf32>, %scale: tensor<?xf32>, %zero_point: tensor<?xi32>) ->tensor<?x?xf32> attributes{tf.entry_function = {control_outputs = "", inputs = "arg_input_0_0_0_arg,arg_scale_0_1_0_arg,arg_zero_point_0_2_0_arg", outputs = "DiscFakeQuant:0"}} {
  // CHECK: mhlo_disc.fake_quant
  // CHECK-SAME: axis = dense<1>
  // CHECK-SAME: num_bits = 4
  // CHECK-SAME: quant_max = 3
  // CHECK-SAME: quant_min = -4
  // CHECK-SAME: use_dynamic = false
  // CHECK-SAME: use_signed = true
  // CHECK-SAME: use_symmetric = true
  %output = "tf.DiscFakeQuant"(%input, %scale, %zero_point) {_XlaAlreadyClustered = true, axis = [1], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 4 : i64, quant_max = 3: i64, quant_min = -4: i64} : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
  func.return %output : tensor<?x?xf32>
}
