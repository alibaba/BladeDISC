// RUN: disc-opt -disc-lower-tf -split-input-file %s -verify-diagnostics


func.func @main(%input: tensor<?x?xf32>, %scale: tensor<?xf32>, %zero_point: tensor<?xi32>) ->tensor<?x?xf32> attributes{tf.entry_function = {control_outputs = "", inputs = "arg_input_0_0_0_arg,arg_scale_0_1_0_arg,arg_zero_point_0_2_0_arg", outputs = "DiscFakeQuant:0"}} {
  // expected-error@+1 {{dynamic quantization is not supported yet}}
  %output = "tf.DiscFakeQuant"(%input, %scale, %zero_point) {_XlaAlreadyClustered = true, axis = [1], use_dynamic = true, use_signed = true, use_symmetric = true, num_bits = 4 : i64, quant_max = 3: i64, quant_min = -4: i64} : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
  func.return %output : tensor<?x?xf32>
}


// -----

func.func @main(%input: tensor<?x?xf32>, %scale: tensor<?xf32>, %zero_point: tensor<?xi32>) ->tensor<?x?xf32> attributes{tf.entry_function = {control_outputs = "", inputs = "arg_input_0_0_0_arg,arg_scale_0_1_0_arg,arg_zero_point_0_2_0_arg", outputs = "DiscFakeQuant:0"}} {
  // expected-error@+1 {{quant_min (20) must be less than quant_max (10)}}
  %output = "tf.DiscFakeQuant"(%input, %scale, %zero_point) {_XlaAlreadyClustered = true, axis = [1], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 10 : i64, quant_min = 20 : i64} : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
  func.return %output : tensor<?x?xf32>
}


// -----

func.func @main(%input: tensor<?x?xf32>, %scale: tensor<?xf32>, %zero_point: tensor<?xi32>) ->tensor<?x?xf32> attributes{tf.entry_function = {control_outputs = "", inputs = "arg_input_0_0_0_arg,arg_scale_0_1_0_arg,arg_zero_point_0_2_0_arg", outputs = "DiscFakeQuant:0"}} {
  // expected-error@+1 {{op quant_max must not be greater than 127 under 8 bits and signed=1, but got: 130}}
  %output = "tf.DiscFakeQuant"(%input, %scale, %zero_point) {_XlaAlreadyClustered = true, axis = [1], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 130 : i64, quant_min = -128 : i64} : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
  func.return %output : tensor<?x?xf32>
}

// -----

func.func @main(%input: tensor<?x?xf32>, %scale: tensor<?xf32>, %zero_point: tensor<?xi32>) ->tensor<?x?xf32> attributes{tf.entry_function = {control_outputs = "", inputs = "arg_input_0_0_0_arg,arg_scale_0_1_0_arg,arg_zero_point_0_2_0_arg", outputs = "DiscFakeQuant:0"}} {
  // expected-error@+1 {{op quant_min must not be less than -128 under 8 bits and signed=1, but got: -130}}
  %output = "tf.DiscFakeQuant"(%input, %scale, %zero_point) {_XlaAlreadyClustered = true, axis = [1], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -130 : i64} : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
  func.return %output : tensor<?x?xf32>
}

// -----

func.func @main(%input: tensor<?x?xf32>, %scale: tensor<?xf32>, %zero_point: tensor<?xi32>) ->tensor<?x?xf32> attributes{tf.entry_function = {control_outputs = "", inputs = "arg_input_0_0_0_arg,arg_scale_0_1_0_arg,arg_zero_point_0_2_0_arg", outputs = "DiscFakeQuant:0"}} {
  // expected-error@+1 {{axis must be empty (per-tensor) or a single element array (per-channel), but got: [1, 2]}}
  %output = "tf.DiscFakeQuant"(%input, %scale, %zero_point) {_XlaAlreadyClustered = true, axis = [1, 2], use_dynamic = false, use_signed = true, use_symmetric = true, num_bits = 8 : i64, quant_max = 127 : i64, quant_min = -128 : i64} : (tensor<?x?xf32>, tensor<?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
  func.return %output : tensor<?x?xf32>
}


