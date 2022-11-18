// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch_blade_custom_quantize_per_tensor_symmetric
func.func @torch_blade_custom_quantize_per_tensor_symmetric(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],si8> {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<0.545411646> : tensor<f32>) : !torch.vtensor<[],f32>
  %1 = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: mhlo_disc.quantize
  // CHECK-SAME: axis = dense<>
  // CHECK-SAME: quant_max = 127
  // CHECK-SAME: quant_min = -128
  // CHECK-SAME: round_mode = 1
  // CHECK-SAME: use_dynamic = false
  // CHECK-SAME: use_symmetric = true
  // CHECK-SAME: tensor<?x?x?xi8>
  %3 = torch.operator "torch_blade.quantize"(%arg0, %0, %1, %int-128, %int127, %int8, %2, %true, %true, %false, %false) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[?,?,?],si8>
  // CHECK: return
  return %3 : !torch.vtensor<[?,?,?],si8>
}

// CHECK-LABEL:  func.func @torch_blade_custom_quantize_per_channel_symmetric
func.func @torch_blade_custom_quantize_per_channel_symmetric(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],si8> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense_resource<__elided__> : tensor<128xf32>) : !torch.vtensor<[128],f32>
  %1 = torch.vtensor.literal(dense<0> : tensor<128xsi32>) : !torch.vtensor<[128],si32>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  // CHECK: mhlo_disc.quantize
  // CHECK-SAME: axis = dense<0>
  // CHECK-SAME: quant_max = 127
  // CHECK-SAME: quant_min = -128
  // CHECK-SAME: round_mode = 1
  // CHECK-SAME: use_dynamic = false
  // CHECK-SAME: use_symmetric = true
  // CHECK-SAME: tensor<128x128xi8>
  %3 = torch.operator "torch_blade.quantize"(%arg0, %0, %1, %int-128, %int127, %int8, %2, %true, %true, %false, %false) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],si32>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[128,128],si8>
  // CHECK: return
  return %3 : !torch.vtensor<[128,128],si8>
}

// CHECK-LABEL:  func.func @torch_blade_custom_dequantize_per_tensor_symmetric
func.func @torch_blade_custom_dequantize_per_tensor_symmetric(%arg0: !torch.vtensor<[?,?,?],si8>) -> !torch.vtensor<[?,?,?],f32> {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense<0.545411646> : tensor<f32>) : !torch.vtensor<[],f32>
  %1 = torch.vtensor.literal(dense<0> : tensor<si32>) : !torch.vtensor<[],si32>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK: mhlo_disc.dequantize
  // CHECK-SAME: axis = dense<>
  // CHECK-SAME: round_mode = 1
  // CHECK-SAME: use_dynamic = false
  // CHECK-SAME: use_symmetric = true
  // CHECK-SAME: tensor<?x?x?xf32>
  %3 = torch.operator "torch_blade.dequantize"(%arg0, %0, %1, %int-128, %int127, %int8, %2, %true, %true, %false, %false) : (!torch.vtensor<[?,?,?],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[?,?,?],f32>
  // CHECK: return
  return %3 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch_blade_custom_dequantize_per_channel_symmetric
func.func @torch_blade_custom_dequantize_per_channel_symmetric(%arg0: !torch.vtensor<[128,128],si8>) -> !torch.vtensor<[128,128],f32> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %0 = torch.vtensor.literal(dense_resource<__elided__> : tensor<128xf32>) : !torch.vtensor<[128],f32>
  %1 = torch.vtensor.literal(dense<0> : tensor<128xsi32>) : !torch.vtensor<[128],si32>
  %int-128 = torch.constant.int -128
  %int127 = torch.constant.int 127
  %int8 = torch.constant.int 8
  %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  // CHECK: mhlo_disc.dequantize
  // CHECK-SAME: axis = dense<0>
  // CHECK-SAME: round_mode = 1
  // CHECK-SAME: use_dynamic = false
  // CHECK-SAME: use_symmetric = true
  // CHECK-SAME: tensor<128x128xi8>
  %3 = torch.operator "torch_blade.dequantize"(%arg0, %0, %1, %int-128, %int127, %int8, %2, %true, %true, %false, %false) : (!torch.vtensor<[128,128],si8>, !torch.vtensor<[128],f32>, !torch.vtensor<[128],si32>, !torch.int, !torch.int, !torch.int, !torch.list<int>, !torch.bool, !torch.bool, !torch.bool, !torch.bool) -> !torch.vtensor<[128,128],f32>
  // CHECK: return
  return %3 : !torch.vtensor<[128,128],f32>
}
