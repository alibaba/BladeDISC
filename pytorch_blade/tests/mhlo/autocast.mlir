// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.autocast_reduced(
func.func @torch.aten.autocast_reduced(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f16>){
  // CHECK:  mhlo.convert
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %int5 = torch.constant.int 5
  %int15 = torch.constant.int 15
  %result = torch.operator "aten._autocast_to_reduced_precision"(%arg0, %true, %false, %int5, %int15): (!torch.vtensor<[?,?],f32>, !torch.bool, !torch.bool, !torch.int, !torch.int) -> !torch.vtensor<[?,?],f16>
  return %result : !torch.vtensor<[?,?],f16>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.autocast_reduced_no_cast(
// CHECK-NOT:   mhlo.convert
func.func @torch.aten.autocast_reduced_no_cast(%arg0: !torch.vtensor<[?,?],f16>) -> (!torch.vtensor<[?,?],f16>){
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %int5 = torch.constant.int 5
  %int15 = torch.constant.int 15
  %result = torch.operator "aten._autocast_to_reduced_precision"(%arg0, %true, %false, %int5, %int15): (!torch.vtensor<[?,?],f16>, !torch.bool, !torch.bool, !torch.int, !torch.int) -> !torch.vtensor<[?,?],f16>
  return %result : !torch.vtensor<[?,?],f16>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.autocast_full(
// CHECK:   mhlo.convert
func.func @torch.aten.autocast_full(%arg0: !torch.vtensor<[?,?],f16>) -> (!torch.vtensor<[?,?],f32>){
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %result = torch.operator "aten._autocast_to_full_precision"(%arg0, %true, %false): (!torch.vtensor<[?,?],f16>, !torch.bool, !torch.bool) -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.autocast_full_no_cast(
// CHECK-NOT:   mhlo.convert
func.func @torch.aten.autocast_full_no_cast(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,?],f32>){
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %result = torch.operator "aten._autocast_to_full_precision"(%arg0, %true, %false): (!torch.vtensor<[?,?],f32>, !torch.bool, !torch.bool) -> !torch.vtensor<[?,?],f32>
  return %result : !torch.vtensor<[?,?],f32>
}