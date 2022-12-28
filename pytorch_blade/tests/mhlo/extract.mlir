// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.item.f64toi64(
// CHECK:  arith.addf
// CHECK:  arith.subf
// CHECK:  arith.cmpf
// CHECK:  arith.select
// CHECK:  arith.minf
// CHECK:  arith.maxf
// CHECK:  arith.fptosi
func.func @torch.aten.item.f64toi64(%arg0: !torch.vtensor<[],f64>) -> !torch.vtensor<[],si64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],f64> -> !torch.int
  %2 = torch.aten.tensor.int %1, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %2 : !torch.vtensor<[],si64>
}

// CHECK-LABEL:  func.func @torch.aten.item.f32toi64(
// CHECK:  arith.addf
// CHECK:  arith.subf
// CHECK:  arith.cmpf
// CHECK:  arith.select
// CHECK:  arith.minf
// CHECK:  arith.maxf
// CHECK:  arith.fptosi
func.func @torch.aten.item.f32toi64(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],si64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],f32> -> !torch.int
  %2 = torch.aten.tensor.int %1, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %2 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.item.i32toi64(
// CHECK:   arith.extsi
func.func @torch.aten.item.i32toi64(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[],si64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.int
  %2 = torch.aten.tensor.int %1, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %2 : !torch.vtensor<[],si64>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.item.i32tof64(
// CHECK:   arith.sitofp
func.func @torch.aten.item.i32tof64(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[],f64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],si32> -> !torch.float
  %2 = torch.aten.tensor.float %1, %none, %none, %false : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f64>
  return %2 : !torch.vtensor<[],f64>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.item.i64tof64(
// CHECK:   arith.sitofp
func.func @torch.aten.item.i64tof64(%arg0: !torch.vtensor<[],si64>) -> !torch.vtensor<[],f64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],si64> -> !torch.float
  %2 = torch.aten.tensor.float %1, %none, %none, %false : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f64>
  return %2 : !torch.vtensor<[],f64>
}

// -----
// CHECK-LABEL:  func.func @torch.aten.item.boolTof64(
// CHECK:   arith.uitofp
func.func @torch.aten.item.boolTof64(%arg0: !torch.vtensor<[],i1>) -> !torch.vtensor<[],f64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],i1> -> !torch.float
  %2 = torch.aten.tensor.float %1, %none, %none, %false : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f64>
  return %2 : !torch.vtensor<[],f64>
}

// -----
// si64 treated as uint here, for torch does not support uint
// CHECK-LABEL:  func.func @torch.aten.item.boolToi64(
// CHECK:   arith.extui
func.func @torch.aten.item.boolToi64(%arg0: !torch.vtensor<[],i1>) -> !torch.vtensor<[],si64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %1 = torch.aten.item %arg0 : !torch.vtensor<[],i1> -> !torch.int
  %2 = torch.aten.tensor.int %1, %none, %none, %false : !torch.int, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %2 : !torch.vtensor<[],si64>
}
