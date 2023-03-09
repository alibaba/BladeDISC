// RUN: torch-mlir-opt <%s --torch-disc-decompose-complex-ops -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @decompose_split(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f16>) -> (!torch.vtensor<[?,?,?],f16>, !torch.vtensor<[?,?,?],f16>) {
// CHECK:           %[[T0:.*]] = torch.aten.add.int %int0_1, %int1280 : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[T1:.*]] = torch.aten.slice.Tensor %arg0, %int-1, %int0_1, %0, %int1_0 : !torch.vtensor<[?,?,?],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f16>
// CHECK:           %[[T2:.*]] = torch.aten.add.int %0, %int1280 : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[T3:.*]] = torch.aten.slice.Tensor %arg0, %int-1, %0, %2, %int1_0 : !torch.vtensor<[?,?,?],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f16>
// CHECK:           %[[L0:.*]] = torch.prim.ListConstruct %[[T1]], %[[T3]] : (!torch.vtensor<[?,?,?],f16>, !torch.vtensor<[?,?,?],f16>) -> !torch.list<vtensor>
func.func @decompose_split(%arg0: !torch.vtensor<[?,?,?],f16>) -> (!torch.vtensor<[?,?,?],f16>, !torch.vtensor<[?,?,?],f16>) {
  %int1280 = torch.constant.int 1280
  %int-1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %10 = torch.operator "aten.split.Tensor"(%arg0, %int1280, %int-1) : (!torch.vtensor<[?,?,?],f16>, !torch.int, !torch.int) -> !torch.list<vtensor>
  %11 = torch.aten.__getitem__.t %10, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?,?],f16>
  %12 = torch.aten.__getitem__.t %10, %int1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?,?],f16>
  return %11, %12 : !torch.vtensor<[?,?,?],f16>, !torch.vtensor<[?,?,?],f16>
}