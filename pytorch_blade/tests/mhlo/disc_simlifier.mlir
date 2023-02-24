// RUN: torch-mlir-opt <%s --disc-simplify-patterns --canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @disc_simplifier_pass(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f16>) -> !torch.vtensor<[?,?,?],f16>
// CHECK: return %[[ARG0]] : !torch.vtensor<[?,?,?],f16>
func.func @disc_simplifier_pass(%arg0: !torch.vtensor<[?,?,?],f16>) -> !torch.vtensor<[?,?,?],f16> {
  %int0 = torch.constant.int 0
  %1 = torch.prim.ListConstruct %arg0 : (!torch.vtensor<[?,?,?],f16>) -> !torch.list<vtensor>
  %2 = torch.aten.__getitem__.t %1, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?,?],f16>
  return %2 : !torch.vtensor<[?,?,?],f16>
}
