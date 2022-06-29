// RUN: torch-mlir-opt <%s --torch-backend-to-mhlo-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s
func @torch.aten.empty() -> !torch.vtensor<[2,3], f32> {
  %none = torch.constant.none
  %1 = torch.constant.int 2
  %2 = torch.constant.int 3
  %3 = torch.prim.ListConstruct %1, %2 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.empty.memory_format %3, %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
  return %4 : !torch.vtensor<[2, 3], f32>
}