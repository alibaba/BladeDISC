/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
// #include "mlir/Pass/Pass.h"
// #include "transforms/PassDetail.h"
//
// namespace mlir {
// namespace scf {
// class SCFDialect;
// }
// namespace memref {
// class MemRefDialect;
// }
// namespace disc_ral {
// namespace {
//
// struct MapParallelLoopsPass : MapParallelLoopsPassBase<MapParallelLoopsPass>
// {
//   void runOnOperation() override {
//     mlir::greedilyMapParallelSCFToGPU((getOperation()).getBody());
//   }
// };
//
// }  // namespace
//
// std::unique_ptr<OperationPass<func::FuncOp>> createMapParallelLoopsPass() {
//   return std::make_unique<MapParallelLoopsPass>();
// }
//
// }  // namespace disc_ral
// }  // namespace mlir
