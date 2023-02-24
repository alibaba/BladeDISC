// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_DISC_MHLO_PASSES_H
#define TORCHMLIR_CONVERSION_DISC_MHLO_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

namespace mlir {
class ModuleOp;

namespace torch {
namespace TorchConversion {
#define GEN_PASS_CLASSES
#include "torch-mlir/Conversion/MhloPasses.h.inc"
} // namespace TorchConversion

/// Registers all TorchToMhlo conversion passes.
void registerTorchToMhloPasses();

void createDiscTorchBackendToMhloBackendPipeline(
    OpPassManager& pm,
    const torch::Torch::TorchLoweringPipelineOptions& options);

// Creates a pipeline that lowers a flat list of funcs and global slots
// with the torch and aten dialects and mutable arrays and converts it to
// the form required by torch-verify-backend-contract.
void createDiscTorchFunctionToTorchBackendPipeline(
    OpPassManager& pm,
    const ::mlir::torch::Torch::TorchLoweringPipelineOptions& options);

} // namespace torch
} // end namespace mlir

namespace mlir {
class ModuleOp;
namespace torch {
namespace Torch {
class TorchLoweringPipelineOptions;
LogicalResult reduceTensorConversions(func::FuncOp& func);
} // namespace Torch

namespace TorchConversion {
std::unique_ptr<OperationPass<func::FuncOp>> createApplyValueSemanticsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createApplyDiscPdlPatternsPass(
    const std::string& pdll_files = "",
    const std::string& pdll_include_dirs = "");
std::unique_ptr<OperationPass<func::FuncOp>> createDiscConvertTorchToMhloPass();
std::unique_ptr<OperationPass<func::FuncOp>> createDiscSimplifyPatternsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createDiscDecomposeComplexOpsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createDiscConvertTorchToDiscMhlo();
std::unique_ptr<OperationPass<func::FuncOp>> createDiscUpgradeLegacyOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createVerifyMhloBackendContractPass();
} // namespace TorchConversion
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_DISC_MHLO_PASSES_H
