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

//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOMHLO_TORCHTOMHLO_H
#define TORCHMLIR_CONVERSION_TORCHTOMHLO_TORCHTOMHLO_H

#include <memory>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace torch {
namespace Torch {
class TorchLoweringPipelineOptions;
LogicalResult reduceTensorConversions(func::FuncOp& func);
} // namespace Torch

namespace TorchConversion {
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToMhloPass();
std::unique_ptr<OperationPass<func::FuncOp>> createApplyValueSemanticsPass();
std::unique_ptr<OperationPass<ModuleOp>> createVerifyMhloBackendContractPass();

void createTorchBackendToMhloBackendPipeline(
    OpPassManager& pm,
    const torch::Torch::TorchLoweringPipelineOptions& options);
} // namespace TorchConversion
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOMHLO_TORCHTOMHLO_H
