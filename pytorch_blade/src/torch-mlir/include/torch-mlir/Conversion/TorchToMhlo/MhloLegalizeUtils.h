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

#ifndef TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZEUTILS_H
#define TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZEUTILS_H

#include "mlir/Dialect/Quant/QuantTypes.h" // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h" // from @llvm-project
#include "mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h" // from @llvm-project

#include "torch-mlir/Conversion/TorchToMhlo/ShapeUtils.h"

namespace mlir {
namespace mhlo {

// Create a 32-bit float constant operator from a float
Value getMhloConstTensorSingleF32(
    PatternRewriter& rewriter,
    Operation* op,
    float val);

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 MHLO constant, need to pass in llvm::APInt instead.
template <typename T>
llvm::Optional<Value> getConstTensor(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<T> vec,
    ArrayRef<int64_t> shape);

// Creates a MHLO operation and performs shape inference on the individual
// op. This allows shape inference during the framework to MHLO lowering.
template <typename MhloOp, typename... Args>
MhloOp CreateOpAndInfer(
    PatternRewriter& rewriter,
    Location loc,
    Type result_ty,
    Args&&... args) {
  auto op = rewriter.create<MhloOp>(loc, result_ty, args...);

  InferShapedTypeOpInterface shapeInterface =
      dyn_cast<InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  SmallVector<ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(
              op.getContext(),
              op.getLoc(),
              op->getOperands(),
              op->getAttrDictionary(),
              op->getRegions(),
              returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  auto result = op->getResult(0);
  auto predictedShape = returnedShapes[0];
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(result_ty);

  // Compute the knowledge based on the inferred type.
  auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
  inferredKnowledge.dtype = result_ty.cast<ShapedType>().getElementType();
  inferredKnowledge.hasRank = predictedShape.hasRank();
  if (predictedShape.hasRank()) {
    for (auto dim : predictedShape.getDims()) {
      inferredKnowledge.sizes.push_back(dim);
    }
  }

  // Compute the new type based on the joined version.
  auto newKnowledge = ValueKnowledge::join(currentKnowledge, inferredKnowledge);
  auto new_ty = newKnowledge.getType();
  result.setType(new_ty);
  return op;
}

template <typename MhloOp, typename... Args>
void CreateReplaceOpAndInfer(
    PatternRewriter& rewriter,
    Operation* op,
    Type result_ty,
    Args&&... args) {
  auto result =
      CreateOpAndInfer<MhloOp>(rewriter, op->getLoc(), result_ty, args...);
  rewriter.replaceOp(op, result->getResults());
}

llvm::Optional<Value> getMhloShapeOfTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value& value);
} // namespace mhlo
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZEUTILS_H
