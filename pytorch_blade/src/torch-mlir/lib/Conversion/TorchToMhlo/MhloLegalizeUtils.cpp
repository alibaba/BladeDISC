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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo

#include "torch-mlir/Conversion/TorchToMhlo/MhloLegalizeUtils.h"

namespace mlir {
namespace mhlo {

// Create a 32-bit float constant operator from a float
Value getMhloConstTensorSingleF32(
    PatternRewriter& rewriter,
    Operation* op,
    float val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
llvm::Optional<Value> getConstTensor(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<T> vec,
    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
llvm::Optional<Value> getConstTensor<APInt>(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<APInt> vec,
    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
llvm::Optional<Value> getConstTensor<float>(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<float> vec,
    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template llvm::Optional<Value> getConstTensor<int32_t>(
    PatternRewriter&,
    Operation*,
    ArrayRef<int32_t> vec,
    ArrayRef<int64_t> shape);

template llvm::Optional<Value> getConstTensor<int64_t>(
    PatternRewriter&,
    Operation*,
    ArrayRef<int64_t> vec,
    ArrayRef<int64_t> shape);

std::vector<Value> getDimSizesOfTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value& value) {
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(value.getType());
  std::vector<Value> dim_sizes;
  if (!currentKnowledge.hasRank) {
    return dim_sizes;
  }

  auto rank = currentKnowledge.sizes.size();
  dim_sizes.reserve(rank);
  auto loc = op->getLoc();
  for (auto d = 0; d < rank; ++d) {
    auto d_size = currentKnowledge.sizes[d];
    // if (d_size == mlir::ShapedType::kDynamicSize) {
    dim_sizes.emplace_back(rewriter.create<mlir::arith::IndexCastOp>(
        loc,
        rewriter.getI32Type(),
        rewriter.create<tensor::DimOp>(loc, value, d)));
    // } else {
    //   dim_sizes.emplace_back(
    //     rewriter.create<mlir::arith::ConstantOp>(
    //       loc, rewriter.getI32IntegerAttr(d_size)));
    // }
  }
  return dim_sizes;
}

llvm::Optional<Value> getMhloShapeOfTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value& value) {
  auto dim_sizes = getDimSizesOfTensor(rewriter, op, value);
  if (dim_sizes.size() == 0) {
    return llvm::None;
  }
  return rewriter.create<mlir::tensor::FromElementsOp>(op->getLoc(), dim_sizes)
      .getResult();
}

} // namespace mhlo
} // namespace mlir
