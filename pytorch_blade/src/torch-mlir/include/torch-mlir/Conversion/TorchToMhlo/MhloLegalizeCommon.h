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

#ifndef TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZECOMMON_H
#define TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZECOMMON_H

#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Support/LLVM.h" // from @llvm-project

namespace mlir {
namespace mhlo {

// Lowers ReduceAll to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceAllOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

// Lowers ReduceAny to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceAnyOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

// Lowers ReduceMin to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceMinOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

// Lowers ReduceMax to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceMaxOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

// Lowers ReduceProd to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceProdOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

// Lowers ReduceSum to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceSumOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

// Lowers ReduceMean to a sequence of MHLO ops.
llvm::Optional<Value> convertReduceMeanOp(
    PatternRewriter& rewriter,
    Operation* op,
    RankedTensorType output_type,
    Value input_value,
    ElementsAttr axes_elems,
    bool keep_dims);

} // namespace mhlo
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZECOMMON_H
