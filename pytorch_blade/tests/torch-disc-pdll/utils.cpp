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

#include "tests/torch-disc-pdll/utils.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/PatternMatch.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace mlir {
namespace torch {

const std::string kDefaultHelperFunctionDeclarations = R"pdll(
  Constraint CheckTorchNone(v : Value);
  Constraint CheckNotTorchNone(v : Value);
  Constraint CheckTorchConstantInt(v : Value);
  Constraint CheckTorchConstantFloat(v : Value);
  Constraint CheckTorchConstantStr(v : Value);
  Constraint CheckTorchConstantBool(v : Value);
  Constraint CheckTorchConstantIntList(v : Value);

  Rewrite CreateTorchCustomCall(tag : Attr, inputs : ValueRange, outputs : ValueRange) -> (op: Op, new_outputs : ValueRange);
  Rewrite ConvertTorchConstantIntListToI64DenseElemsAttr(cst: Value) -> Attr;
  Rewrite ConvertTorchConstantIntToI64Attr(cst: Value) -> Attr;
)pdll";

static LogicalResult checkTorchNone(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  if (values[0].cast<Value>().getDefiningOp<Torch::ConstantNoneOp>())
    return success();
  return failure();
}

static LogicalResult checkNotTorchNone(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  if (values[0].cast<Value>().getDefiningOp<Torch::ConstantNoneOp>())
    return failure();
  return success();
}

static LogicalResult checkTorchConstantInt(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  int64_t elem;
  if (!matchPattern(values[0].cast<Value>(), Torch::m_TorchConstantInt(&elem)))
    return failure();
  return success();
}

static LogicalResult checkTorchConstantFloat(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  double elem;
  if (!matchPattern(
          values[0].cast<Value>(), Torch::m_TorchConstantFloat(&elem)))
    return failure();
  return success();
}

static LogicalResult checkTorchConstantStr(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  std::string elem;
  if (!matchPattern(values[0].cast<Value>(), Torch::m_TorchConstantStr(elem)))
    return failure();
  return success();
}

static LogicalResult checkTorchConstantBool(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  bool elem;
  if (!matchPattern(values[0].cast<Value>(), Torch::m_TorchConstantBool(&elem)))
    return failure();
  return success();
}

static LogicalResult checkTorchConstantIntList(
    PatternRewriter& rewriter,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  SmallVector<int64_t> elems;
  if (!matchPattern(
          values[0].cast<Value>(), Torch::m_TorchConstantIntList(elems)))
    return failure();
  return success();
}

static void createTorchCustomCall(
    PatternRewriter& rewriter,
    PDLResultList& results,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 3);

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = disc_ral::getThreadLocalValueRangeStorage(tag);
  vs.clear();
  auto inputs = values[1].cast<ValueRange>();
  auto outputs = values[2].cast<ValueRange>();

  SmallVector<Type> outputTypes;
  for (Value v : outputs)
    outputTypes.push_back(v.getType());
  assert(outputTypes.size() > 0);
  auto ctx = (*outputs.begin()).getContext();
  auto nameAttr = StringAttr::get(ctx, "torch_blade.custom_call");
  Operation* op = rewriter.create<Torch::OperatorOp>(
      (*outputs.begin()).getLoc(), outputTypes, nameAttr, inputs);

  for (Value out : op->getResults())
    vs.push_back(out);

  results.push_back(op);
  results.push_back(ValueRange(vs));
}

static void convertTorchConstantIntListToI64DenseElemsAttr(
    PatternRewriter& rewriter,
    PDLResultList& results,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);

  SmallVector<int64_t> elems;
  auto status = matchPattern(
      values[0].cast<Value>(), Torch::m_TorchConstantIntList(elems));
  assert(status);

  results.push_back(rewriter.getI64TensorAttr(elems));
}

static void convertTorchConstantIntToI64Attr(
    PatternRewriter& rewriter,
    PDLResultList& results,
    ArrayRef<PDLValue> values) {
  assert(values.size() == 1);

  int64_t elem;
  auto status =
      matchPattern(values[0].cast<Value>(), Torch::m_TorchConstantInt(&elem));
  assert(status);

  results.push_back(rewriter.getIntegerAttr(rewriter.getIntegerType(64), elem));
}

// Register some pre-defined helper functions for torch pdl patterns.
void registerPredefinedHelperFunctions(PDLPatternModule& pdlPatterns) {
  pdlPatterns.registerRewriteFunction(
      "CreateTorchCustomCall", createTorchCustomCall);
  pdlPatterns.registerRewriteFunction(
      "ConvertTorchConstantIntListToI64DenseElemsAttr",
      convertTorchConstantIntListToI64DenseElemsAttr);
  pdlPatterns.registerRewriteFunction(
      "ConvertTorchConstantIntToI64Attr", convertTorchConstantIntToI64Attr);

  pdlPatterns.registerConstraintFunction("CheckTorchNone", checkTorchNone);
  pdlPatterns.registerConstraintFunction(
      "CheckNotTorchNone", checkNotTorchNone);
  pdlPatterns.registerConstraintFunction(
      "CheckTorchConstantInt", checkTorchConstantInt);
  pdlPatterns.registerConstraintFunction(
      "CheckTorchConstantFloat", checkTorchConstantFloat);
  pdlPatterns.registerConstraintFunction(
      "CheckTorchConstantStr", checkTorchConstantStr);
  pdlPatterns.registerConstraintFunction(
      "CheckTorchConstantBool", checkTorchConstantBool);
  pdlPatterns.registerConstraintFunction(
      "CheckTorchConstantIntList", checkTorchConstantIntList);
}

} // namespace torch
} // end namespace mlir
