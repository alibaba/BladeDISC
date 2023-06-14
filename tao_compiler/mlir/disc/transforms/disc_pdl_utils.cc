/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/disc/transforms/disc_pdl_utils.h"

#include <cstring>
#include <unordered_map>

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/tsl/platform/default/logging.h"

#define DEBUG_TYPE "disc-pdl-utils"

namespace mlir {

using namespace mlir::pdll;

namespace disc_ral {

SmallVector<Value>& getThreadLocalValueRangeStorage(StringRef tag) {
  thread_local static auto valueRangeMap =
      new std::unordered_map<std::string, SmallVector<Value>>{};
  return (*valueRangeMap)[tag.str()];
}

std::vector<int64_t> ConvertArrayAttrToInt(mlir::ArrayAttr array_attr) {
  SmallVector<float, 4> values;
  values.reserve(array_attr.getValue().size());
  for (Attribute val : array_attr.getValue()) {
    values.push_back(static_cast<int64_t>(val.cast<IntegerAttr>().getInt()));
  }
  return {values.begin(), values.end()};
}

std::vector<int64_t> ConvertDenseIntAttr(mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

namespace {

static const std::string kDefaultHelperFunctionDeclarations = R"pdll(
  Rewrite PackValue_0(attr: Attr) -> ValueRange;
  Rewrite PackValue_1(attr: Attr, v0: Value) -> ValueRange;
  Rewrite PackValue_2(attr: Attr, v0: Value, v1: Value) -> ValueRange;
  Rewrite PackValue_3(attr: Attr, v0: Value, v1: Value, v2: Value) -> ValueRange;
  Rewrite PackValue_4(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value) -> ValueRange;
  Rewrite PackValue_5(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value) -> ValueRange;
  Rewrite PackValue_6(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value) -> ValueRange;
  Rewrite PackValue_7(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value) -> ValueRange;
  Rewrite PackValue_8(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value) -> ValueRange;
  Rewrite PackValue_9(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value) -> ValueRange;
  Rewrite PackValue_10(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value) -> ValueRange;
  Rewrite PackValue_11(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value, v10: Value) -> ValueRange;
  Rewrite PackValue_12(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value, v10: Value, v11: Value) -> ValueRange;
  Rewrite PackValue_13(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value, v10: Value, v11: Value, v12: Value) -> ValueRange;
  Rewrite PackValue_14(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value, v10: Value, v11: Value, v12: Value, v13: Value) -> ValueRange;
  Rewrite PackValue_15(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value, v10: Value, v11: Value, v12: Value, v13: Value, v14: Value) -> ValueRange;
  Rewrite PackValue_16(attr: Attr, v0: Value, v1: Value, v2: Value, v3: Value, v4: Value, v5: Value, v6: Value, v7: Value, v8: Value, v9: Value, v10: Value, v11: Value, v12: Value, v13: Value, v14: Value, v15: Value) -> ValueRange;

  Rewrite UnpackValue_1(v : ValueRange) -> (Value);
  Rewrite UnpackValue_2(v : ValueRange) -> (Value, Value);
  Rewrite UnpackValue_3(v : ValueRange) -> (Value, Value, Value);
  Rewrite UnpackValue_4(v : ValueRange) -> (Value, Value, Value, Value);
  Rewrite UnpackValue_5(v : ValueRange) -> (Value, Value, Value, Value, Value);
  Rewrite UnpackValue_6(v : ValueRange) -> (Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_7(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_8(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_9(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_10(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_11(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_12(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_13(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_14(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_15(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);
  Rewrite UnpackValue_16(v : ValueRange) -> (Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value, Value);

  Rewrite CreateCustomCall(tag : Attr, inputs : ValueRange, outputs : ValueRange) -> (op: Op, new_outputs : ValueRange);
  Rewrite SetAttr(op : Op, key : Attr, value : Attr);
  Rewrite SetCustomAttr(op : Op, key : Attr, value : Attr);
  Rewrite GetAttrOrDefault(op : Op, key : Attr, value : Attr) -> (Attr);

  Constraint CheckConstantTensor(v : Value);
  Constraint CheckConstantTensorValueIs(v : Value, expected : Attr);
  Rewrite IsConstantTensor(v : Value) -> Attr;
  Rewrite CreateSparseSegmentReduction(tag : Attr, inputs : ValueRange, outputs : ValueRange, reduction_mode : Attr) -> (op: Op, new_outputs : ValueRange);
  Rewrite CloneOpWithNewOperand(op : Op, new_operand : Value, old_operand : Value) -> Op;
  Constraint CheckSliceOpAttribute(slice_attr : Attr, limit : Attr, start : Attr, strides : Attr);
)pdll";

// Combines the `chunkBuffer` with some pre-defined helper function prototypes.
// The result is written to a newly allocated buffer which will be returned.
std::unique_ptr<llvm::MemoryBuffer> addPredefinedPrototypes(
    StringRef data, const std::string& customPredefinedFunctionPrototypes) {
  std::string prefix =
      kDefaultHelperFunctionDeclarations + customPredefinedFunctionPrototypes;
  size_t bytes = prefix.size() + data.size() + 1;
  auto combinedBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(bytes);
  char* dst = combinedBuffer->getBufferStart();
  const char* src0 = prefix.data();
  size_t src0Size = prefix.size();
  std::memcpy(dst, src0, src0Size);
  std::memcpy(dst + src0Size, data.data(), data.size());
  dst[bytes - 1] = '\0';
  return combinedBuffer;
}

// Compiles a pdll module into a pdl pattern module.
OwningOpRef<ModuleOp> compilePDLL(
    MLIRContext& mlirContext, StringRef chunkBuffer,
    const std::vector<std::string>& includeDirs,
    const std::string& customPredefinedFunctionPrototypes) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(
      addPredefinedPrototypes(chunkBuffer, customPredefinedFunctionPrototypes),
      SMLoc());

  ods::Context odsContext;
  ast::Context astContext(odsContext);
  FailureOr<ast::Module*> module = parsePDLLAST(astContext, sourceMgr, false);
  if (failed(module)) return nullptr;

  auto pdlModule =
      codegenPDLLToMLIR(&mlirContext, astContext, sourceMgr, **module);
  if (pdlModule) {
    SmallVector<pdl::RewriteOp> ops;
    pdlModule->walk([&](pdl::RewriteOp op) { ops.push_back(op); });
    for (pdl::RewriteOp op : ops) {
      OpBuilder b(op);
      auto newOp =
          b.create<pdl::RewriteOp>(op.getLoc(), nullptr, nullptr, ValueRange{});
      newOp.getBodyRegion().getBlocks().splice(
          newOp.getBodyRegion().getBlocks().begin(),
          op.getBodyRegion().getBlocks());
      op->erase();
    }
  }
  if (VLOG_IS_ON(1)) {
    llvm::errs() << "/////// Parsed PDL module: \n"
                 << pdlModule.get() << "\n\n";
  }
  return pdlModule;
}

template <int ExpectedNum>
static LogicalResult packValues(PatternRewriter& rewriter,
                                PDLResultList& results,
                                ArrayRef<PDLValue> values) {
  int numValueInputs = static_cast<int>(values.size()) - 1;
  if (numValueInputs != ExpectedNum) {
    llvm::errs() << "PackValue expects " << ExpectedNum << " values but got "
                 << numValueInputs << "\n";
    return failure();
  }

  if (values.size() <= 1) {
    results.push_back(ValueRange{});
    return success();
  }

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = getThreadLocalValueRangeStorage(tag);
  vs.clear();
  for (auto& v : values.drop_front()) {
    vs.push_back(v.cast<Value>());
  }
  results.push_back(ValueRange{vs});
  return success();
}

template <int ExpectedNum>
static LogicalResult unpackValues(PatternRewriter& rewriter,
                                  PDLResultList& results,
                                  ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  int numResults = 0;
  for (Value v : values[0].cast<ValueRange>()) {
    results.push_back(v);
    ++numResults;
  }

  if (numResults != ExpectedNum) {
    llvm::errs() << "PackValue expects " << ExpectedNum << " values but got "
                 << numResults << "\n";
    return failure();
  }
  return success();
}

static LogicalResult createCustomCall(PatternRewriter& rewriter,
                                      PDLResultList& results,
                                      ArrayRef<PDLValue> values) {
  assert(values.size() == 3);

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = getThreadLocalValueRangeStorage(tag);
  vs.clear();
  auto inputs = values[1].cast<ValueRange>();
  auto outputs = values[2].cast<ValueRange>();

  SmallVector<Type> outputTypes;
  for (Value v : outputs) outputTypes.push_back(v.getType());
  assert(outputTypes.size() > 0);
  auto ctx = (*outputs.begin()).getContext();
  auto emptyStrAttr = StringAttr::get(ctx, "");
  Operation* op = rewriter.create<mhlo_disc::CustomCallV2Op>(
      (*outputs.begin()).getLoc(), outputTypes, inputs, "",
      DictionaryAttr::get(ctx, {}), false, emptyStrAttr, emptyStrAttr,
      emptyStrAttr, emptyStrAttr, emptyStrAttr, emptyStrAttr, emptyStrAttr);

  for (Value out : op->getResults()) vs.push_back(out);

  results.push_back(op);
  results.push_back(ValueRange(vs));
  return success();
}

static LogicalResult createSparseSegmentReduction(PatternRewriter& rewriter,
                                                  PDLResultList& results,
                                                  ArrayRef<PDLValue> values) {
  assert(values.size() == 4);

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = getThreadLocalValueRangeStorage(tag);
  vs.clear();
  auto inputs = values[1].cast<ValueRange>();
  auto outputs = values[2].cast<ValueRange>();
  auto reduction_attr =
      values[3].cast<Attribute>().cast<mhlo_disc::ReductionModeEnumAttr>();

  SmallVector<Type> outputTypes;
  for (Value v : outputs) outputTypes.push_back(v.getType());
  assert(outputTypes.size() == 2);
  assert(inputs.size() == 4);
  auto ctx = (*outputs.begin()).getContext();
  Operation* op =
      rewriter.create<mhlo_disc::SparseSegmentReductionWithEmptyRowsOp>(
          (*outputs.begin()).getLoc(), outputTypes[0], outputTypes[1],
          inputs[0], inputs[1], inputs[2], inputs[3], reduction_attr);

  for (Value out : op->getResults()) vs.push_back(out);

  results.push_back(op);
  results.push_back(ValueRange(vs));
  return success();
}

static LogicalResult cloneOpWithNewOperand(PatternRewriter& rewriter,
                                           PDLResultList& results,
                                           ArrayRef<PDLValue> values) {
  assert(values.size() == 3);

  auto origin_op = values[0].cast<Operation*>();
  auto new_operand = values[1].cast<Value>();
  auto old_operand = values[2].cast<Value>();

  IRMapping mapping;
  mapping.map(old_operand, new_operand);
  rewriter.setInsertionPoint(origin_op);
  Operation* op = rewriter.clone(*origin_op, mapping);
  results.push_back(op);
  return success();
}

static LogicalResult checkConstantTensor(PatternRewriter& rewriter,
                                         ArrayRef<PDLValue> values) {
  assert(values.size() == 1);
  auto v = values[0].cast<Value>();
  DenseElementsAttr denseAttr;
  return matchPattern(v, m_Constant(&denseAttr)) ? success() : failure();
}

static LogicalResult checkConstantTensorValueIs(PatternRewriter& rewriter,
                                                ArrayRef<PDLValue> values) {
  assert(values.size() == 2);
  auto v = values[0].cast<Value>();
  APInt val;
  bool is_constant = matchPattern(v, m_ConstantInt(&val));
  if (is_constant) {
    int64_t constant_val = static_cast<int64_t>(val.getSExtValue());
    int64_t expected_val = static_cast<int64_t>(
        values[1].cast<Attribute>().cast<IntegerAttr>().getInt());
    if (constant_val == expected_val) {
      return success();
    }
  }
  return failure();
}

static LogicalResult checkSliceOpAttribute(PatternRewriter& rewriter,
                                           ArrayRef<PDLValue> values) {
  assert(values.size() == 4);
  auto slice_attr = values[0].cast<Attribute>().cast<DictionaryAttr>();
  auto slice_limit = ConvertDenseIntAttr(
      slice_attr.getAs<DenseIntElementsAttr>("limit_indices"));
  auto slice_start = ConvertDenseIntAttr(
      slice_attr.getAs<DenseIntElementsAttr>("start_indices"));
  auto slice_strides =
      ConvertDenseIntAttr(slice_attr.getAs<DenseIntElementsAttr>("strides"));
  auto expected_limit =
      ConvertArrayAttrToInt(values[1].cast<Attribute>().cast<ArrayAttr>());
  auto expected_start =
      ConvertArrayAttrToInt(values[2].cast<Attribute>().cast<ArrayAttr>());
  auto expected_strides =
      ConvertArrayAttrToInt(values[3].cast<Attribute>().cast<ArrayAttr>());
  auto check_all_equal = [&](std::vector<int64_t> origin,
                             std::vector<int64_t> expected) {
    if (origin.size() != expected.size()) {
      VLOG(2) << origin.size() << " " << expected.size();
      return false;
    }
    for (int i = 0; i < origin.size(); ++i) {
      if (origin[i] != expected[i]) {
        VLOG(2) << origin[i] << " " << expected[i];
        return false;
      }
    }
    return true;
  };
  if (check_all_equal(slice_limit, expected_limit) &&
      check_all_equal(slice_start, expected_start) &&
      check_all_equal(slice_strides, expected_strides)) {
    return success();
  }
  return failure();
}

static LogicalResult isConstantTensor(PatternRewriter& rewriter,
                                      PDLResultList& results,
                                      ArrayRef<PDLValue> values) {
  assert(values.size() == 1);

  auto v = values[0].cast<Value>();
  DenseElementsAttr denseAttr;
  results.push_back(matchPattern(v, m_Constant(&denseAttr))
                        ? BoolAttr::get(v.getContext(), true)
                        : BoolAttr::get(v.getContext(), false));
  return success();
}

void registerPredefinedHelperFunctions(PDLPatternModule& pdlPatterns,
                                       RegisterPDLFunctionsCallback callback) {
  pdlPatterns.registerRewriteFunction(
      "SetAttr", [](PatternRewriter& rewriter, Operation* op, Attribute keyAttr,
                    Attribute valueAttr) {
        StringRef key = keyAttr.cast<StringAttr>().getValue();
        op->setAttr(key, valueAttr);
      });
  pdlPatterns.registerRewriteFunction(
      "SetCustomAttr", [](PatternRewriter& rewriter, Operation* op,
                          Attribute keyAttr, Attribute valueAttr) {
        auto customAttrs = op->getAttrOfType<DictionaryAttr>("custom_attrs");
        if (!customAttrs) {
          customAttrs = DictionaryAttr::get(op->getContext(), {});
        }
        StringRef key = keyAttr.cast<StringAttr>().getValue();
        SmallVector<NamedAttribute> newAttrs;
        for (auto& attr : customAttrs) {
          if (attr.getName().getValue() == key) continue;
          newAttrs.push_back(attr);
        }
        newAttrs.push_back(
            NamedAttribute(keyAttr.cast<StringAttr>(), valueAttr));
        auto newCustomAttrs = DictionaryAttr::get(op->getContext(), newAttrs);
        op->setAttr("custom_attrs", newCustomAttrs);
      });
  pdlPatterns.registerRewriteFunction(
      "GetAttrOrDefault", [](PatternRewriter& rewriter, Operation* op,
                             Attribute keyAttr, Attribute valueAttr) {
        StringRef key = keyAttr.cast<StringAttr>().getValue();
        return op->hasAttr(key) ? op->getAttr(key) : valueAttr;
      });
  pdlPatterns.registerRewriteFunction("CreateCustomCall", createCustomCall);
  pdlPatterns.registerRewriteFunction("PackValue_0", packValues<0>);
  pdlPatterns.registerRewriteFunction("IsConstantTensor", isConstantTensor);
  pdlPatterns.registerRewriteFunction("CreateSparseSegmentReduction",
                                      createSparseSegmentReduction);
  pdlPatterns.registerRewriteFunction("CloneOpWithNewOperand",
                                      cloneOpWithNewOperand);

#define REGISTER_PACK_AND_UNPACK(N)                                    \
  pdlPatterns.registerRewriteFunction("PackValue_" #N, packValues<N>); \
  pdlPatterns.registerRewriteFunction("UnpackValue_" #N, unpackValues<N>);

  REGISTER_PACK_AND_UNPACK(1);
  REGISTER_PACK_AND_UNPACK(2);
  REGISTER_PACK_AND_UNPACK(3);
  REGISTER_PACK_AND_UNPACK(4);
  REGISTER_PACK_AND_UNPACK(5);
  REGISTER_PACK_AND_UNPACK(6);
  REGISTER_PACK_AND_UNPACK(7);
  REGISTER_PACK_AND_UNPACK(8);
  REGISTER_PACK_AND_UNPACK(9);
  REGISTER_PACK_AND_UNPACK(10);
  REGISTER_PACK_AND_UNPACK(11);
  REGISTER_PACK_AND_UNPACK(12);
  REGISTER_PACK_AND_UNPACK(13);
  REGISTER_PACK_AND_UNPACK(14);
  REGISTER_PACK_AND_UNPACK(15);
  REGISTER_PACK_AND_UNPACK(16);

#undef REGISTER_PACK_AND_UNPACK

  pdlPatterns.registerConstraintFunction("CheckConstantTensor",
                                         checkConstantTensor);
  pdlPatterns.registerConstraintFunction("CheckConstantTensorValueIs",
                                         checkConstantTensorValueIs);
  pdlPatterns.registerConstraintFunction("CheckSliceOpAttribute",
                                         checkSliceOpAttribute);

  if (callback) callback(pdlPatterns);
}

}  // namespace

std::vector<std::string> ParseFileString(const std::string& str) {
  std::vector<std::string> parsedStrs;
  if (str.empty()) return parsedStrs;
  SmallVector<StringRef> items;
  StringRef(str.data(), str.size())
      .split(items, ',', /*MaxSplit=*/-1,
             /*KeepEmpty=*/false);
  for (const auto& item : items) {
    parsedStrs.push_back(item.str());
  }
  return parsedStrs;
}

// Adds related depedent dialects (e.g. PDL dialect).
void getPDLDependentDialects(DialectRegistry& registry) {
  registry.insert<mhlo_disc::MhloDiscDialect, pdl::PDLDialect>();
}

// Parses pdll patterns from string, compile them and then add to `patterns`.
LogicalResult populateDiscPdlPatternsFromString(
    RewritePatternSet* patterns, StringRef pdllModule,
    const std::vector<std::string>& includeDirs,
    const std::string& customPredefinedFunctionPrototypes,
    RegisterPDLFunctionsCallback callback) {
  if (pdllModule.empty()) return success();
  auto pdlModule = compilePDLL(*patterns->getContext(), pdllModule, includeDirs,
                               customPredefinedFunctionPrototypes);
  if (!pdlModule) {
    llvm::errs() << "failed to compile pdll patern from string:\n"
                 << pdllModule << "\n\n";
    return failure();
  }

  PDLPatternModule pdlPatterns(std::move(pdlModule));
  registerPredefinedHelperFunctions(pdlPatterns, callback);
  patterns->add(std::move(pdlPatterns));
  return success();
}

// Adds customized pdl patterns
LogicalResult populateDiscPdlPatternsFromFiles(
    RewritePatternSet* patterns, const std::vector<std::string>& pdlFiles,
    const std::vector<std::string>& includeDirs,
    const std::string& customPredefinedFunctionPrototypes,
    RegisterPDLFunctionsCallback callback) {
  std::string errorMessage;
  for (auto& file : pdlFiles) {
    std::unique_ptr<llvm::MemoryBuffer> pdllFile =
        openInputFile(file, &errorMessage);
    if (!pdllFile) {
      llvm::errs() << "fail to open pdll file@" << file << ": " << errorMessage
                   << "\n";
      return failure();
    }

    auto pdlModule =
        compilePDLL(*patterns->getContext(), pdllFile->getBuffer(), includeDirs,
                    customPredefinedFunctionPrototypes);
    if (!pdlModule) {
      llvm::errs() << "failed to compile pdll patern from file@" << file
                   << ":\n"
                   << pdllFile->getBuffer() << "\n\n";
      return failure();
    }

    PDLPatternModule pdlPatterns(std::move(pdlModule));
    registerPredefinedHelperFunctions(pdlPatterns, callback);
    patterns->add(std::move(pdlPatterns));
  }
  return success();
}

}  // namespace disc_ral
}  // namespace mlir
