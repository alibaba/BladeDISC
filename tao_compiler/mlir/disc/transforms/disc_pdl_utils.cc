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

#include "tensorflow/compiler/mlir/disc/transforms/disc_pdl_utils.h"

#include <cstring>
#include <unordered_map>

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"

namespace mlir {

using namespace mlir::pdll;

namespace disc_ral {

SmallVector<Value>& getThreadLocalValueRangeStorage(StringRef tag) {
  thread_local static auto valueRangeMap =
      new std::unordered_map<std::string, SmallVector<Value>>{};
  return (*valueRangeMap)[tag.str()];
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
  llvm::errs() << "/////// Parsed PDL module: \n" << pdlModule.get() << "\n\n";
  return pdlModule;
}

template <int ExpectedNum>
static void packValues(PatternRewriter& rewriter, PDLResultList& results,
                       ArrayRef<PDLValue> values) {
  int numValueInputs = static_cast<int>(values.size()) - 1;
  if (numValueInputs != ExpectedNum) {
    llvm::errs() << "PackValue expects " << ExpectedNum << " values but got "
                 << numValueInputs << "\n";
    return;
  }

  if (values.size() <= 1) {
    results.push_back(ValueRange{});
    return;
  }

  auto tag = values[0].cast<Attribute>().cast<StringAttr>().getValue();
  auto& vs = getThreadLocalValueRangeStorage(tag);
  vs.clear();
  for (auto& v : values.drop_front()) {
    vs.push_back(v.cast<Value>());
  }
  results.push_back(ValueRange{vs});
}

template <int ExpectedNum>
static void unpackValues(PatternRewriter& rewriter, PDLResultList& results,
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
    return;
  }
}

static void createCustomCall(PatternRewriter& rewriter, PDLResultList& results,
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
  pdlPatterns.registerRewriteFunction("CreateCustomCall", createCustomCall);
  pdlPatterns.registerRewriteFunction("PackValue_0", packValues<0>);

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

  if (callback) callback(pdlPatterns);
}

}  // namespace

// Adds related depedent dialects (e.g. PDL dialect).
void getDependentDialects(DialectRegistry& registry) {
  registry.insert<mhlo_disc::MhloDiscDialect, pdl::PDLDialect>();
}

// Parses pdll patterns from string, compile them and then add to `patterns`.
LogicalResult populateDiscPdlPatternsFromString(
    RewritePatternSet* patterns, StringRef pdllModule,
    const std::vector<std::string>& includeDirs,
    const std::string& customPredefinedFunctionPrototypes,
    RegisterPDLFunctionsCallback callback) {
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
