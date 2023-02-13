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

// Currently, this file implements some checking logic to emit CUDA code for
// lmhlo ops and attach them as attributes.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/disc/utils/source_emitter.h"

using namespace mlir;

static void loadDependentDialects(mlir::MLIRContext& context) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect>();

  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames()) {
    context.getOrLoadDialect(name);
  }
}

static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(StringRef inputFilename,
                                                        MLIRContext* context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return mlir::OwningOpRef<mlir::ModuleOp>(
      parseSourceFile<mlir::ModuleOp>(sourceMgr, context));
}

static LogicalResult emitCUDASourceAndAttach(func::FuncOp func) {
  OpBuilder builder(func);

  mlir::disc_ral::SourceEmitterCUDA source_emitter;
  mlir::disc_ral::SourceEmitterCUDA::ValueNameBinding binding;
  for (auto argument : llvm::enumerate(func.getArguments())) {
    source_emitter.bindValueNames(
        SmallVector<Value>({argument.value()}),
        SmallVector<std::string>({"arg" + std::to_string(argument.index())}),
        binding);
  }

  for (auto& op : func.getRegion().getOps()) {
    if (!isa<func::ReturnOp>(&op)) {
      if (!source_emitter.isSupportedOp(&op)) {
        return failure();
      }
      auto instruction = source_emitter.EmitOp(&op, binding);
      if (!instruction.has_value()) {
        return failure();
      } else {
        op.setAttr("cuda-code", builder.getStringAttr(instruction.value()));
      }
    }
  }

  return success();
}

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> payloadInputFilename(
      "payload-input", llvm::cl::desc("<input payload file>"),
      llvm::cl::value_desc("filename"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "DISC CUDA emitter.");

  MLIRContext mlirContext;
  loadDependentDialects(mlirContext);
  RewritePatternSet patternList(&mlirContext);

  OwningOpRef<ModuleOp> payloadModule =
      parseMLIRInput(payloadInputFilename, &mlirContext);
  if (!payloadModule) {
    llvm::errs() << "failed to load payload from file: " << payloadInputFilename
                 << "\n";
    return 1;
  }
  llvm::errs() << "/////// Parsed Payload module: \n"
               << payloadModule.get() << "\n\n";

  SmallVector<func::FuncOp> funcs;
  payloadModule.get().walk([&](func::FuncOp func) { funcs.push_back(func); });
  for (auto func : funcs) {
    if (failed(emitCUDASourceAndAttach(func))) {
      llvm::errs() << "failed to emit CUDA code\n";
      return 1;
    }
  }

  llvm::outs() << "/////// Rewrited Payload module: \n"
               << payloadModule.get() << "\n\n";

  return 0;
}
