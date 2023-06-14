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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "torch-mlir/InitAll.h"

#include <string>

using namespace mlir;

static void loadDependentDialects(mlir::MLIRContext& context) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  mlir::disc_ral::getPDLDependentDialects(registry);

  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);
}

static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(
    StringRef inputFilename,
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

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> pdllInputFilename(
      "pdl-input",
      llvm::cl::desc("<input pdll file>"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> payloadInputFilename(
      "payload-input",
      llvm::cl::desc("<input payload file>"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "output",
      llvm::cl::desc("Output filename"),
      llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I",
      llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"),
      llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "DISC PDLL Frontend");

  MLIRContext mlirContext;
  loadDependentDialects(mlirContext);
  RewritePatternSet patternList(&mlirContext);
  std::vector<std::string> pdllFiles{pdllInputFilename};
  if (failed(mlir::disc_ral::populateDiscPdlPatternsFromFiles(
          &patternList,
          pdllFiles,
          includeDirs,
          torch::kDefaultHelperFunctionDeclarations,
          torch::registerPredefinedHelperFunctions))) {
    llvm::errs() << "failed to compile the pdll file: " << pdllInputFilename
                 << "\n";
    return 1;
  }

  OwningOpRef<ModuleOp> payloadModule =
      parseMLIRInput(payloadInputFilename, &mlirContext);
  if (!payloadModule) {
    llvm::errs() << "failed to load payload from file: " << payloadInputFilename
                 << "\n";
    return 1;
  }
  llvm::errs() << "/////// Parsed Payload module: \n"
               << payloadModule.get() << "\n\n";

  if (failed(applyPatternsAndFoldGreedily(
          payloadModule.get().getBodyRegion(), std::move(patternList)))) {
    llvm::errs() << "failed to apply patterns\n";
    return 1;
  }

  llvm::outs() << "/////// Rewrited Payload module: \n"
               << payloadModule.get() << "\n\n";

  return 0;
}
