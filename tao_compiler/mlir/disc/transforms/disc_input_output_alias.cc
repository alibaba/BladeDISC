// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <stdexcept>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace disc_ral {

bool ParseInputOutputAliasInfo(func::FuncOp main, std::vector<int>& params,
                               std::vector<int>& outputs) {
  auto dict_attr = main->getAttrOfType<DictionaryAttr>("tf.entry_function");
  if (!dict_attr) {
    return false;
  }

  const std::string inputOutputAliasParamsKey = "input_output_alias_params";
  const std::string inputOutputAliasOutputsKey = "input_output_alias_outputs";

  if (!dict_attr.get(inputOutputAliasParamsKey) ||
      !dict_attr.get(inputOutputAliasParamsKey)) {
    return false;
  }

  auto param_str =
      dict_attr.get(inputOutputAliasParamsKey).dyn_cast<mlir::StringAttr>();
  auto outputs_str =
      dict_attr.get(inputOutputAliasOutputsKey).dyn_cast<mlir::StringAttr>();

  SmallVector<StringRef, 4> parsed_params, parsed_outputs;
  param_str.getValue().split(parsed_params, ',', /*MaxSplit=*/-1,
                             /*KeepEmpty=*/false);
  outputs_str.getValue().split(parsed_outputs, ',', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);

  for (StringRef str : parsed_params) {
    try {
      params.push_back(std::stoi(str.str()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("An invalid value " + str.str() +
                                  " is received when converting index in "
                                  "input_output_alias_params to int value");
    }
  }

  for (StringRef str : parsed_outputs) {
    try {
      outputs.push_back(std::stoi(str.str()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("An invalid value " + str.str() +
                                  " is received when converting index in "
                                  "input_output_alias_outputs to int value");
    }
  }

  return true;
}

struct DiscInputOutputAliasPass
    : public DiscInputOutputAliasPassBase<DiscInputOutputAliasPass> {
  using DiscInputOutputAliasPassBase<
      DiscInputOutputAliasPass>::DiscInputOutputAliasPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo_disc::MhloDiscDialect>();
  }

 public:
  DiscInputOutputAliasPass() {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) {
      signalPassFailure();
      return;
    }

    // Parse attribute info
    std::vector<int> params_index, outputs_index;
    try {
      if (!ParseInputOutputAliasInfo(main_func, params_index, outputs_index)) {
        return;
      }
    } catch (const std::invalid_argument& e) {
      main_func.emitOpError() << e.what();
    }

    if (params_index.size() != outputs_index.size()) {
      main_func.emitOpError()
          << "input_output_alias_params and input_output_alias_outputs should "
             "have same number of index";
      signalPassFailure();
    }

    OpBuilder builder(main_func.getBody());
    auto returnOp =
        cast<mhlo::ReturnOp>(main_func.getBody().back().getTerminator());

    // Get input and output tensor for main function
    auto params = main_func.getArguments();
    auto outputs = returnOp.getOperands();

    // Insert mhlo_disc::ArgsMutationOp
    for (int i = 0; i < params_index.size(); i++) {
      if (outputs[outputs_index[i]] == params[params_index[i]]) {
        continue;
      }
      // DISC now only support one-hop buffer sharing.
      auto defineOp = outputs[outputs_index[i]].getDefiningOp();
      if (llvm::isa<mhlo::OptimizationBarrierOp>(defineOp)) {
        continue;
      }

      builder.setInsertionPointAfter(defineOp);
      builder.create<mhlo_disc::ArgsMutationOp>(main_func.getLoc(),
                                                outputs[outputs_index[i]],
                                                params[params_index[i]]);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDiscInputOutputAliasPass() {
  return std::make_unique<DiscInputOutputAliasPass>();
}

}  // namespace disc_ral
}  // namespace mlir
