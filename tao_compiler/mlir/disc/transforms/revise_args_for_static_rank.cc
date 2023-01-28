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

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"  // TF:llvm-project
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"              // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {

using disc_ral::kDhloInputShapeAttr;
using disc_ral::kDhloInputValueAttr;
using placement_utils::kConst;
using placement_utils::kCpu;
using placement_utils::kGpu;
using placement_utils::kInputPlacementAttr;

namespace disc_ral {
namespace {

// Replace const arguments to ConstOp and update argument type if it is a
// fixed-shaped input
struct ReviseArgsForStaticRankPass
    : public ReviseArgumentsForStaticRankPassBase<ReviseArgsForStaticRankPass> {
  explicit ReviseArgsForStaticRankPass()
      : ReviseArgumentsForStaticRankPassBase<ReviseArgsForStaticRankPass>::
            ReviseArgumentsForStaticRankPassBase() {}

  void runOnOperation() override;
  void replaceArgWithConstOp(func::FuncOp main, DictionaryAttr dict_attr,
                             unsigned idx);
};

void ReviseArgsForStaticRankPass::replaceArgWithConstOp(
    func::FuncOp main, DictionaryAttr dict_attr, unsigned idx) {
  ModuleOp module = getOperation();
  auto attr = dict_attr.get(
      (disc_ral::kDhloInputValueAttr + ("_" + llvm::Twine(idx))).str());
  if (!attr) {
    module.emitError() << "kMhloInputValueAttr not found for idx: " << idx
                       << ".\n";
    return signalPassFailure();
  }
  OpBuilder b(&main.front().front());
  Value const_input = b.create<TF::ConstOp>(main.getLoc(), attr);
  main.getArgument(idx).replaceAllUsesWith(const_input);
  // Here we will not erase the unused argument, since the arg indices in ctx
  // has to match the arg indices of the Mlir Module when ral recv inputs
}

void ReviseArgsForStaticRankPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp main_func = module.lookupSymbol<func::FuncOp>("main");
  if (!main_func) {
    module.emitError("Error: main_func not found.\n");
    return signalPassFailure();
  }
  auto num_inputs = main_func.getNumArguments();
  auto dict_attr =
      main_func->getAttrOfType<DictionaryAttr>("tf.entry_function");
  if (!dict_attr) {
    return;
  }
  auto input_placements_attr =
      dict_attr.get(placement_utils::kInputPlacementAttr);
  if (!input_placements_attr) {
    return;
  }
  SmallVector<StringRef, 4> input_placements;
  input_placements_attr.cast<mlir::StringAttr>().getValue().split(
      input_placements, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  assert(input_placements.size() == num_inputs &&
         "input_placements.size() is not equal to num of inputs");
  SmallVector<StringRef, 4> new_input_placements;

  // Step 1, for each const input, create a ConstOp and replace the Arg
  for (int i = 0; i < num_inputs; ++i) {
    auto placement = input_placements[i];
    if (placement == kConst) {
      // It makes no difference whether cpu or gpu for const args here,
      // since the arg will have no consumer. However, let's just leave
      // it as gpu
      new_input_placements.push_back(placement_utils::kGpu);
      replaceArgWithConstOp(main_func, dict_attr, i);
    } else {
      new_input_placements.push_back(placement);
    }
  }

  // Step 2, for each fixed-shaped input, update the type of the Arg
  // A shape inference pass will run in seperate to propagate the shape
  // information to the needed nodes
  auto func_type = main_func.getFunctionType();
  SmallVector<Type, 4> input_types(func_type.getInputs().begin(),
                                   func_type.getInputs().end());
  assert(input_types.size() == num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto attr = dict_attr.get(
        (disc_ral::kDhloInputShapeAttr + ("_" + llvm::Twine(i))).str());
    if (attr) {
      assert(attr.isa<DenseElementsAttr>() && "unexpected kMhloInputShapeAttr");
      auto type = attr.cast<DenseElementsAttr>().getType();
      main_func.getArgument(i).setType(type);
      input_types[i] = type;
    }
  }
  OpBuilder builder(&main_func.front().front());
  auto new_func_type =
      builder.getFunctionType(input_types, func_type.getResults());
  main_func.setType(new_func_type);

  // Step 3, update input_placement attr
  SmallVector<mlir::NamedAttribute, 4> new_attributes;
  for (auto attr : dict_attr) {
    if (attr.getName() != placement_utils::kInputPlacementAttr) {
      new_attributes.push_back(attr);
    }
  }
  new_attributes.push_back(builder.getNamedAttr(
      "input_placements",
      builder.getStringAttr(llvm::join(new_input_placements, ","))));
  main_func->setAttr("tf.entry_function",
                     builder.getDictionaryAttr(new_attributes));

  return;
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createReviseArgsForStaticRankPass() {
  return std::make_unique<ReviseArgsForStaticRankPass>();
}

}  // namespace disc_ral
}  // namespace mlir
