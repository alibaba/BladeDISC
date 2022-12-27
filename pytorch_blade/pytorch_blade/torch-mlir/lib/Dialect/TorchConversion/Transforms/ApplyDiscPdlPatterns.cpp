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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_pdl_utils.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "tests/torch-disc-pdll/utils.h"

using namespace mlir;
using namespace mlir::torch;

namespace {

static bool wouldOpBeTriviallyDeadImplDisc(Operation* rootOp) {
  // The set of operations to consider when checking for side effects.
  SmallVector<Operation*, 1> effectingOps(1, rootOp);
  while (!effectingOps.empty()) {
    Operation* op = effectingOps.pop_back_val();

    // If the operation has recursive effects, push all of the nested operations
    // on to the stack to consider.
    bool hasRecursiveEffects =
        op->hasTrait<::mlir::OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      // Modification 1, if has recursive effects, directly return false.
      return false;
    }

    // If the op has memory effects, try to characterize them to see if the op
    // is trivially dead here.
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Check to see if this op either has no effects, or only allocates/reads
      // memory.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);

      // Gather all results of this op that are allocated.
      SmallPtrSet<Value, 4> allocResults;
      for (const MemoryEffects::EffectInstance& it : effects)
        if (isa<MemoryEffects::Allocate>(it.getEffect()) && it.getValue() &&
            it.getValue().getDefiningOp() == op)
          allocResults.insert(it.getValue());

      if (!llvm::all_of(
              effects,
              [&allocResults](const MemoryEffects::EffectInstance& it) {
                // We can drop effects if the value is an allocation and is a
                // result of the operation.
                if (allocResults.contains(it.getValue()))
                  return true;
                // Otherwise, the effect must be a read.
                return isa<MemoryEffects::Read>(it.getEffect());
              })) {
        return false;
      }
      continue;

      // Otherwise, if the op has recursive side effects we can treat the
      // operation itself as having no effects.
    }
    if (hasRecursiveEffects)
      continue;

    // If there were no effect interfaces, we treat this op as conservatively
    // having effects.
    return true;
  }

  // Modification 2:
  // If we get here, we mark the op as "dead".
  return true;
}

bool wouldOpBeTriviallyDeadDisc(Operation* op) {
  if (op->mightHaveTrait<::mlir::OpTrait::IsTerminator>()) {
    return false;
  }
  return wouldOpBeTriviallyDeadImplDisc(op);
}

bool isOpTriviallyDeadDisc(Operation* op) {
  return op->use_empty() && wouldOpBeTriviallyDeadDisc(op);
}

// add pre-defined pdll patterns here.
std::string getTorchPredefinedPDLPatterns() {
  std::string preDefinedPatterns;
#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
  preDefinedPatterns += R"pdll(
    Rewrite ConvertToF16(value: Value) -> Value {
      let f16_dtype = op<torch.constant.int> {value = attr<"5">} -> (type<"!torch.int">);
      let old_type = GetTorchTensorType(value);
      let new_type = ConvertTorchTensorElemType(old_type, attr<"\"f16\"">);
      let false_val = op<torch.constant.bool> {value = attr<"false">} -> (type<"!torch.bool">);
      let none_val = op<torch.constant.none> -> (type<"!torch.none">);
      let f16_value = op<torch.aten.to.dtype>(
        value, f16_dtype, false_val, false_val, none_val
      ) -> (new_type);
    
      return f16_value.0;
    }
    
    Rewrite ConvertToF32(value: Value) -> Value {
      let f32_dtype = op<torch.constant.int> {value = attr<"6">} -> (type<"!torch.int">);
      let old_type = GetTorchTensorType(value);
      let new_type = ConvertTorchTensorElemType(old_type, attr<"\"f32\"">);
      let false_val = op<torch.constant.bool> {value = attr<"false">} -> (type<"!torch.bool">);
      let none_val = op<torch.constant.none> -> (type<"!torch.none">);
      let f32_value = op<torch.aten.to.dtype>(
        value, f32_dtype, false_val, false_val, none_val
      ) -> (new_type);
    
      return f32_value.0;
    }

    Pattern TorchGroupNormOpF32 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f32\"">);
    
      /// rewrite phase
      rewrite gn with {
        let f16_input = ConvertToF16(input);
        let f16_weight = ConvertToF16(weight);
        let f16_bias = ConvertToF16(bias);
        let f16_output = ConvertToF16(gn.0);
    
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, f16_input, f16_weight, f16_bias);
        let outputs = PackValue_1(attr<"\"out\"">, f16_output);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"false">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        let new_output = ConvertToF32(rs);
    
        replace gn with new_output;
      };
    }

    Pattern TorchGroupNormWithSiluOpF32 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      let silu = op<torch.aten.silu>(gn.0);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f32\"">);
    
      /// rewrite phase
      rewrite silu with {
        let f16_input = ConvertToF16(input);
        let f16_weight = ConvertToF16(weight);
        let f16_bias = ConvertToF16(bias);
        let f16_output = ConvertToF16(silu.0);
    
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, f16_input, f16_weight, f16_bias);
        let outputs = PackValue_1(attr<"\"out\"">, f16_output);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"true">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        let new_output = ConvertToF32(rs);
    
        replace silu with new_output;
      };
    }

    Pattern TorchGroupNormOpF16 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f16\"">);
    
      /// rewrite phase
      rewrite gn with {
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, input, weight, bias);
        let outputs = PackValue_1(attr<"\"out\"">, gn.0);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"false">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        replace gn with rs;
      };
    } 

    Pattern TorchGroupNormWithSiluOpF16 {
      /// match phase: define the pattern
      let eps_attr : Attr;
      let eps = op<torch.constant.float> { value = eps_attr };
      let gn = op<torch.aten.group_norm>(
        input: Value,
        num_group: Value,
        weight: Value,
        bias: Value,
        eps.0,
        cudnn_enabled: Value
      ) -> (old_type: Type);
      let silu = op<torch.aten.silu>(gn.0);
      CheckNotTorchNone(bias);
      CheckTorchConstantInt(num_group);
      CheckTorchTensorElemType(input, attr<"\"f16\"">);
    
      /// rewrite phase
      rewrite silu with {
        /// 1. create custom call op
        let inputs = PackValue_3(attr<"\"in\"">, input, weight, bias);
        let outputs = PackValue_1(attr<"\"out\"">, silu.0);
        let infos = CreateTorchCustomCall(attr<"\"op\"">, inputs, outputs);
    
        /// 2. set attrs that are used by bladedisc.
        SetAttr(infos.op, attr<"\"call_target_name\"">, attr<"\"ral_pdll_group_norm\"">);
        SetAttr(infos.op, attr<"\"input_placements\"">, attr<"\"x,x,x\"">);
        SetAttr(infos.op, attr<"\"output_placements\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"device\"">, attr<"\"x\"">);
        SetAttr(infos.op, attr<"\"input_layouts\"">, attr<"\"NCHW,*,*\"">);
        SetAttr(infos.op, attr<"\"output_layouts\"">, attr<"\"NCHW\"">);
        SetAttr(infos.op, attr<"\"expected_input_layouts\"">, attr<"\"NHWC,*,*\"">);
        SetAttr(infos.op, attr<"\"expected_output_layouts\"">, attr<"\"NHWC\"">);
    
        /// 3. set attrs that are directly passed to the custom call kernel.
        let num_group_attr = ConvertTorchConstantIntToI64Attr(num_group);
        SetCustomAttr(infos.op, attr<"\"num_group\"">, num_group_attr);
        SetCustomAttr(infos.op, attr<"\"eps\"">, eps_attr);
        SetCustomAttr(infos.op, attr<"\"silu\"">, attr<"true">);
    
        let rs = UnpackValue_1(infos.new_outputs);
        replace silu with rs;
      };
    }
    
  )pdll";
#endif

  return preDefinedPatterns;
}

struct ApplyDiscPdlPatternsPass
    : public mlir::torch::TorchConversion::ApplyDiscPdlPatternsBase<
          ApplyDiscPdlPatternsPass> {
  ApplyDiscPdlPatternsPass(
      const std::string& pdll_files,
      const std::string& pdll_include_dirs)
      : ApplyDiscPdlPatternsBase<
            ApplyDiscPdlPatternsPass>::ApplyDiscPdlPatternsBase() {
    this->pdll_files_ = pdll_files;
    this->pdll_include_dirs_ = pdll_include_dirs;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<mhlo_disc::MhloDiscDialect>();
    registry.insert<tensor::TensorDialect>();
    mlir::disc_ral::getPDLDependentDialects(registry);
  }
  void runOnOperation() override;
};

struct PdlDeadCodeElimination : public RewritePattern {
  PdlDeadCodeElimination(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  bool isInplaceSafe(Value& input) const {
    if (input.getType().isa<mlir::torch::Torch::NonValueTensorType>()) {
      return false;
    }
    return true;
  }

  LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter)
      const override {
    for (Value operand : op->getOperands()) {
      // All inputs must not be NonValueTensorType.
      if (!isInplaceSafe(operand)) {
        return failure();
      }
    }
    if (!isOpTriviallyDeadDisc(op)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void ApplyDiscPdlPatternsPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);

  auto pdll_include_dirs = mlir::disc_ral::ParseFileString(pdll_include_dirs_);

  (void)mlir::disc_ral::populateDiscPdlPatternsFromString(
      &patterns,
      getTorchPredefinedPDLPatterns(),
      pdll_include_dirs,
      torch::kDefaultHelperFunctionDeclarations,
      torch::registerPredefinedHelperFunctions);

  (void)mlir::disc_ral::populateDiscPdlPatternsFromFiles(
      &patterns,
      mlir::disc_ral::ParseFileString(pdll_files_),
      pdll_include_dirs,
      torch::kDefaultHelperFunctionDeclarations,
      torch::registerPredefinedHelperFunctions);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  // All ops in torch-mlir have side effects, so that the dce pass
  // in mlir cannot take effect on the graph of torch-mlir.
  // So we copied the code of dce from mlir and made some modifications,
  // so that it can take effect on the graph of torch-mlir.
  RewritePatternSet pdlDecPatterns(context);
  pdlDecPatterns.add<PdlDeadCodeElimination>(context);
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(pdlDecPatterns))))
    return signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createApplyDiscPdlPatternsPass(
        const std::string& pdll_files,
        const std::string& pdll_include_dirs) {
  return std::make_unique<ApplyDiscPdlPatternsPass>(
      pdll_files, pdll_include_dirs);
}
