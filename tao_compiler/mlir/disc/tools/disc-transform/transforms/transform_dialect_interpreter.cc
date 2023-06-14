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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterPassBase.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h"
#include "mlir/disc/tools/disc-transform/transforms/PassDetail.h"
#include "mlir/disc/tools/disc-transform/utils.h"

#define DEBUG_TYPE "disc-transform-dialect-interpreter"

// This file implements the logic to apply transform dialect ops for codegen.

namespace mlir {
namespace disc_ral {

void addTransformDialectDependentDialects(DialectRegistry& registry) {
  // TODO: this is only necessary to make registry subset happy when running
  // the lowering to LLVM. The lowering should be changed to stop using the
  // nested pass manager and this will go away.

  // clang-format off
  registry.insert<
      arith::ArithDialect,
      AffineDialect,
      bufferization::BufferizationDialect,
      disc_ral::disc_linalg_ext::DISCLinalgExtDialect,
      iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
      func::FuncDialect,
      linalg::LinalgDialect,
      linalg::transform::LinalgTransformDialect,
      LLVM::LLVMDialect,
      pdl::PDLDialect,
      pdl_interp::PDLInterpDialect,
      scf::SCFDialect,
      tensor::TensorDialect,
      transform::TransformDialect,
      vector::VectorDialect,
      nvgpu::NVGPUDialect,
      disc_shape::DISCShapeDialect
  >();
  // clang-format on

  // TODO: these should be registered by the extension instead, but there is
  // no support for it in core currently.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);

  registry.addExtensions<
      mlir::iree_compiler::IREE::LinalgExt::LinalgExtTransformOpsExtension,
      transform_ext::StructuredTransformOpsExtension>();
  linalg::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  registerTransformDialectCommonExtension(registry);
  disc_linalg_ext::registerTilingInterfaceExternalModels(registry);
  disc_linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
}

namespace {

struct DiscTransformDialectInterpreterPass
    : public DiscTransformDialectInterpreterPassBase<
          DiscTransformDialectInterpreterPass> {
  explicit DiscTransformDialectInterpreterPass(const std::string& fileName,
                                               bool enableExpensiveChecks)
      : DiscTransformDialectInterpreterPassBase<
            DiscTransformDialectInterpreterPass>::
            DiscTransformDialectInterpreterPassBase() {
    this->transformFileName_ = fileName;
    this->enableExpensiveChecks_ = enableExpensiveChecks;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    addTransformDialectDependentDialects(registry);
  }

  void runOnOperation() override;
};

void DiscTransformDialectInterpreterPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (transformFileName_.empty()) {
    llvm::errs() << "no transform file name specified, assuming the transform "
                    "module is embedded in the IR next to the top-level\n";
    // Parse transform ops from the module itself.

    // Note that We cloned the transform IR into a standalone module to avoid
    // it's being modified during transformation.
    auto transformModule = ModuleOp::create(module->getLoc());
    auto b = OpBuilder::atBlockBegin(transformModule.getBody());
    for (auto op : module.getBody()->getOps<transform::TransformOpInterface>())
      b.clone(*op);

    // Apply the transform IR.
    for (auto op :
         transformModule.getBody()->getOps<transform::TransformOpInterface>()) {
      if (failed(transform::applyTransforms(
              module, op, {},
              transform::TransformOptions().enableExpensiveChecks(
                  enableExpensiveChecks_))))
        return signalPassFailure();
    }
  } else {
    // parse transform ops from a standalone file.
    OwningOpRef<ModuleOp> transformModule;
    if (failed(parseTransformModuleFromFile(
            module.getContext(), transformFileName_, transformModule))) {
      llvm::errs() << "failed to load transform ops from file "
                   << transformFileName_ << "\n";
      return signalPassFailure();
    }
    for (auto op : transformModule.get()
                       .getBody()
                       ->getOps<transform::TransformOpInterface>()) {
      if (failed(transform::applyTransforms(
              module, op, {},
              transform::TransformOptions().enableExpensiveChecks(
                  enableExpensiveChecks_))))
        return signalPassFailure();
    }
  }
}

struct DiscTransformDialectEraseSchedulePass
    : public DiscTransformDialectEraseSchedulePassBase<
          DiscTransformDialectEraseSchedulePass> {
  void runOnOperation() override;
};

void DiscTransformDialectEraseSchedulePass::runOnOperation() {
  getOperation()->walk<WalkOrder::PreOrder>([&](Operation* nestedOp) {
    if (isa<transform::TransformOpInterface>(nestedOp)) {
      nestedOp->erase();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createDiscTransformDialectInterpreterPass(const std::string& fileName,
                                          bool enableExpensiveChecks) {
  return std::make_unique<DiscTransformDialectInterpreterPass>(
      fileName, enableExpensiveChecks);
}

std::unique_ptr<OperationPass<ModuleOp>>
createDiscTransformDialectEraseSchedulePass() {
  return std::make_unique<DiscTransformDialectEraseSchedulePass>();
}

}  // namespace disc_ral
}  // namespace mlir
