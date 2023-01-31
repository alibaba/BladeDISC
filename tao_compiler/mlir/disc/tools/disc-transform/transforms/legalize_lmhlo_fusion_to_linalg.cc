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

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/legalize_to_linalg_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/transforms/PassDetail.h"
#include "mlir/disc/tools/disc-transform/utils.h"

#define DEBUG_TYPE "disc-legalize-lmhlo-fusion-to-linalg"

// This file implements the logic to convert a lmhlo fusion op in side a
// function to its linalg on tensor equivalent.
//
// Assume: Each candidate function should only have one lhlo fusion + one return
// ops. Example convert from:
// ```
//  func.func @name(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C:
//  memref<?x?xf32>) {
//    "lmhlo.fusion"() ({
//      "lmhlo.dot_general"(%A, %B, %C) {...}
//      "lmhlo.terminator"() : () -> ()
//    })
//    return %arg3 : memref<?x?xf32>
//  }
// to:
//  func.func @name(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C:
//  tensor<?x?xf32>) {
//    %0 = linalg.matmul(%A, %B, ...)
//    return %0 : tensor<?x?xf32>
//  }

namespace mlir {
namespace disc_ral {
namespace {

using func::FuncOp;

struct DiscLegalizeLmhloFusionToLinalgPass
    : public DiscLegalizeLmhloFusionToLinalgPassBase<
          DiscLegalizeLmhloFusionToLinalgPass> {
  void runOnOperation() override;
};

Type convertMemRefToTensorType(Type type) {
  auto memrefTy = type.cast<MemRefType>();
  return RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
}

SmallVector<Type> convertMemRefToTensorType(TypeRange range) {
  SmallVector<Type> newTypes;
  for (Type type : range) newTypes.push_back(convertMemRefToTensorType(type));
  return newTypes;
}

// forward decalration.
LogicalResult emitOp(Operation* op, OpBuilder& b,
                     BlockAndValueMapping& mapping);

LogicalResult emitReturnOp(func::ReturnOp op, OpBuilder& b,
                           BlockAndValueMapping& mapping) {
  SmallVector<Value> newOperands;
  for (Value v : op->getOperands()) newOperands.push_back(mapping.lookup(v));
  b.create<func::ReturnOp>(op.getLoc(), newOperands);
  return success();
}

LogicalResult emitDotGeneralOp(lmhlo::DotGeneralOp op, OpBuilder& b,
                               BlockAndValueMapping& mapping,
                               const std::string& name) {
  Value A = op->getOperand(0);
  Value B = op->getOperand(1);
  Value C = op->getOperand(2);

  Value newA = mapping.lookup(A);
  Value newB = mapping.lookup(B);
  Value newC = mapping.lookup(C);

  auto lhsTy = A.getType().cast<MemRefType>();
  auto rhsTy = B.getType().cast<MemRefType>();
  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
    return op->emitError() << "only support rank 2 GEMM a.t.m.\n";
  }

  auto dimNumbers = op.getDotDimensionNumbers();
  auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
  auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
  auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
  auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
  if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
    return op->emitError() << "only support exactly 1 contract dim\n";
  }

  auto numContracting = lhsContractingDims.size();
  auto outputType = newC.getType().cast<ShapedType>();
  auto targetRank = outputType.getRank();
  auto totalLoopCount = numContracting + targetRank;
  auto lhsRank = lhsTy.getRank();
  auto lhsExtraDims =
      lhsRank - lhsBatchingDims.size() - lhsContractingDims.size();
  auto rhsRank = rhsTy.getRank();

  // 1, fill the output buffer using zero.
  auto ty = newC.getType().cast<ShapedType>();
  auto zeroAttr = b.getZeroAttr(ty.getElementType());
  Location loc = op->getLoc();
  Value zero = b.create<arith::ConstantOp>(loc, zeroAttr);
  auto fillOp = b.create<linalg::FillOp>(loc, zero, newC);
  fillOp->setAttr(kDISCLinalgTransformName, b.getStringAttr(name));

  // 2, build the matmul op.
  Operation* matmulOp;
  if (lhsContractingDims[0] == 1 && rhsContractingDims[0] == 0) {
    // nn format gemm
    matmulOp = b.create<linalg::MatmulOp>(loc, ValueRange{newA, newB},
                                          ValueRange{fillOp.getResult(0)});
  } else {
    // nt, tn, tt format gemm
    SmallVector<AffineMap, 3> indexingMaps;

    auto getMap = [&](int64_t rank, ArrayRef<int64_t> batchingDims,
                      ArrayRef<int64_t> contractingDims, size_t extraDims) {
      llvm::SmallVector<AffineExpr> indices(rank);
      for (const auto& i : llvm::enumerate(batchingDims)) {
        indices[i.value()] = b.getAffineDimExpr(i.index());
      }
      for (const auto& i : llvm::enumerate(contractingDims)) {
        indices[i.value()] = b.getAffineDimExpr(i.index() + targetRank);
      }
      for (int i = 0; i < rank; ++i) {
        if (!indices[i]) {
          indices[i] = b.getAffineDimExpr(extraDims++);
        }
      }
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/totalLoopCount,
                                            /*symbolCount=*/0, indices,
                                            op->getContext()));
    };
    getMap(lhsRank, lhsBatchingDims, lhsContractingDims,
           lhsBatchingDims.size());
    getMap(rhsRank, rhsBatchingDims, rhsContractingDims,
           rhsBatchingDims.size() + lhsExtraDims);

    {
      SmallVector<AffineExpr, 4> dimExprs;
      dimExprs.reserve(targetRank);
      for (unsigned i = 0; i < targetRank; ++i)
        dimExprs.push_back(b.getAffineDimExpr(i));
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/totalLoopCount,
                                            /*symbolCount=*/0, dimExprs,
                                            op.getContext()));
    }

    matmulOp = b.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{newA, newB},
        /*outputBuffers=*/ValueRange{fillOp.getResult(0)}, indexingMaps,
        mhlo::getParallelAndReductionIterators(
            /*nLoops=*/totalLoopCount,
            /*nReduction=*/numContracting),
        [](OpBuilder& b, Location loc, ValueRange) {
          ImplicitLocOpBuilder builder(loc, b);
          linalg::MatmulOp::regionBuilder(builder, *b.getInsertionBlock(), {});
        },
        linalg::getPrunedAttributeList(op));
  }
  matmulOp->setAttr(kDISCLinalgTransformName, b.getStringAttr(name));
  mapping.erase(C);
  mapping.map(C, matmulOp->getResult(0));

  return success();
}

LogicalResult emitConstOp(lmhlo::ConstantOp op, OpBuilder& b,
                          BlockAndValueMapping& mapping,
                          const std::string& name) {
  auto resultTy = convertMemRefToTensorType(op->getOperand(0).getType());
  Location loc = op->getLoc();
  auto newOp = b.create<disc_linalg_ext::ConstantWrapperOp>(loc, resultTy,
                                                            op.getValue());
  mapping.erase(op->getOperand(0));
  mapping.map(op->getOperand(0), newOp.getResult());
  newOp->setAttr(kDISCLinalgTransformName, b.getStringAttr(name));
  return success();
}

LogicalResult emitLmhloOp(Operation* op, OpBuilder& b,
                          BlockAndValueMapping& mapping,
                          const std::string& name) {
  if (auto dotGeneralOp = dyn_cast<lmhlo::DotGeneralOp>(op)) {
    return emitDotGeneralOp(dotGeneralOp, b, mapping, name);
  } else if (auto constOp = dyn_cast<lmhlo::ConstantOp>(op)) {
    return emitConstOp(constOp, b, mapping, name);
  }
  // TODO(wyzero): support other lmhlo ops.
  return failure();
}

LogicalResult emitLmhloFusionOp(lmhlo::FusionOp op, OpBuilder& b,
                                BlockAndValueMapping& mapping) {
  TransformNameAssigner assigner;
  for (Operation& op : op.getRegion().getBlocks().front()) {
    if (isa<lmhlo::TerminatorOp>(&op)) continue;
    if (failed(emitLmhloOp(&op, b, mapping, assigner.nameNewOperation(&op))))
      return failure();
  }
  return success();
}

LogicalResult emitOp(Operation* op, OpBuilder& b,
                     BlockAndValueMapping& mapping) {
  if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
    return emitReturnOp(returnOp, b, mapping);
  } else if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(op)) {
    return emitLmhloFusionOp(fusionOp, b, mapping);
  }
  return failure();
}

LogicalResult rewriteFuncOp(FuncOp func, FuncOp& newFunc) {
  if (func.getBody().getBlocks().size() != 1) return failure();
  if (func.getBody().front().getOperations().size() != 2) return failure();
  auto fusionOp = dyn_cast<lmhlo::FusionOp>(
      &func.getBody().front().getOperations().front());
  if (!fusionOp) return failure();

  OpBuilder b(func.getContext());
  auto newArgumentTypes = convertMemRefToTensorType(func.getArgumentTypes());
  auto newResultTypes = convertMemRefToTensorType(func.getResultTypes());
  auto newFuncType =
      FunctionType::get(func->getContext(), newArgumentTypes, newResultTypes);
  newFunc = b.create<FuncOp>(func.getLoc(), func.getName(), newFuncType);
  Block* entryBlock = newFunc.addEntryBlock();
  BlockAndValueMapping mapping;
  for (const auto& z :
       llvm::zip(func.getArguments(), entryBlock->getArguments()))
    mapping.map(std::get<0>(z), std::get<1>(z));
  b.setInsertionPoint(entryBlock, entryBlock->begin());
  for (Operation& op : func.getBody().front().getOperations())
    if (failed(emitOp(&op, b, mapping))) return failure();
  return success();
}

void DiscLegalizeLmhloFusionToLinalgPass::runOnOperation() {
  for (FuncOp funcOp :
       llvm::to_vector<4>(getOperation().getOps<func::FuncOp>())) {
    FuncOp newFuncOp;
    if (failed(rewriteFuncOp(funcOp, newFuncOp))) {
      signalPassFailure();
      return;
    }
    funcOp->erase();
    getOperation().push_back(newFuncOp);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createDiscLegalizeLmhloFusionToLinalgPass() {
  return std::make_unique<DiscLegalizeLmhloFusionToLinalgPass>();
}

}  // namespace disc_ral
}  // namespace mlir
