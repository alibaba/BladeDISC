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

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mhlo/utils/legalize_to_linalg_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/transforms/PassDetail.h"
#include "mlir/disc/tools/disc-transform/utils.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"

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
LogicalResult emitOp(Operation* op, OpBuilder& b, IRMapping& mapping);

LogicalResult emitReturnOp(func::ReturnOp op, OpBuilder& b,
                           IRMapping& mapping) {
  SmallVector<Value> newOperands;
  for (Value v : op->getOperands()) newOperands.push_back(mapping.lookup(v));
  b.create<func::ReturnOp>(op.getLoc(), newOperands);
  return success();
}

LogicalResult emitDotGeneralOp(lmhlo::DotGeneralOp op, OpBuilder& b,
                               IRMapping& mapping, const std::string& name) {
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
                          IRMapping& mapping, const std::string& name) {
  auto resultTy = convertMemRefToTensorType(op->getOperand(0).getType());
  Location loc = op->getLoc();
  auto newOp = b.create<disc_linalg_ext::ConstantWrapperOp>(loc, resultTy,
                                                            op.getValue());
  mapping.erase(op->getOperand(0));
  mapping.map(op->getOperand(0), newOp.getResult());
  newOp->setAttr(kDISCLinalgTransformName, b.getStringAttr(name));
  return success();
}

LogicalResult emitBcastOp(Operation* op, OpBuilder& b, IRMapping& mapping,
                          const std::string& name) {
  SmallVector<NamedAttribute> prunedAttrs;
  if (auto dynBcastInDim = dyn_cast<lmhlo::DynamicBroadcastInDimOp>(op)) {
    prunedAttrs = linalg::getPrunedAttributeList(dynBcastInDim);
  } else if (auto bcastInDim = dyn_cast<lmhlo::BroadcastInDimOp>(op)) {
    prunedAttrs = linalg::getPrunedAttributeList(bcastInDim);
  } else if (auto bcast = dyn_cast<lmhlo::BroadcastOp>(op)) {
    prunedAttrs = linalg::getPrunedAttributeList(bcast);
  } else {
    return op->emitError() << "not support bcast op\n";
  }
  Value in = op->getOperand(0);
  auto inType = in.getType().cast<MemRefType>();
  Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
  auto outType = out.getType().cast<MemRefType>();
  auto dimAttr = op->getAttrOfType<DenseElementsAttr>("broadcast_dimensions");
  if (!dimAttr)
    return op->emitError() << "no broadcast_dimensions attr found\n";
  auto dimensions = dimAttr.getValues<int64_t>();

  // Determine dimension expressions based on whether the dimension is
  // expanding (0) or non-expanding (identity).
  // Our fusion pass will make sure there are only supported bcast ops in the
  // fusion Patterns, thus we can always decide if it's one of the above two
  // cases in the compile-time.
  SmallVector<AffineExpr> dimExprs(inType.getRank(), nullptr);
  if (inType.getRank() != dimensions.size())
    return op->emitError()
           << "size of broadcast_dimensions not equal rank of input\n";
  for (auto [inDimIdx, inDimSize] : llvm::enumerate(inType.getShape())) {
    int64_t outDimIdx = dimensions[inDimIdx];
    int64_t outDimSize = outType.getShape()[outDimIdx];
    bool isExpanding = (inDimSize == 1 && outDimSize != 1);
    dimExprs[inDimIdx] = isExpanding ? b.getAffineConstantExpr(0)
                                     : b.getAffineDimExpr(outDimIdx);
  }

  Location loc = op->getLoc();
  int64_t nloops = outType.getRank();
  Value newIn = mapping.lookup(in);
  Value newOut = mapping.lookup(out);
  auto newOp = b.create<linalg::GenericOp>(
      loc, TypeRange{newOut.getType()}, ValueRange{newIn}, ValueRange{newOut},
      llvm::ArrayRef({AffineMap::get(/*dimCount=*/nloops, /*symbolCount=*/0,
                                     dimExprs, b.getContext()),
                      b.getMultiDimIdentityMap(nloops)}),
      mhlo::getParallelAndReductionIterators(
          /*nLoops=*/nloops,
          /*nReduction=*/0),
      [&](OpBuilder& nestedBuilder, Location /*nested_loc*/, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
      },
      prunedAttrs);
  newOp->setAttr(kDISCLinalgTransformName, b.getStringAttr(name));
  mapping.erase(out);
  mapping.map(out, newOp->getResult(0));
  return success();
}

template <typename OpTy>
LogicalResult emitElemOp(OpTy op, OpBuilder& b, IRMapping& mapping,
                         const std::string& name) {
  if (!isa<lmhlo::LmhloOp>(op.getOperation()) || op->getNumOperands() < 2)
    return op->emitError() << "not support lmhlo op\n";

  auto loc = op.getLoc();
  // Find maximum rank / number of loops.
  auto getRank = [](Value v) {
    return v.getType().cast<ShapedType>().getRank();
  };
  auto isScalar = [&](Value v) { return getRank(v) == 0; };
  auto ins = op->getOperands().drop_back();
  auto it = llvm::find_if_not(ins, isScalar);
  Value maxRankArg = it != ins.end() ? *it : ins.front();
  int64_t nloops = getRank(maxRankArg);
  if (nloops == 0) return op->emitError() << "not support 0d elem ops a.t.m.\n";

  // Apply only if all operands are scalar or have the same rank. Some ops,
  // like `mhlo.select`, support implicit broadcasting of scalars.
  if (!llvm::all_of(op->getOperands(), [&](Value v) {
        int64_t r = getRank(v);
        return r == 0 || r == nloops;
      })) {
    return op->emitError() << "Operands must be os same rank or scalar.\n";
  }

  Value out = op->getOperands().back();
  if (getRank(out) != nloops)
    return op->emitError() << "output of op should have the max rank\n";
  Value newOut = mapping.lookup(out);
  SmallVector<Value> newIns;
  AffineMap scalarMap = AffineMap::get(nloops, 0, b.getContext());
  AffineMap idMap = b.getMultiDimIdentityMap(nloops);
  SmallVector<AffineMap, 4> maps;
  for (Value in : ins) {
    newIns.push_back(mapping.lookup(in));
    maps.push_back(isScalar(in) ? scalarMap : idMap);
  }
  maps.push_back(b.getMultiDimIdentityMap(nloops));

  auto newOp = b.create<linalg::GenericOp>(
      loc, TypeRange{newOut.getType()}, newIns, newOut, maps,
      mhlo::getParallelAndReductionIterators(
          /*nLoops=*/nloops,
          /*nReduction=*/0),
      [&](OpBuilder& nestedBuilder, Location /*nested_loc*/, ValueRange args) {
        Type innerResultTy = getElementTypeOrSelf(newOut);
        auto argvec = llvm::to_vector<2>(args.take_front(newIns.size()));
        auto innerResult = LhloOpToStdScalarOp::map<OpTy>(
            op, innerResultTy, argvec, &nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(loc, innerResult);
      },
      linalg::getPrunedAttributeList(op));

  newOp->setAttr(kDISCLinalgTransformName, b.getStringAttr(name));
  mapping.erase(out);
  mapping.map(out, newOp->getResult(0));
  return success();
}

LogicalResult emitLmhloOp(Operation* op, OpBuilder& b, IRMapping& mapping,
                          const std::string& name) {
  if (auto dotGeneralOp = dyn_cast<lmhlo::DotGeneralOp>(op)) {
    return emitDotGeneralOp(dotGeneralOp, b, mapping, name);
  } else if (auto constOp = dyn_cast<lmhlo::ConstantOp>(op)) {
    return emitConstOp(constOp, b, mapping, name);
  } else if (isa<lmhlo::BroadcastInDimOp, lmhlo::BroadcastOp,
                 lmhlo::DynamicBroadcastInDimOp>(op)) {
    return emitBcastOp(op, b, mapping, name);
  }

  // clang-format off

  #define ELEM_OP_HANDLER(opType) \
    else if (auto t##opType = dyn_cast<lmhlo::opType>(op)) { \
      return emitElemOp(t##opType, b, mapping, name); \
    }
  ELEM_OP_HANDLER(AddOp)
  ELEM_OP_HANDLER(SubtractOp)
  ELEM_OP_HANDLER(AbsOp)
  ELEM_OP_HANDLER(CeilOp)
  ELEM_OP_HANDLER(ConvertOp)
  ELEM_OP_HANDLER(CopyOp)
  ELEM_OP_HANDLER(CosineOp)
  ELEM_OP_HANDLER(ExpOp)
  ELEM_OP_HANDLER(FloorOp)
  ELEM_OP_HANDLER(IsFiniteOp)
  ELEM_OP_HANDLER(LogOp)
  ELEM_OP_HANDLER(Log1pOp)
  ELEM_OP_HANDLER(LogisticOp)
  ELEM_OP_HANDLER(NegOp)
  ELEM_OP_HANDLER(NotOp)
  ELEM_OP_HANDLER(RsqrtOp)
  ELEM_OP_HANDLER(SignOp)
  ELEM_OP_HANDLER(SineOp)
  ELEM_OP_HANDLER(SqrtOp)
  ELEM_OP_HANDLER(TanhOp)
  ELEM_OP_HANDLER(RoundOp)
  ELEM_OP_HANDLER(RoundNearestEvenOp)
  ELEM_OP_HANDLER(AddOp)
  ELEM_OP_HANDLER(AndOp)
  ELEM_OP_HANDLER(CompareOp)
  ELEM_OP_HANDLER(DivOp)
  ELEM_OP_HANDLER(MaxOp)
  ELEM_OP_HANDLER(MinOp)
  ELEM_OP_HANDLER(MulOp)
  ELEM_OP_HANDLER(OrOp)
  ELEM_OP_HANDLER(PowOp)
  ELEM_OP_HANDLER(RemOp)
  ELEM_OP_HANDLER(SubtractOp)
  ELEM_OP_HANDLER(SelectOp)
  ELEM_OP_HANDLER(ClampOp)

  #undef ELEM_OP_HANDLER

  // clang-format on

  // TODO(wyzero): support other lmhlo ops.
  return failure();
}

LogicalResult emitLmhloFusionOp(lmhlo::FusionOp op, OpBuilder& b,
                                IRMapping& mapping) {
  TransformNameAssigner assigner;
  for (Operation& op : op.getRegion().getBlocks().front()) {
    if (isa<lmhlo::TerminatorOp>(&op)) continue;
    if (failed(emitLmhloOp(&op, b, mapping, assigner.nameNewOperation(&op))))
      return failure();
  }
  return success();
}

LogicalResult emitOp(Operation* op, OpBuilder& b, IRMapping& mapping) {
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
  IRMapping mapping;
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
