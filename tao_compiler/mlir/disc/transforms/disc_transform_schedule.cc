/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/disc/transforms/disc_transform_schedule.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/transforms/passes.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "tensorflow/tsl/platform/default/logging.h"

// This file implements logic to assign transform schedule for the given pattern
// description.

namespace mlir {
namespace disc_ral {

namespace {

std::unordered_map<PatternKind, std::string>& getPatternKindToStringMap() {
  static std::unordered_map<PatternKind, std::string> patternKindToStringMap;
  return patternKindToStringMap;
}

std::unordered_map<std::string, PatternKind>& getStringToPatternKindMap() {
  static std::unordered_map<std::string, PatternKind> stringToPatternKindMap;
  return stringToPatternKindMap;
}

bool PatternKindAndStringMapRegistrar = []() {
  auto& patternKindToStringMap = getPatternKindToStringMap();
  auto& stringToPatternKindMap = getStringToPatternKindMap();
  patternKindToStringMap.emplace(PatternKind::kNone, "kNone");
  patternKindToStringMap.emplace(PatternKind::kGEMM, "kGEMM");
  for (auto& pair : patternKindToStringMap) {
    stringToPatternKindMap[pair.second] = pair.first;
  }
  return true;
}();

using transform::FuseIntoContainingOp;
using transform::FuseOp;
using transform::MatchOp;
using transform::SplitHandlesOp;
using transform::TileOp;
using transform::TileToForeachThreadOp;
using transform::VectorizeOp;

MatchOp buildMatchOp(OpBuilder& b, Location& loc, Value target,
                     ArrayRef<StringRef> ops, StringRef name = {}) {
  ArrayAttr opNames;
  if (!ops.empty()) {
    opNames = b.getStrArrayAttr(ops);
  }
  DictionaryAttr attrs;
  if (!name.empty()) {
    attrs = b.getDictionaryAttr(
        b.getNamedAttr(kDISCLinalgTransformName, b.getStringAttr(name)));
  }

  return b.create<MatchOp>(loc, pdl::OperationType::get(b.getContext()), target,
                           opNames, transform::MatchInterfaceEnumAttr{}, attrs,
                           TypeAttr{});
}

TileToForeachThreadOp buildTileToForEachThreadOp(OpBuilder& b, Location& loc,
                                                 Value target,
                                                 ArrayRef<int64_t> numThreads) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<TileToForeachThreadOp>(
      loc, TypeRange{pdlType, pdlType}, target,
      /* num_threads */ ValueRange{},
      /* tile_sizes */ ValueRange{},
      /* static_num_threads */ b.getI64ArrayAttr(numThreads),
      /* static_num_threads */ ArrayAttr{},
      /* thread_dim_mapping */ ArrayAttr{});
}

FuseIntoContainingOp buildFuseIntoContainingOp(OpBuilder& b, Location& loc,
                                               Value target, Value anchor) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<FuseIntoContainingOp>(loc, pdlType, target, anchor);
}

FuseOp buildFuseOp(OpBuilder& b, Location& loc, Value target,
                   ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> interchange) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  SmallVector<Type> loopTypes;
  for (int64_t tileSize : tileSizes)
    if (tileSize) loopTypes.push_back(pdlType);
  return b.create<FuseOp>(loc, pdlType, loopTypes, target,
                          b.getI64ArrayAttr(tileSizes),
                          b.getI64ArrayAttr(interchange));
}

TileOp buildTileOp(OpBuilder& b, Location& loc, Value target,
                   ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> interchange) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  SmallVector<Type> loopTypes;
  for (int64_t tileSize : tileSizes)
    if (tileSize) loopTypes.push_back(pdlType);
  return b.create<TileOp>(loc, pdlType, loopTypes, target, ValueRange{},
                          b.getI64ArrayAttr(tileSizes),
                          b.getI64ArrayAttr(interchange));
}

transform_dialect::ApplyPatternsOp buildRunCanonicalizer(OpBuilder& b,
                                                         Location& loc,
                                                         Value target) {
  return b.create<transform_dialect::ApplyPatternsOp>(loc, target, true);
}

transform::GetProducerOfOperand buildGetProducerOfOperand(OpBuilder& b,
                                                          Location& loc,
                                                          Value target,
                                                          int64_t operandIdx) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<transform::GetProducerOfOperand>(loc, pdlType, target,
                                                   operandIdx);
}

transform_dialect::FoldProducerExtractSliceOp buildFoldProducerExtractSlice(
    OpBuilder& b, Location& loc, Value target, int64_t repeat) {
  return b.create<transform_dialect::FoldProducerExtractSliceOp>(loc, target,
                                                                 repeat);
}

transform::PadOp buildPadOp(OpBuilder& b, Location& loc, Value target,
                            ArrayRef<int64_t> paddingDimensions) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  // TODO(wyzero): support other types.
  SmallVector<Attribute> paddingAttrs(paddingDimensions.size(),
                                      b.getZeroAttr(b.getF32Type()));
  return b.create<transform::PadOp>(loc, pdlType, target,
                                    b.getArrayAttr(paddingAttrs),
                                    b.getI64ArrayAttr(paddingDimensions),
                                    ArrayAttr{}, ArrayAttr{}, ArrayAttr{});
}

transform::GetParentForOp buildGetParentForOp(OpBuilder& b, Location& loc,
                                              Value target, int64_t num_loops) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<transform::GetParentForOp>(loc, pdlType, target, num_loops);
}

transform_dialect::CacheReadOp buildCacheRead(OpBuilder& b, Location& loc,
                                              Value target, Value anchor,
                                              ArrayRef<int64_t> tileLevels,
                                              ArrayRef<int64_t> tileSizes,
                                              bool padded,
                                              ArrayRef<int64_t> permutation) {
  return b.create<transform_dialect::CacheReadOp>(
      loc, target, anchor, tileLevels, tileSizes, padded, permutation);
}

transform_dialect::LowerMultiLevelPackToLoopOp buildLowerMultiLevelPackToLoop(
    OpBuilder& b, Location& loc, Value target) {
  return b.create<transform_dialect::LowerMultiLevelPackToLoopOp>(loc, target);
}

VectorizeOp buildVectorize(OpBuilder& b, Location& loc, Value target,
                           bool vectorizePad) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<VectorizeOp>(loc, pdlType, target, vectorizePad);
}

transform_dialect::DISCBufferizeOp buildDISCBufferize(OpBuilder& b,
                                                      Location& loc,
                                                      Value target) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<transform_dialect::DISCBufferizeOp>(loc, pdlType, target);
}

void buildLowerVectors(OpBuilder& b, Location& loc, ArrayRef<int64_t> stages,
                       StringRef contractionLowering,
                       StringRef multireductionLowering,
                       StringRef splitTransfers, bool unrollVectorTransfers,
                       StringRef transposeLowering,
                       bool transposeAvx2Lowering) {
  b.create<transform_ext::LowerVectorsOp>(
      loc, b.getI64ArrayAttr(stages), contractionLowering,
      multireductionLowering, splitTransfers, unrollVectorTransfers,
      transposeLowering, transposeAvx2Lowering);
}

SplitHandlesOp buildSplitHandlesOp(OpBuilder& b, Location& loc, Value target,
                                   uint64_t num_result_handles) {
  SmallVector<Type> pdlTypes(num_result_handles,
                             pdl::OperationType::get(b.getContext()));
  return b.create<SplitHandlesOp>(loc, pdlTypes, target, num_result_handles);
}

transform_dialect::InlineReductionInitializerOp
buildInlineReductionInitializerOp(OpBuilder& b, Location& loc, Value initOp,
                                  Value loopOp, Value readerOp) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<transform_dialect::InlineReductionInitializerOp>(
      loc, pdlType, initOp, loopOp, readerOp);
}

transform_dialect::DecomposeVectorsOp buildDecomposeVectors(
    OpBuilder& b, Location& loc, Value target, int64_t vectorSize) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  return b.create<transform_dialect::DecomposeVectorsOp>(loc, pdlType, target,
                                                         vectorSize);
}

class ParsedFromFileScheduleFactory : public ScheduleFactoryWithNoGuard {
 public:
  explicit ParsedFromFileScheduleFactory(int64_t id, PatternKind kind,
                                         ArrayRef<StringRef> tags,
                                         ModuleOp transformModule);
  LogicalResult assignSchedule(PatternDescription&, ModuleOp) override;

 private:
  ModuleOp transformModule_;
};

ParsedFromFileScheduleFactory::ParsedFromFileScheduleFactory(
    int64_t id, PatternKind kind, ArrayRef<StringRef> tags,
    ModuleOp transformModule)
    : ScheduleFactoryWithNoGuard(id, kind, tags),
      transformModule_(transformModule) {}

LogicalResult ParsedFromFileScheduleFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m) {
  OpBuilder b(m);
  for (auto& op : transformModule_.getBody()->getOperations()) {
    if (!isa<transform::TransformOpInterface>(&op)) continue;
    m.push_back(b.clone(op));
  }
  return success();
}

class Aarch64GEMMDefaultScheduleFactory : public ScheduleFactoryWithNoGuard {
 public:
  using ScheduleFactoryWithNoGuard::ScheduleFactoryWithNoGuard;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp) override;
};

LogicalResult Aarch64GEMMDefaultScheduleFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto seqOp = b.create<transform_ext::CanonicalizedSequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate, Value{});
  seqOp.getBody().push_back(new Block);
  auto& bodyBlock = seqOp.getBody().front();
  auto pdlOpType = pdl::OperationType::get(ctx);
  bodyBlock.addArgument(pdl::OperationType::get(ctx), loc);
  b.setInsertionPointToStart(&bodyBlock);
  Value variant = bodyBlock.getArgument(0);

  auto& fusionPattern = pd.getFusionPattern();
  auto nameMap = TransformNameAssigner(fusionPattern.getOpList()).getNameMap();
  auto dotOp =
      dyn_cast_or_null<lmhlo::DotGeneralOp>(fusionPattern.getDominantOp());
  if (!dotOp) {
    return m->emitError() << "expect dot_general op as dominant\n";
  }
  Value lhs = dotOp->getOperand(0);
  Value rhs = dotOp->getOperand(1);
  auto lhsTy = lhs.getType().cast<MemRefType>();
  auto rhsTy = rhs.getType().cast<MemRefType>();
  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
    return m->emitError() << "only support rank 2 GEMM a.t.m.\n";
  }

  auto dimNumbers = dotOp.getDotDimensionNumbers();
  auto lhsCntractingDims = dimNumbers.getLhsContractingDimensions();
  auto rhsCntractingDims = dimNumbers.getRhsContractingDimensions();
  if (lhsCntractingDims.size() != 1 || rhsCntractingDims.size() != 1) {
    return m->emitError() << "only support exactly 1 contract dim\n";
  }
  bool lhsTranspose = (lhsCntractingDims[0] == lhsTy.getRank() - 2);
  bool rhsTranspose = (rhsCntractingDims[0] == rhsTy.getRank() - 1);
  int64_t M = lhsTranspose ? lhsTy.getShape()[lhsTy.getRank() - 1]
                           : lhsTy.getShape()[lhsTy.getRank() - 2];
  int64_t K = lhsTranspose ? lhsTy.getShape()[lhsTy.getRank() - 2]
                           : lhsTy.getShape()[lhsTy.getRank() - 1];
  int64_t N = rhsTranspose ? rhsTy.getShape()[rhsTy.getRank() - 2]
                           : rhsTy.getShape()[rhsTy.getRank() - 1];

  // build handle to target dot op.
  Value fillAndMatmul = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
  auto matmulSplitOp = buildSplitHandlesOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);

  // transform.structured.tile_to_foreach_thread_op %matmul num_threads [1, 1]
  auto forEachThreadOp = buildTileToForEachThreadOp(b, loc, matmul, {1, 1});
  Value forEachThreadLoop = forEachThreadOp->getResult(0);
  Value tiledMatmul = forEachThreadOp->getResult(1);

  // transform.structured.fuse_into_containing_op %fill into %0#0
  auto fuseIntoContainingOp =
      buildFuseIntoContainingOp(b, loc, fill, forEachThreadLoop);

  // first/second level tile size for dimension m
  int64_t M0 = 288, M1 = 6;
  // first/second level tile size for dimension n
  int64_t N0 = 48, N1 = 16;
  // first level tile size for dimension k
  int64_t K0 = 1;

  // first level tile and fuse matmul and fill op.
  auto fuseOp0 = buildFuseOp(b, loc, tiledMatmul, {M0, N0, 0}, {0, 1, 2});

  // second level tile and fuse matmul and fill op.
  auto fuseOp1 =
      buildFuseOp(b, loc, fuseOp0->getResult(0), {M1, N1, 0}, {0, 1, 2});

  // gemm reduction axis tiling
  auto tileOp =
      buildTileOp(b, loc, fuseOp1->getResult(0), {0, 0, K0}, {0, 1, 2});

  variant = buildRunCanonicalizer(b, loc, variant);

  // fold two extract_slice ops generated by two-level tiling. It's needed to
  // enable following pad and hosit schedule.
  Value weightInnerSlice =
      buildGetProducerOfOperand(b, loc, tileOp->getResult(0), 1);
  buildFoldProducerExtractSlice(b, loc, weightInnerSlice, 2);

  // pad to match the requirement of hardware vector/tensor instruction.
  auto padOp = buildPadOp(b, loc, tileOp->getResult(0), {0, 1, 2});

  Value padForInput = buildGetProducerOfOperand(b, loc, padOp, 0);
  Value padForWeight = buildGetProducerOfOperand(b, loc, padOp, 1);

  // Check if we need to pad dimension `m/n/k` if input or weight is packed
  bool mIsPadded = (M1 != 1) && (M == ShapedType::kDynamicSize ||
                                 (M > M0 && (M % M0 != 0 || M0 % M1 != 0)) ||
                                 (M <= M0 && M > M1 && M % M1 != 0));
  bool nIsPadded = (N1 != 1) && (N == ShapedType::kDynamicSize ||
                                 (N > N0 && (N % N0 != 0 || N0 % N1 != 0)) ||
                                 (N <= N0 && N > N1 && N % N1 != 0));
  bool kIsPadded =
      (K0 != 1) && (K == ShapedType::kDynamicSize || K > K0 && K % K0 != 0);

  // Check if we need to pack the input:
  bool packInput = ((M == ShapedType::kDynamicSize || M >= M1) &&
                    (K == ShapedType::kDynamicSize || K > K0) &&
                    (N == ShapedType::kDynamicSize || N > N0));
  if (packInput) {
    // supposed loop order:
    //  loop_m0
    //   loop_n0
    //    loop_m1
    //     loop_n1
    //      loop_k0 {
    //        inner_most_gemm
    //      }
    // We want to cache the packed A below loop_m0 and above loop_n0.
    // Thus the initial loop_level is 4.
    int loopLevel = 4;
    // in case:
    // - the size of dimension N <= N0, then loop_n0 will be folded.
    loopLevel -= (N != ShapedType::kDynamicSize && N <= N0);
    // - the size of dimension M <= M1, then loop_m1 will be folded.
    loopLevel -= (M != ShapedType::kDynamicSize && M <= M1);
    // - the size of dimension N <= N1, then loop_n1 will be folded.
    loopLevel -= (N != ShapedType::kDynamicSize && N <= N1);
    // - the size of dimension K <= K0, then loop_k0 will be folded.
    loopLevel -= (K != ShapedType::kDynamicSize && K <= K0);

    if (loopLevel <= 0) {
      return m->emitError()
             << "failed to cache the packed input due to loopLevel = "
             << loopLevel << " is invalid\n";
    }
    auto loopN0 = buildGetParentForOp(b, loc, padForInput, loopLevel);
    bool inputIsPadded = mIsPadded || kIsPadded;
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (lhsTranspose) {
      tileSizes = {K0, M1};
      permutation = {2, 0, 1, 3};
    } else {
      tileSizes = {M1, K0};
      permutation = {0, 2, 3, 1};
    }
    buildCacheRead(b, loc, padForInput, loopN0, {1, 1}, tileSizes,
                   inputIsPadded, permutation);
  }

  // Check if we need to pack the weight, one of the following conditions:
  // - if M, N and K are both dynamic, we always pad input a.t.m.
  // - if N is known and N >= N0 && N0 > N1
  bool packWeight = ((K == ShapedType::kDynamicSize || K > K0) &&
                     (N == ShapedType::kDynamicSize || N > N1));
  if (packWeight) {
    bool weightIsPadded = nIsPadded || kIsPadded;
    forEachThreadLoop = buildMatchOp(b, loc, variant, {"scf.foreach_thread"});
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (rhsTranspose) {
      tileSizes = {N1, K0};
      permutation = {0, 2, 3, 1};
    } else {
      tileSizes = {K0, N1};
      permutation = {2, 0, 1, 3};
    }
    buildCacheRead(b, loc, padForWeight, forEachThreadLoop, {1, 1}, tileSizes,
                   weightIsPadded, permutation);
  }

  variant = buildRunCanonicalizer(b, loc, variant);

  Value multiLevelPackOps =
      buildMatchOp(b, loc, variant, {"disc_linalg_ext.multi_level_pack"});
  buildLowerMultiLevelPackToLoop(b, loc, multiLevelPackOps);

  variant = buildRunCanonicalizer(b, loc, variant);

  Value func = buildMatchOp(b, loc, variant, {"func.func"});
  buildVectorize(b, loc, func, true);

  variant = buildRunCanonicalizer(b, loc, variant);
  variant = buildDISCBufferize(b, loc, variant);

  buildLowerVectors(b, loc, {0, 1, 2, 3, 4}, "outerproduct", "innerparallel",
                    "linalg-copy", true, "eltwise", false);
  buildLowerVectors(b, loc, {5, 6, 7}, "outerproduct", "innerparallel",
                    "linalg-copy", true, "eltwise", false);
  b.create<transform::YieldOp>(loc);
  return success();
}

class Aarch64GEMMLargeKScheduleFactory : public ScheduleFactory {
 public:
  using ScheduleFactory::ScheduleFactory;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp) override;
  LogicalResult buildGuardCondition(OpBuilder& b, Location loc,
                                    PatternDescription&, Value&) override;
};

LogicalResult Aarch64GEMMLargeKScheduleFactory::buildGuardCondition(
    OpBuilder& b, Location loc, PatternDescription& pd, Value& pred) {
  auto& fusionPattern = pd.getFusionPattern();
  auto dotOp =
      dyn_cast_or_null<lmhlo::DotGeneralOp>(fusionPattern.getDominantOp());
  if (!dotOp) {
    llvm::dbgs() << "expect dot_general op as dominant\n";
    return failure();
  }
  Value lhs = dotOp->getOperand(0);
  auto lhsTy = lhs.getType().cast<MemRefType>();
  if (lhsTy.getRank() != 2) {
    return dotOp->emitError() << "only support rank 2 GEMM a.t.m.\n";
  }

  auto dimNumbers = dotOp.getDotDimensionNumbers();
  auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
  if (lhsContractingDims.size() != 1) {
    return dotOp->emitError() << "only support exactly 1 contract dim\n";
  }
  bool lhsTranspose = (lhsContractingDims[0] == lhsTy.getRank() - 2);
  Value dimK = b.create<memref::DimOp>(loc, lhs, lhsTranspose ? 0 : 1);
  Value largeK = b.create<arith::ConstantIndexOp>(loc, 768);
  pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, dimK, largeK);
  return success();
}

LogicalResult Aarch64GEMMLargeKScheduleFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto seqOp = b.create<transform_ext::CanonicalizedSequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate, Value{});
  seqOp.getBody().push_back(new Block);
  auto& bodyBlock = seqOp.getBody().front();
  auto pdlOpType = pdl::OperationType::get(ctx);
  bodyBlock.addArgument(pdl::OperationType::get(ctx), loc);
  b.setInsertionPointToStart(&bodyBlock);
  Value variant = bodyBlock.getArgument(0);

  auto& fusionPattern = pd.getFusionPattern();
  auto nameMap = TransformNameAssigner(fusionPattern.getOpList()).getNameMap();
  auto dotOp =
      dyn_cast_or_null<lmhlo::DotGeneralOp>(fusionPattern.getDominantOp());
  if (!dotOp) {
    return m->emitError() << "expect dot_general op as dominant\n";
  }
  Value lhs = dotOp->getOperand(0);
  Value rhs = dotOp->getOperand(1);
  auto lhsTy = lhs.getType().cast<MemRefType>();
  auto rhsTy = rhs.getType().cast<MemRefType>();
  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
    return m->emitError() << "only support rank 2 GEMM a.t.m.\n";
  }

  auto dimNumbers = dotOp.getDotDimensionNumbers();
  auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
  auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
  if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
    return m->emitError() << "only support exactly 1 contract dim\n";
  }
  bool lhsTranspose = (lhsContractingDims[0] == lhsTy.getRank() - 2);
  bool rhsTranspose = (rhsContractingDims[0] == rhsTy.getRank() - 1);
  int64_t M = lhsTranspose ? lhsTy.getShape()[lhsTy.getRank() - 1]
                           : lhsTy.getShape()[lhsTy.getRank() - 2];
  int64_t K = lhsTranspose ? lhsTy.getShape()[lhsTy.getRank() - 2]
                           : lhsTy.getShape()[lhsTy.getRank() - 1];
  int64_t N = rhsTranspose ? rhsTy.getShape()[rhsTy.getRank() - 2]
                           : rhsTy.getShape()[rhsTy.getRank() - 1];

  // build handle to target dot op.
  Value fillAndMatmul = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
  auto matmulSplitOp = buildSplitHandlesOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);

  // transform.structured.tile_to_foreach_thread_op %matmul num_threads [1, 1]
  auto forEachThreadOp = buildTileToForEachThreadOp(b, loc, matmul, {1, 1});
  Value forEachThreadLoop = forEachThreadOp->getResult(0);
  Value tiledMatmul = forEachThreadOp->getResult(1);

  // transform.structured.fuse_into_containing_op %fill into %0#0
  auto fuseIntoContainingOp =
      buildFuseIntoContainingOp(b, loc, fill, forEachThreadLoop);

  // first level tile size for dimension m
  int64_t M0 = 288, M1 = 8;
  // first/second level tile size for dimension n
  int64_t N0 = 204, N1 = 12;
  // first/second level tile size for dimension k
  int64_t K0 = 512, K1 = 1;
  // TODO(wyzero): query cpuinfo.
  int64_t hardwareVectorSizeInBytes = 4;

  auto tileOp0 = buildTileOp(b, loc, tiledMatmul, {M0, N0, K0}, {0, 2, 1});
  auto tileOp1 =
      buildTileOp(b, loc, tileOp0->getResult(0), {M1, N1, K1}, {0, 1, 2});

  // fold extract_slice ops generated by two-level tiling. It's needed to
  // enable following pad and cache_read schedule.
  Value weightInnerSlice =
      buildGetProducerOfOperand(b, loc, tileOp1->getResult(0), 1);
  buildFoldProducerExtractSlice(b, loc, weightInnerSlice, 2);

  // pad to match the requirement of hardware vector/tensor instruction.
  auto padOp = buildPadOp(b, loc, tileOp1->getResult(0), {0, 1, 2});

  Value padForInput = buildGetProducerOfOperand(b, loc, padOp, 0);
  Value padForWeight = buildGetProducerOfOperand(b, loc, padOp, 1);

  // Check if we need to pad dimension `m/n/k` if input or weight is packed
  bool mIsPadded = (M1 != 1) && (M == ShapedType::kDynamicSize ||
                                 (M > M0 && (M % M0 != 0 || M0 % M1 != 0)) ||
                                 (M <= M0 && M > M1 && M % M1 != 0));
  bool nIsPadded = (N1 != 1) && (N == ShapedType::kDynamicSize ||
                                 (N > N0 && (N % N0 != 0 || N0 % N1 != 0)) ||
                                 (N <= N0 && N > N1 && N % N1 != 0));
  bool kIsPadded = (K1 != 1) && (K == ShapedType::kDynamicSize ||
                                 (K > K0 && (K % K0 != 0 || K0 % K1 != 0)) ||
                                 (K <= K0 && K > K1 && K % K1 != 0));

  // Check if we need to pack the input:
  bool packInput = ((M == ShapedType::kDynamicSize || M >= M1) &&
                    (K == ShapedType::kDynamicSize || K > K0) &&
                    (N == ShapedType::kDynamicSize || N > N0));
  // supposed loop order:
  //  loop_m0
  //   loop_k0
  //    loop_n0
  //     loop_m1
  //      loop_n1
  //       loop_k1 {
  //         inner_most_gemm
  //       }
  // in case:
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool m0Skipped = (M != ShapedType::kDynamicSize && M <= M0);
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool k0Skipped = (K != ShapedType::kDynamicSize && K <= K0);
  // - the size of dimension N <= N0, then loop_n0 will be folded.
  bool n0Skipped = (N != ShapedType::kDynamicSize && N <= N0);
  // - the size of dimension M <= M1, then loop_m1 will be folded.
  bool m1Skipped = (M != ShapedType::kDynamicSize && M <= M1);
  // - the size of dimension N <= N1, then loop_n1 will be folded.
  bool n1Skipped = (N != ShapedType::kDynamicSize && N <= N1);
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool k1Skipped = (K != ShapedType::kDynamicSize && K <= K1);
  if (packInput) {
    // We want to cache the packed A below loop_k0 and above loop_n0.
    // Thus the initial loop_level is 4.
    int loopLevel = 4 - n0Skipped - m1Skipped - n1Skipped - k1Skipped;
    if (loopLevel <= 0) {
      return m->emitError()
             << "failed to cache the packed input due to loopLevel = "
             << loopLevel << " is invalid\n";
    }
    auto loopN0 = buildGetParentForOp(b, loc, padForInput, loopLevel);
    bool inputIsPadded = mIsPadded || kIsPadded;
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (lhsTranspose) {
      tileSizes = {K1, M1};
      permutation = {2, 0, 1, 3};
    } else {
      tileSizes = {M1, K1};
      permutation = {0, 2, 3, 1};
    }
    buildCacheRead(b, loc, padForInput, loopN0, {1, 1}, tileSizes,
                   inputIsPadded, permutation);
  }

  // Check if we need to pack the weight, one of the following conditions:
  // - if M, N and K are both dynamic, we always pad input a.t.m.
  // - if N is known and N >= N0 && N0 > N1
  bool packWeight = ((K == ShapedType::kDynamicSize || K > K1) &&
                     (N == ShapedType::kDynamicSize || N > N1));
  if (packWeight) {
    bool weightIsPadded = nIsPadded || kIsPadded;
    forEachThreadLoop = buildMatchOp(b, loc, variant, {"scf.foreach_thread"});
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (rhsTranspose) {
      tileSizes = {N1, K1};
      permutation = {0, 2, 3, 1};
    } else {
      tileSizes = {K1, N1};
      permutation = {2, 0, 1, 3};
    }
    buildCacheRead(b, loc, padForWeight, forEachThreadLoop, {1, 1}, tileSizes,
                   weightIsPadded, permutation);
  }

  variant = buildRunCanonicalizer(b, loc, variant);

  Value multiLevelPackOps =
      buildMatchOp(b, loc, variant, {"disc_linalg_ext.multi_level_pack"});
  buildLowerMultiLevelPackToLoop(b, loc, multiLevelPackOps);

  variant = buildRunCanonicalizer(b, loc, variant);

  Value func = buildMatchOp(b, loc, variant, {"func.func"});
  buildVectorize(b, loc, func, true);

  variant = buildRunCanonicalizer(b, loc, variant);
  variant = buildDISCBufferize(b, loc, variant);

  if (!k0Skipped && !k1Skipped) {
    Value leftFillOp = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
    Value contractOp = buildMatchOp(b, loc, variant, {"vector.contract"});
    int outterMostKLoopLevel =
        5 - k0Skipped - n0Skipped - m1Skipped - n1Skipped - k1Skipped;
    auto loop0 = buildGetParentForOp(b, loc, contractOp, outterMostKLoopLevel);
    auto loop1 = buildGetParentForOp(b, loc, contractOp, 2);
    auto readers = buildMatchOp(b, loc, loop1, {"vector.transfer_read"});
    auto splitedReaders = buildSplitHandlesOp(b, loc, readers, 3);
    buildInlineReductionInitializerOp(b, loc, leftFillOp, loop0,
                                      splitedReaders->getResult(0));
  }

  buildLowerVectors(b, loc, {0, 1, 2, 3, 4}, "outerproduct", "innerparallel",
                    "linalg-copy", true, "eltwise", false);
  buildLowerVectors(b, loc, {5, 6, 7}, "outerproduct", "innerparallel",
                    "linalg-copy", true, "eltwise", false);
  variant = buildDecomposeVectors(b, loc, variant, hardwareVectorSizeInBytes);
  b.create<transform::YieldOp>(loc);
  return success();
}

DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, kDefaultScheduleFactoryPriority,
                        Aarch64GEMMDefaultScheduleFactory,
                        ArrayRef<StringRef>{kDefaultScheduleFactoryTag});

DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, 100,
                        Aarch64GEMMLargeKScheduleFactory,
                        ArrayRef<StringRef>{"large_k"});

}  // namespace

const char* kDefaultScheduleFactoryTag = "default";

std::string patternKindToString(PatternKind kind) {
  auto& map = getPatternKindToStringMap();
  auto it = map.find(kind);
  if (it != map.end()) return it->second;
  llvm_unreachable("unknown pattern kind");
  return "";
}

PatternKind patternKindFromString(const std::string& str) {
  auto& map = getStringToPatternKindMap();
  auto it = map.find(str);
  if (it != map.end()) return it->second;
  llvm_unreachable("unknown pattern kind str");
  return PatternKind::kNone;
}

PatternDescription::PatternDescription(lmhlo::FusionOp op,
                                       FusionPattern& fusionPattern,
                                       ShapeAnalysis& shapeAnalysis)
    : op_(op),
      fusionPattern_(fusionPattern),
      shapeAnalysis_(shapeAnalysis),
      tagSet_(parsefusionTagSetFromStr(getFusionTagStr(op))) {
  // TODO(wyzero): select the pattern kind according to the `fusionPattern`.
  patternKind_ = PatternKind::kGEMM;
}

PatternKind PatternDescription::getPatternKind() const { return patternKind_; }

std::string PatternDescription::getPatternTagStr() const {
  return getFusionTagStr(op_).str();
}

const std::set<std::string>& PatternDescription::getPatternTagSet() const {
  return tagSet_;
}

std::string PatternDescription::getTaggedPatternStr() const {
  return patternKindToString(patternKind_) + "@" + getPatternTagStr();
}

ScheduleFactory::ScheduleFactory(int64_t id, PatternKind kind,
                                 ArrayRef<StringRef> tags)
    : id_(id), kind_(kind) {
  tagSet_.insert(Twine(id).str());
  for (auto tag : tags) {
    tagSet_.insert(tag.str());
  }
}

bool ScheduleFactory::accept(PatternDescription& pattern) {
  return checkKindAndTags(pattern) && checkFusionPatternProperties(pattern);
}

bool ScheduleFactory::checkKindAndTags(PatternDescription& pattern) {
  if (pattern.getPatternKind() != kind_) return false;
  for (auto& tag : pattern.getPatternTagSet())
    if (tagSet_.find(tag) == tagSet_.end()) return false;
  return true;
}

bool ScheduleFactory::checkFusionPatternProperties(PatternDescription&) {
  return true;
}

LogicalResult ScheduleFactory::buildGuardCondition(OpBuilder& b, Location loc,
                                                   PatternDescription&,
                                                   Value&) {
  return failure();
}

LogicalResult ScheduleFactory::assignSchedule(PatternDescription&, ModuleOp) {
  return failure();
}

bool ScheduleFactory::noGuardCondition(PatternDescription&) { return false; }

/* static */ ScheduleFactoryRegistry& ScheduleFactoryRegistry::get() {
  static ScheduleFactoryRegistry instance;
  return instance;
}

/* static */ int64_t ScheduleFactoryRegistry::getNextUniqueId() {
  static std::atomic<int64_t> nextIdx{0};
  return nextIdx++;
}

bool ScheduleFactoryRegistry::registerScheduleFactory(
    PatternKind kind, int priority, ScheduleFactoryPtr factory) {
  return patternMap_[kind].emplace(priority, std::move(factory)).second;
}

void ScheduleFactoryRegistry::unregisterScheduleFactory(PatternKind kind,
                                                        int priority) {
  patternMap_[kind].erase(priority);
}

ScheduleFactory* ScheduleFactoryRegistry::getScheduleFactoryWithHighestPriority(
    PatternDescription& pd) {
  auto& factoryMap = patternMap_[pd.getPatternKind()];
  for (auto it = factoryMap.rbegin(); it != factoryMap.rend(); ++it) {
    if (it->second->accept(pd)) return it->second.get();
  }
  return nullptr;
}

SmallVector<ScheduleFactory*>
ScheduleFactoryRegistry::getAllCandidateScheduleFactories(
    PatternDescription& pd) {
  SmallVector<ScheduleFactory*> factories;
  auto& factoryMap = patternMap_[pd.getPatternKind()];
  for (auto it = factoryMap.rbegin(); it != factoryMap.rend(); ++it) {
    if (it->second->accept(pd)) {
      factories.push_back(it->second.get());
      // early stop
      if (it->second->noGuardCondition(pd)) break;
    }
  }
  return factories;
}

ScheduleDispatcher::ScheduleDispatcher(const std::string& transformFileName)
    : transformFileName_(transformFileName) {}

ScheduleDispatcher::~ScheduleDispatcher() {
  int priority = kParsedFromFileScheduleFactoryStartPriority;
  for (const auto& outter : parsedModuleMap_)
    for (const auto& inner : outter.second)
      ScheduleFactoryRegistry::get().unregisterScheduleFactory(outter.first,
                                                               priority++);
}

LogicalResult ScheduleDispatcher::parseModuleFromFile(MLIRContext* ctx) {
  if (transformFileName_.empty() || !parsedModuleMap_.empty()) return success();
  std::string expectedFormatStr =
      "<pattern-kind-0>:<tag-str-0>:<filename-0>;<pattern-kind-1>:<tag-str-1>:<"
      "filename-1>";
  SmallVector<StringRef> patternSettings;
  StringRef(transformFileName_)
      .split(patternSettings, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  int priority = kParsedFromFileScheduleFactoryStartPriority;
  for (auto& patternSetting : patternSettings) {
    SmallVector<StringRef> items;
    patternSetting.split(items, ":", /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    if (items.size() != 3) {
      llvm::dbgs() << "illegal transform file setting, expected format:  "
                   << expectedFormatStr << "\n";
      return failure();
    }
    PatternKind kind = patternKindFromString(items[0].str());
    if (kind == PatternKind::kNone) {
      llvm::dbgs() << "illegal transform file setting, unknown pattern kind: "
                   << items[0] << "\n";
      return failure();
    }

    auto& transformModule = parsedModuleMap_[kind][items[1].str()];
    if (failed(parseTransformModuleFromFile(ctx, items[2], transformModule))) {
      llvm::dbgs()
          << "illegal transform file setting, unable to load module from: "
          << items[2] << "\n";
      return failure();
    }
    SmallVector<StringRef> tags;
    items[1].split(tags, kFusionTagSeparator, /*MaxSplit*/ -1,
                   /*KeepEmpty*/ false);
    ScheduleFactoryRegistry::get().registerScheduleFactory(
        kind, priority++,
        std::make_unique<ParsedFromFileScheduleFactory>(
            ScheduleFactoryRegistry::get().getNextUniqueId(), kind, tags,
            transformModule.get()));
  }
  return success();
}

LogicalResult ScheduleDispatcher::dispatch(PatternDescription& pd, ModuleOp m) {
  auto factory =
      ScheduleFactoryRegistry::get().getScheduleFactoryWithHighestPriority(pd);
  if (!factory) {
    llvm::dbgs() << "no default schedule for pattern: "
                 << pd.getTaggedPatternStr() << "\n";
    return failure();
  }

  return factory->assignSchedule(pd, m);
}

}  // namespace disc_ral
}  // namespace mlir
