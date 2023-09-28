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

#include "mlir/disc/transforms/disc_transform_schedule.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h"
#include "mlir/disc/tools/disc-transform/transforms/passes.h"
#include "mlir/disc/tools/disc-transform/utils.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"
#include "mlir/disc/transforms/placement_utils.h"
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

std::unordered_map<std::string, DeviceType>& getStringToDeviceTypeMap() {
  static std::unordered_map<std::string, DeviceType> stringToDeviceTypeMap;
  return stringToDeviceTypeMap;
}

bool PatternKindAndStringMapRegistrar = []() {
  auto& patternKindToStringMap = getPatternKindToStringMap();
  auto& stringToPatternKindMap = getStringToPatternKindMap();
  patternKindToStringMap.emplace(PatternKind::kNone, "kNone");
  patternKindToStringMap.emplace(PatternKind::kGEMM, "kGEMM");
  for (auto& pair : patternKindToStringMap) {
    stringToPatternKindMap[pair.second] = pair.first;
  }
  auto& stringToDeviceTypeMap = getStringToDeviceTypeMap();
  stringToDeviceTypeMap.emplace("CPU", DeviceType::kCPU);
  stringToDeviceTypeMap.emplace("GPU", DeviceType::kGPU);
  return true;
}();

using transform::FuseIntoContainingOp;
using transform::FuseOp;
using transform::MatchOp;
using transform::SplitHandleOp;
using transform::TileOp;
using transform::TileToForallOp;
using transform::VectorizeOp;

MatchOp buildMatchOp(OpBuilder& b, Location& loc, Value target,
                     ArrayRef<StringRef> ops, StringRef name = {},
                     DictionaryAttr givenAttrs = nullptr) {
  ArrayAttr opNames;
  if (!ops.empty()) {
    opNames = b.getStrArrayAttr(ops);
  }
  DictionaryAttr attrs = givenAttrs;
  if (!name.empty() && givenAttrs == nullptr) {
    attrs = b.getDictionaryAttr(
        b.getNamedAttr(kDISCLinalgTransformName, b.getStringAttr(name)));
  }

  return b.create<MatchOp>(loc, transform::AnyOpType::get(b.getContext()),
                           target, opNames, transform::MatchInterfaceEnumAttr{},
                           attrs, TypeAttr{});
}

TileToForallOp buildTileToForallOp(OpBuilder& b, Location& loc, Value target,
                                   ArrayRef<int64_t> threads,
                                   transform::NumThreadsSpec numThreadsSpec,
                                   ArrayAttr mapping) {
  return b.create<TileToForallOp>(loc, target, threads, numThreadsSpec,
                                  mapping);
}

TileToForallOp buildTileToForallOp(OpBuilder& b, Location& loc, Value target,
                                   ArrayRef<int64_t> tiles,
                                   transform::TileSizesSpec tileSizesSpec,
                                   ArrayAttr mapping) {
  return b.create<TileToForallOp>(loc, target, tiles, tileSizesSpec, mapping);
}

Value buildFuseIntoContainingOp(OpBuilder& b, Location& loc, Value target,
                                Value anchor) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  SmallVector<Type> resultTypes{transformOpType, transformOpType};
  return b.create<FuseIntoContainingOp>(loc, resultTypes, target, anchor)
      .getFusedOp();
}

FuseOp buildFuseOp(OpBuilder& b, Location& loc, Value target,
                   ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> interchange) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  SmallVector<Type> loopTypes;
  for (int64_t tileSize : tileSizes) {
    if (tileSize) {
      loopTypes.push_back(transformOpType);
    }
  }
  return b.create<FuseOp>(loc, transformOpType, loopTypes, target,
                          b.getI64ArrayAttr(tileSizes),
                          b.getI64ArrayAttr(interchange));
}

TileOp buildTileOp(OpBuilder& b, Location& loc, Value target,
                   ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> interchange) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  SmallVector<Type> loopTypes;
  for (int64_t tileSize : tileSizes) {
    if (tileSize) {
      loopTypes.push_back(transformOpType);
    }
  }
  return b.create<TileOp>(loc, transformOpType, loopTypes, target, ValueRange{},
                          tileSizes, interchange);
}

transform_dialect::ApplyPatternsOp buildRunCanonicalizer(OpBuilder& b,
                                                         Location& loc,
                                                         Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::ApplyPatternsOp>(loc, transformOpType,
                                                      target, true);
}

transform::GetProducerOfOperand buildGetProducerOfOperand(OpBuilder& b,
                                                          Location& loc,
                                                          Value target,
                                                          int64_t operandIdx) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform::GetProducerOfOperand>(loc, transformOpType, target,
                                                   operandIdx);
}

transform_dialect::FoldProducerExtractSliceOp buildFoldProducerExtractSlice(
    OpBuilder& b, Location& loc, Value target, int64_t repeat) {
  return b.create<transform_dialect::FoldProducerExtractSliceOp>(loc, target,
                                                                 repeat);
}

transform::PadOp buildPadOp(OpBuilder& b, Location& loc, Value target,
                            ArrayRef<int64_t> paddingDimensions,
                            int64_t numOperands,
                            ArrayRef<Type> paddingTypes = {}) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  // TODO(wyzero): support other types.
  SmallVector<Attribute> paddingAttrs(numOperands,
                                      b.getZeroAttr(b.getF32Type()));
  for (const auto& [idx, type] : llvm::enumerate(paddingTypes)) {
    paddingAttrs[idx] = b.getZeroAttr(type);
  }
  return b.create<transform::PadOp>(loc, transformOpType, target,
                                    b.getArrayAttr(paddingAttrs),
                                    b.getI64ArrayAttr(paddingDimensions),
                                    ArrayAttr{}, ArrayAttr{}, ArrayAttr{});
}

transform::GetParentForOp buildGetParentForOp(OpBuilder& b, Location& loc,
                                              Value target, int64_t num_loops) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform::GetParentForOp>(loc, transformOpType, target,
                                             num_loops);
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
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<VectorizeOp>(loc, transformOpType, target, vectorizePad);
}

transform_dialect::DISCBufferizeOp buildDISCBufferize(OpBuilder& b,
                                                      Location& loc,
                                                      Value target,
                                                      bool targetGpu) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::DISCBufferizeOp>(loc, transformOpType,
                                                      target, targetGpu);
}

vector::LowerVectorsOptions getDefaultLowerVectorsOptions() {
  vector::LowerVectorsOptions options;
  options.setVectorTransformsOptions(
      vector::VectorContractLowering::OuterProduct);
  options.setVectorMultiReductionLowering(
      vector::VectorMultiReductionLowering::InnerParallel);
  options.setVectorTransposeLowering(vector::VectorTransposeLowering::EltWise);
  options.setVectorTransferSplit(vector::VectorTransferSplit::LinalgCopy);
  options.setTransposeAVX2Lowering(false);
  options.setUnrollVectorTransfers(true);
  return options;
}

transform_dialect::DISCLowerVectorsOp buildLowerVectors(
    OpBuilder& b, Location& loc, Value target,
    const vector::LowerVectorsOptions& options =
        getDefaultLowerVectorsOptions()) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::DISCLowerVectorsOp>(loc, transformOpType,
                                                         target, options);
}

SplitHandleOp buildSplitHandleOp(OpBuilder& b, Location& loc, Value target,
                                 uint64_t num_result_handles) {
  return b.create<SplitHandleOp>(loc, target, num_result_handles);
}

transform_dialect::InlineReductionInitializerOp
buildInlineReductionInitializerOp(OpBuilder& b, Location& loc, Value initOp,
                                  Value loopOp, Value readerOp) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::InlineReductionInitializerOp>(
      loc, transformOpType, initOp, loopOp, readerOp);
}

transform_dialect::DecomposeVectorsOp buildDecomposeVectors(
    OpBuilder& b, Location& loc, Value target, int64_t vectorSize) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::DecomposeVectorsOp>(loc, transformOpType,
                                                         target, vectorSize);
}

transform_dialect::LinalgFuseProducersOp buildLinalgFuseProducersOp(
    OpBuilder& b, Location& loc, Value target, ValueRange producers) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::LinalgFuseProducersOp>(
      loc, transformOpType, target, producers);
}

transform_dialect::ReplaceConstPaddingValueOp buildReplaceConstPaddingValueOp(
    OpBuilder& b, Location& loc, Value target, StringRef mode) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::ReplaceConstPaddingValueOp>(
      loc, transformOpType, target, mode);
}

transform_dialect::ConvertPaddingPlaceholderToConstOp
buildConvertPaddingPlaceholderToConstOp(OpBuilder& b, Location& loc,
                                        Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::ConvertPaddingPlaceholderToConstOp>(
      loc, transformOpType, target);
}

transform_dialect::LinalgEagerlyBackwardInitTensorOp
buildLinalgEagerlyBackwardInitTensorOp(OpBuilder& b, Location& loc,
                                       Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::LinalgEagerlyBackwardInitTensorOp>(
      loc, transformOpType, target);
}

transform_dialect::DISCFuseIntoContainingOp buildDISCFuseIntoContainingOp(
    OpBuilder& b, Location& loc, Value target, Value anchor) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::DISCFuseIntoContainingOp>(
      loc, transformOpType, target, anchor);
}

transform_dialect::ReductionOutputFuseOp buildReductionOutputFuseOp(
    OpBuilder& b, Location& loc, Value target, Value loop) {
  SmallVector<Type> transformOpTypes(2,
                                     transform::AnyOpType::get(b.getContext()));
  return b.create<transform_dialect::ReductionOutputFuseOp>(
      loc, transformOpTypes, target, loop);
}

transform_dialect::ReductionInputFuseOp buildReductionInputFuseOp(OpBuilder& b,
                                                                  Location& loc,
                                                                  Value target,
                                                                  Value loop) {
  SmallVector<Type> transformOpTypes(2,
                                     transform::AnyOpType::get(b.getContext()));
  return b.create<transform_dialect::ReductionInputFuseOp>(
      loc, transformOpTypes, target, loop);
}

transform_dialect::VectorizeConditionalGenericOp
buildVectorizeConditionalGenericOp(OpBuilder& b, Location& loc, Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::VectorizeConditionalGenericOp>(
      loc, transformOpType, target);
}

transform_dialect::SplitVectorTransferIntoFullAndPartialOp
buildSplitVectorTransferIntoFullAndPartialOp(OpBuilder& b, Location& loc,
                                             Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::SplitVectorTransferIntoFullAndPartialOp>(
      loc, transformOpType, target);
}

transform_dialect::LowerConditionalGenericOp buildLowerConditionalGenericOp(
    OpBuilder& b, Location& loc, Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::LowerConditionalGenericOp>(
      loc, transformOpType, target);
}

transform_dialect::ApplyCommonSubexpressionEliminationOp buildCSEOp(
    OpBuilder& b, Location& loc, Value target) {
  return b.create<transform_dialect::ApplyCommonSubexpressionEliminationOp>(
      loc, target);
}

transform_dialect::ApplyDeadCodeEliminationOp buildDCEOp(OpBuilder& b,
                                                         Location& loc,
                                                         Value target) {
  return b.create<transform_dialect::ApplyDeadCodeEliminationOp>(loc, target);
}

transform_dialect::ApplyLoopIndependentCodeMotionOp buildLICMOp(OpBuilder& b,
                                                                Location& loc,
                                                                Value target) {
  return b.create<transform_dialect::ApplyLoopIndependentCodeMotionOp>(loc,
                                                                       target);
}

transform_dialect::DISCPromoteDotOperandsOp buildPromoteDotOperandsOp(
    OpBuilder& b, Location& loc, Value target, ArrayRef<int64_t> indices) {
  SmallVector<Type> transformOpTypes(3,
                                     transform::AnyOpType::get(b.getContext()));
  return b.create<transform_dialect::DISCPromoteDotOperandsOp>(
      loc, transformOpTypes, target, indices);
}

transform_dialect::DISCSplitReductionSerialOp buildSplitReductionSerialOp(
    OpBuilder& b, Location& loc, Value target, ArrayRef<int64_t> tileSizes,
    StringAttr loopType) {
  SmallVector<Type> transformOpTypes(2,
                                     transform::AnyOpType::get(b.getContext()));
  return b.create<transform_dialect::DISCSplitReductionSerialOp>(
      loc, transformOpTypes, target, tileSizes, loopType);
}

transform_dialect::DISCVectorToMMAConversionOp buildVectorToMMAConversionOp(
    OpBuilder& b, Location& loc, Value target) {
  return b.create<transform_dialect::DISCVectorToMMAConversionOp>(loc, target);
}

transform_dialect::DISCForallToGPUCTAsOp buildForallToGPUCTAsOp(OpBuilder& b,
                                                                Location& loc,
                                                                Value target) {
  auto transformOpType = transform::AnyOpType::get(b.getContext());
  return b.create<transform_dialect::DISCForallToGPUCTAsOp>(
      loc, transformOpType, target);
}

transform_dialect::DISCForallToGPUWarpsOp buildForallToGPUWarpsOp(
    OpBuilder& b, Location& loc, Value target) {
  return b.create<transform_dialect::DISCForallToGPUWarpsOp>(loc, target);
}

transform_dialect::DISCLowerGmemToSmemOp buildLowerGmemToSmemOp(OpBuilder& b,
                                                                Location& loc,
                                                                Value target) {
  return b.create<transform_dialect::DISCLowerGmemToSmemOp>(loc, target);
}

transform_dialect::DISCTransferWriteZeroToSCFOp buildTransferWriteZeroToSCFOp(
    OpBuilder& b, Location& loc, Value target) {
  return b.create<transform_dialect::DISCTransferWriteZeroToSCFOp>(loc, target);
}

transform_dialect::DISCEraseDeallocOp buildEraseDeallocOp(OpBuilder& b,
                                                          Location& loc,
                                                          Value target) {
  return b.create<transform_dialect::DISCEraseDeallocOp>(loc, target);
}

transform_dialect::DISCInlineAndConvertGPUIdsOp buildInlineAndConvertGPUIdsOp(
    OpBuilder& b, Location& loc, Value target) {
  return b.create<transform_dialect::DISCInlineAndConvertGPUIdsOp>(loc, target);
}

class ParsedFromFileScheduleFactory : public ScheduleFactoryWithNoGuard {
 public:
  explicit ParsedFromFileScheduleFactory(int64_t id, PatternKind kind,
                                         ArrayRef<StringRef> tags,
                                         DeviceType deviceType,
                                         ModuleOp transformModule);
  LogicalResult assignSchedule(PatternDescription&, ModuleOp,
                               DeviceInfo) override;

 private:
  ModuleOp transformModule_;
};

ParsedFromFileScheduleFactory::ParsedFromFileScheduleFactory(
    int64_t id, PatternKind kind, ArrayRef<StringRef> tags,
    DeviceType deviceType, ModuleOp transformModule)
    : ScheduleFactoryWithNoGuard(id, kind, tags, deviceType),
      transformModule_(transformModule) {}

LogicalResult ParsedFromFileScheduleFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m, DeviceInfo deviceInfo) {
  OpBuilder b(m);
  for (auto& op : transformModule_.getBody()->getOperations()) {
    if (!isa<transform::TransformOpInterface>(&op)) continue;
    m.push_back(b.clone(op));
  }
  return success();
}

// The following AArch64 codegen schedules are buggy and temporarily disabled.
// The problem of these schedules is that many customized Transform-dialect ops
// for these schedules rely on passes like Canonicalization, LICM, DCE, and CSE
// to generate the correct code, which is not the prefered way of IR building
// and lowering.
#define ENABLE_AARCH64_SCHEDUELS 0
#if ENABLE_AARCH64_SCHEDUELS
class Aarch64GEMMDefaultScheduleFactory : public ScheduleFactoryWithNoGuard {
 public:
  using ScheduleFactoryWithNoGuard::ScheduleFactoryWithNoGuard;
  bool checkFusionPatternProperties(PatternDescription&) override;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp,
                               DeviceInfo) override;
};

// TODO(wyzero): merge default schedule and default with epilogue schedule.
bool Aarch64GEMMDefaultScheduleFactory::checkFusionPatternProperties(
    PatternDescription& pd) {
  if (!ScheduleFactory::checkFusionPatternProperties(pd)) {
    return false;
  }
  auto& fusionPattern = pd.getFusionPattern();
  auto& rootOps = fusionPattern.getRootOps();
  // Only support single output a.t.m.
  if (rootOps.size() != 1) {
    return false;
  }

  // This schedule not support epilogue fusion
  auto dominantOp = fusionPattern.getDominantOp();
  return rootOps[0] == dominantOp && isa<lmhlo::DotGeneralOp>(dominantOp);
}

LogicalResult Aarch64GEMMDefaultScheduleFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m, DeviceInfo deviceInfo) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto transformOpType = transform::AnyOpType::get(ctx);
  auto seqOp = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate,
      transformOpType, [&](OpBuilder& b, Location loc, Value variantH) {});
  auto& bodyBlock = seqOp.getBody().front();
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
  auto matmulSplitOp = buildSplitHandleOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);

  // transform.structured.tile_to_forall_op %matmul num_threads [1, 1]
  auto forallOp = buildTileToForallOp(b, loc, matmul, {1, 1},
                                      transform::NumThreadsSpec(), ArrayAttr{});
  Value forallLoop = forallOp->getResult(0);
  Value tiledMatmul = forallOp->getResult(1);

  // transform.structured.fuse_into_containing_op %fill into %0#0
  auto fuseIntoContainingOp =
      buildFuseIntoContainingOp(b, loc, fill, forallLoop);

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
  auto padOp = buildPadOp(b, loc, tileOp->getResult(0), {0, 1, 2}, 3);

  Value padForInput = buildGetProducerOfOperand(b, loc, padOp, 0);
  Value padForWeight = buildGetProducerOfOperand(b, loc, padOp, 1);

  // Check if we need to pad dimension `m/n/k` if input or weight is packed
  bool mIsPadded = (M1 != 1) && (M == ShapedType::kDynamic ||
                                 (M > M0 && (M % M0 != 0 || M0 % M1 != 0)) ||
                                 (M <= M0 && M > M1 && M % M1 != 0));
  bool nIsPadded = (N1 != 1) && (N == ShapedType::kDynamic ||
                                 (N > N0 && (N % N0 != 0 || N0 % N1 != 0)) ||
                                 (N <= N0 && N > N1 && N % N1 != 0));
  bool kIsPadded =
      (K0 != 1) && (K == ShapedType::kDynamic || K > K0 && K % K0 != 0);

  // Check if we need to pack the input:
  bool packInput = ((M == ShapedType::kDynamic || M >= M1) &&
                    (K == ShapedType::kDynamic || K > K0) &&
                    (N == ShapedType::kDynamic || N > N0));
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
    loopLevel -= (N != ShapedType::kDynamic && N <= N0);
    // - the size of dimension M <= M1, then loop_m1 will be folded.
    loopLevel -= (M != ShapedType::kDynamic && M <= M1);
    // - the size of dimension N <= N1, then loop_n1 will be folded.
    loopLevel -= (N != ShapedType::kDynamic && N <= N1);
    // - the size of dimension K <= K0, then loop_k0 will be folded.
    loopLevel -= (K != ShapedType::kDynamic && K <= K0);

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
  bool packWeight = ((K == ShapedType::kDynamic || K > K0) &&
                     (N == ShapedType::kDynamic || N > N1));
  if (packWeight) {
    bool weightIsPadded = nIsPadded || kIsPadded;
    forallLoop = buildMatchOp(b, loc, variant, {"scf.forall"});
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (rhsTranspose) {
      tileSizes = {N1, K0};
      permutation = {0, 2, 3, 1};
    } else {
      tileSizes = {K0, N1};
      permutation = {2, 0, 1, 3};
    }
    buildCacheRead(b, loc, padForWeight, forallLoop, {1, 1}, tileSizes,
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
  variant = buildDISCBufferize(b, loc, variant, false);

  variant = buildLowerVectors(b, loc, variant);
  b.create<transform::YieldOp>(loc);
  return success();
}

class Aarch64GEMMDefaultScheduleWithEpilogueFactory
    : public ScheduleFactoryWithNoGuard {
 public:
  using ScheduleFactoryWithNoGuard::ScheduleFactoryWithNoGuard;
  bool checkFusionPatternProperties(PatternDescription&) override;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp,
                               DeviceInfo) override;
};

bool Aarch64GEMMDefaultScheduleWithEpilogueFactory::
    checkFusionPatternProperties(PatternDescription& pd) {
  if (!ScheduleFactory::checkFusionPatternProperties(pd)) {
    return false;
  }
  auto& fusionPattern = pd.getFusionPattern();
  auto& rootOps = fusionPattern.getRootOps();
  // Only support single output a.t.m.
  if (rootOps.size() != 1) return false;

  // This schedule only works for gemm with epilogue fusion
  auto dominantOp = fusionPattern.getDominantOp();
  return rootOps[0] != dominantOp && isa<lmhlo::DotGeneralOp>(dominantOp);
}

LogicalResult Aarch64GEMMDefaultScheduleWithEpilogueFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m, DeviceInfo deviceInfo) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto transformOpType = transform::AnyOpType::get(ctx);
  auto seqOp = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate,
      transformOpType, [&](OpBuilder& b, Location loc, Value variantH) {});
  auto& bodyBlock = seqOp.getBody().front();
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

  if (fusionPattern.getRootOps().size() != 1) {
    return m->emitError() << "only support single output a.t.m.\n";
  }
  Operation* rootOp = fusionPattern.getRootOps()[0];
  Value rootHandle = buildMatchOp(b, loc, variant, {}, nameMap[rootOp]);

  // merge elemwise ops in case there are many.
  SmallVector<Value> otherElemOpHandles;
  for (Operation* op : fusionPattern.getOpList()) {
    if (op == rootOp || op == dotOp.getOperation()) continue;
    otherElemOpHandles.push_back(
        buildMatchOp(b, loc, variant, {}, nameMap[op]));
  }
  if (!otherElemOpHandles.empty()) {
    buildLinalgFuseProducersOp(b, loc, rootHandle, otherElemOpHandles);
    variant = buildRunCanonicalizer(b, loc, variant);
    rootHandle = buildMatchOp(b, loc, variant, {}, nameMap[rootOp]);
  }

  auto forallOp = buildTileToForallOp(b, loc, rootHandle, {1, 1},
                                      transform::NumThreadsSpec(), ArrayAttr{});
  Value forallLoop = forallOp->getResult(0);
  rootHandle = forallOp->getResult(1);

  Value fillAndMatmul = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
  auto matmulSplitOp = buildSplitHandleOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);
  matmul = buildFuseIntoContainingOp(b, loc, matmul, forallLoop);
  fill = buildFuseIntoContainingOp(b, loc, fill, forallLoop);

  // first/second level tile size for dimension m
  int64_t M0 = 288, M1 = 6;
  // first/second level tile size for dimension n
  int64_t N0 = 48, N1 = 16;
  // first level tile size for dimension k
  int64_t K0 = 1;
  // TODO(wyzero): query cpuinfo.
  int64_t hardwareVectorSizeInBytes = 4;

  // supposed loop order:
  //  loop_m0
  //   loop_n0
  //    loop_m1
  //     loop_n1
  //      loop_k0 {
  //        inner_most_gemm
  //      }
  bool m0Skipped = (M != ShapedType::kDynamic && M <= M0);
  bool n0Skipped = (N != ShapedType::kDynamic && N <= N0);
  bool m1Skipped = (M != ShapedType::kDynamic && M <= M1);
  bool n1Skipped = (N != ShapedType::kDynamic && N <= N1);
  bool k0Skipped = (K != ShapedType::kDynamic && K <= K0);

  // for very small m or n
  if (m1Skipped || n1Skipped) {
    // TODO(wyzero): finetune the schedule for small m or n
    buildTileOp(b, loc, matmul, {1, 1, 1}, {0, 2, 1});
    variant = buildRunCanonicalizer(b, loc, variant);
    variant = buildDISCBufferize(b, loc, variant, false);
    b.create<transform::YieldOp>(loc);
    return success();
  }

  // first level tile and fuse matmul and fill op.
  if (!m0Skipped) {
    auto tileOp1 = buildTileOp(b, loc, rootHandle, {M0, N0}, {0, 1});
    rootHandle = tileOp1->getResult(0);
    matmul = buildFuseIntoContainingOp(b, loc, matmul, tileOp1->getResult(1));
    fill = buildFuseIntoContainingOp(b, loc, fill, tileOp1->getResult(1));
  } else if (!n0Skipped) {
    auto tileOp1 = buildTileOp(b, loc, rootHandle, {M0, N0}, {0, 1});
    rootHandle = tileOp1->getResult(0);
    matmul = buildFuseIntoContainingOp(b, loc, matmul, tileOp1->getResult(2));
    fill = buildFuseIntoContainingOp(b, loc, fill, tileOp1->getResult(2));
  }

  // second level tile and fuse matmul and fill op.
  auto tileOp2 = buildTileOp(b, loc, rootHandle, {M1, N1}, {0, 1});
  // pad root ops to register level size (static size).
  int64_t numOperandsUpperBound = fusionPattern.getOperands().size() +
                                  fusionPattern.getResults().size() +
                                  fusionPattern.getInternalResults().size() + 1;
  auto padRootOp =
      buildPadOp(b, loc, tileOp2->getResult(0), {0, 1}, numOperandsUpperBound);
  auto padOps = buildMatchOp(b, loc, variant, {"tensor.pad"}, {});
  buildReplaceConstPaddingValueOp(b, loc, padOps, "kAny");
  matmul = buildFuseIntoContainingOp(b, loc, matmul, tileOp2->getResult(1));
  fill = buildFuseIntoContainingOp(b, loc, fill, tileOp2->getResult(1));

  // gemm reduction axis tiling
  auto tileOp3 = buildTileOp(b, loc, matmul, {0, 0, K0}, {0, 1, 2});

  variant = buildRunCanonicalizer(b, loc, variant);

  // fold two extract_slice ops generated by two-level tiling. It's needed to
  // enable following pad and hosit schedule.
  Value weightInnerSlice =
      buildGetProducerOfOperand(b, loc, tileOp3->getResult(0), 1);
  buildFoldProducerExtractSlice(b, loc, weightInnerSlice, 2);

  // pad to match the requirement of hardware vector/tensor instruction.
  auto padGEMMOp = buildPadOp(b, loc, tileOp3->getResult(0), {0, 1, 2}, 3);

  Value padForInput = buildGetProducerOfOperand(b, loc, padGEMMOp, 0);
  Value padForWeight = buildGetProducerOfOperand(b, loc, padGEMMOp, 1);

  // Check if we need to pad dimension `m/n/k` if input or weight is packed
  bool mIsPadded = (M1 != 1) && (M == ShapedType::kDynamic ||
                                 (M > M0 && (M % M0 != 0 || M0 % M1 != 0)) ||
                                 (M <= M0 && M > M1 && M % M1 != 0));
  bool nIsPadded = (N1 != 1) && (N == ShapedType::kDynamic ||
                                 (N > N0 && (N % N0 != 0 || N0 % N1 != 0)) ||
                                 (N <= N0 && N > N1 && N % N1 != 0));
  bool kIsPadded =
      (K0 != 1) && (K == ShapedType::kDynamic || K > K0 && K % K0 != 0);

  // Check if we need to pack the input:
  bool packInput = ((M == ShapedType::kDynamic || M >= M1) &&
                    (K == ShapedType::kDynamic || K > K0) &&
                    (N == ShapedType::kDynamic || N > N0));
  if (packInput) {
    // We want to cache the packed A below loop_m0 and above loop_n0.
    // Thus the initial loop_level is 4.
    int loopLevel = 4 - n0Skipped - m1Skipped - n1Skipped - k0Skipped;
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
  bool packWeight = ((K == ShapedType::kDynamic || K > K0) &&
                     (N == ShapedType::kDynamic || N > N1));
  if (packWeight) {
    bool weightIsPadded = nIsPadded || kIsPadded;
    forallLoop = buildMatchOp(b, loc, variant, {"scf.forall"});
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (rhsTranspose) {
      tileSizes = {N1, K0};
      permutation = {0, 2, 3, 1};
    } else {
      tileSizes = {K0, N1};
      permutation = {2, 0, 1, 3};
    }
    buildCacheRead(b, loc, padForWeight, forallLoop, {1, 1}, tileSizes,
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
  auto placeholderOps = buildMatchOp(
      b, loc, variant, {"disc_linalg_ext.padding_value_placeholder"}, {});
  buildConvertPaddingPlaceholderToConstOp(b, loc, placeholderOps);
  variant = buildDISCBufferize(b, loc, variant, false);

  variant = buildLowerVectors(b, loc, variant);
  // de-compose large size vector operations
  variant = buildDecomposeVectors(b, loc, variant, hardwareVectorSizeInBytes);
  b.create<transform::YieldOp>(loc);
  return success();
}

class Aarch64GEMMLargeKScheduleFactory : public ScheduleFactory {
 public:
  using ScheduleFactory::ScheduleFactory;
  bool checkFusionPatternProperties(PatternDescription&) override;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp,
                               DeviceInfo) override;
  LogicalResult buildGuardCondition(OpBuilder& b, Location loc,
                                    PatternDescription&, Value&) override;
};

bool Aarch64GEMMLargeKScheduleFactory::checkFusionPatternProperties(
    PatternDescription& pd) {
  if (!ScheduleFactory::checkFusionPatternProperties(pd)) {
    return false;
  }
  auto& fusionPattern = pd.getFusionPattern();
  auto& rootOps = fusionPattern.getRootOps();
  // Only support single output a.t.m.
  if (rootOps.size() != 1) return false;

  // This schedule not support epilogue fusion
  auto dominantOp = fusionPattern.getDominantOp();
  return rootOps[0] == dominantOp && isa<lmhlo::DotGeneralOp>(dominantOp);
}

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
    PatternDescription& pd, ModuleOp m, DeviceInfo deviceInfo) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto transformOpType = transform::AnyOpType::get(ctx);
  auto seqOp = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate,
      transformOpType, [&](OpBuilder& b, Location loc, Value variantH) {});
  auto& bodyBlock = seqOp.getBody().front();
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
  auto matmulSplitOp = buildSplitHandleOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);

  // transform.structured.tile_to_forall_op %matmul num_threads [1, 1]
  auto forallOp = buildTileToForallOp(b, loc, matmul, {1, 1},
                                      transform::NumThreadsSpec(), ArrayAttr{});
  Value forallLoop = forallOp->getResult(0);
  Value tiledMatmul = forallOp->getResult(1);

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // transform.structured.fuse_into_containing_op %fill into %0#0
  auto fuseIntoContainingOp =
      buildFuseIntoContainingOp(b, loc, fill, forallLoop);

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

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // fold extract_slice ops generated by two-level tiling. It's needed to
  // enable following pad and cache_read schedule.
  Value weightInnerSlice =
      buildGetProducerOfOperand(b, loc, tileOp1->getResult(0), 1);
  buildFoldProducerExtractSlice(b, loc, weightInnerSlice, 2);

  // pad to match the requirement of hardware vector/tensor instruction.
  auto padOp = buildPadOp(b, loc, tileOp1->getResult(0), {0, 1, 2}, 3);

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  Value padForInput = buildGetProducerOfOperand(b, loc, padOp, 0);
  Value padForWeight = buildGetProducerOfOperand(b, loc, padOp, 1);

  // Check if we need to pad dimension `m/n/k` if input or weight is packed
  bool mIsPadded = (M1 != 1) && (M == ShapedType::kDynamic ||
                                 (M > M0 && (M % M0 != 0 || M0 % M1 != 0)) ||
                                 (M <= M0 && M > M1 && M % M1 != 0));
  bool nIsPadded = (N1 != 1) && (N == ShapedType::kDynamic ||
                                 (N > N0 && (N % N0 != 0 || N0 % N1 != 0)) ||
                                 (N <= N0 && N > N1 && N % N1 != 0));
  bool kIsPadded = (K1 != 1) && (K == ShapedType::kDynamic ||
                                 (K > K0 && (K % K0 != 0 || K0 % K1 != 0)) ||
                                 (K <= K0 && K > K1 && K % K1 != 0));

  // Check if we need to pack the input:
  bool packInput = ((M == ShapedType::kDynamic || M >= M1) &&
                    (K == ShapedType::kDynamic || K > K0) &&
                    (N == ShapedType::kDynamic || N > N0));
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
  bool m0Skipped = (M != ShapedType::kDynamic && M <= M0);
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool k0Skipped = (K != ShapedType::kDynamic && K <= K0);
  // - the size of dimension N <= N0, then loop_n0 will be folded.
  bool n0Skipped = (N != ShapedType::kDynamic && N <= N0);
  // - the size of dimension M <= M1, then loop_m1 will be folded.
  bool m1Skipped = (M != ShapedType::kDynamic && M <= M1);
  // - the size of dimension N <= N1, then loop_n1 will be folded.
  bool n1Skipped = (N != ShapedType::kDynamic && N <= N1);
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool k1Skipped = (K != ShapedType::kDynamic && K <= K1);
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

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // Check if we need to pack the weight, one of the following conditions:
  // - if M, N and K are both dynamic, we always pad input a.t.m.
  // - if N is known and N >= N0 && N0 > N1
  bool packWeight = ((K == ShapedType::kDynamic || K > K1) &&
                     (N == ShapedType::kDynamic || N > N1));
  if (packWeight) {
    bool weightIsPadded = nIsPadded || kIsPadded;
    forallLoop = buildMatchOp(b, loc, variant, {"scf.forall"});
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (rhsTranspose) {
      tileSizes = {N1, K1};
      permutation = {0, 2, 3, 1};
    } else {
      tileSizes = {K1, N1};
      permutation = {2, 0, 1, 3};
    }
    buildCacheRead(b, loc, padForWeight, forallLoop, {1, 1}, tileSizes,
                   weightIsPadded, permutation);
  }

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  variant = buildRunCanonicalizer(b, loc, variant);

  Value multiLevelPackOps =
      buildMatchOp(b, loc, variant, {"disc_linalg_ext.multi_level_pack"});
  buildLowerMultiLevelPackToLoop(b, loc, multiLevelPackOps);

  variant = buildRunCanonicalizer(b, loc, variant);

  Value func = buildMatchOp(b, loc, variant, {"func.func"});
  buildVectorize(b, loc, func, true);

  variant = buildRunCanonicalizer(b, loc, variant);
  variant = buildDISCBufferize(b, loc, variant, false);

  if (!k0Skipped && !k1Skipped) {
    Value leftFillOp = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
    Value contractOp = buildMatchOp(b, loc, variant, {"vector.contract"});
    int outterMostKLoopLevel =
        5 - k0Skipped - n0Skipped - m1Skipped - n1Skipped - k1Skipped;
    auto loop0 = buildGetParentForOp(b, loc, contractOp, outterMostKLoopLevel);
    auto loop1 = buildGetParentForOp(b, loc, contractOp, 2);
    auto readers = buildMatchOp(b, loc, loop1, {"vector.transfer_read"});
    auto splitedReaders = buildSplitHandleOp(b, loc, readers, 3);
    buildInlineReductionInitializerOp(b, loc, leftFillOp, loop0,
                                      splitedReaders->getResult(0));
  }

  variant = buildLowerVectors(b, loc, variant);
  variant = buildDecomposeVectors(b, loc, variant, hardwareVectorSizeInBytes);
  b.create<transform::YieldOp>(loc);
  return success();
}

class Aarch64GEMMLargeKScheduleWithEpilogueFactory
    : public Aarch64GEMMLargeKScheduleFactory {
 public:
  using Aarch64GEMMLargeKScheduleFactory::Aarch64GEMMLargeKScheduleFactory;
  bool checkFusionPatternProperties(PatternDescription&) override;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp,
                               DeviceInfo) override;
};

bool Aarch64GEMMLargeKScheduleWithEpilogueFactory::checkFusionPatternProperties(
    PatternDescription& pd) {
  if (!ScheduleFactory::checkFusionPatternProperties(pd)) {
    return false;
  }
  auto& fusionPattern = pd.getFusionPattern();
  auto& rootOps = fusionPattern.getRootOps();
  // Only support single output a.t.m.
  if (rootOps.size() != 1) return false;

  auto dominantOp = fusionPattern.getDominantOp();
  return rootOps[0] != dominantOp && isa<lmhlo::DotGeneralOp>(dominantOp);
}

LogicalResult Aarch64GEMMLargeKScheduleWithEpilogueFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m, DeviceInfo deviceInfo) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto transformOpType = transform::AnyOpType::get(ctx);
  auto seqOp = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate,
      transformOpType, [&](OpBuilder& b, Location loc, Value variantH) {});
  auto& bodyBlock = seqOp.getBody().front();
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

  if (fusionPattern.getRootOps().size() != 1) {
    return m->emitError() << "only support single output a.t.m.\n";
  }
  Operation* rootOp = fusionPattern.getRootOps()[0];
  Value rootHandle = buildMatchOp(b, loc, variant, {}, nameMap[rootOp]);

  // merge elemwise ops in case there are many.
  SmallVector<Value> otherElemOpHandles;
  for (Operation* op : fusionPattern.getOpList()) {
    if (op == rootOp || op == dotOp.getOperation()) continue;
    otherElemOpHandles.push_back(
        buildMatchOp(b, loc, variant, {}, nameMap[op]));
  }
  if (!otherElemOpHandles.empty()) {
    buildLinalgFuseProducersOp(b, loc, rootHandle, otherElemOpHandles);
    variant = buildRunCanonicalizer(b, loc, variant);
    rootHandle = buildMatchOp(b, loc, variant, {}, nameMap[rootOp]);
  }
  rootHandle = buildLinalgEagerlyBackwardInitTensorOp(b, loc, rootHandle);

  auto forallOp = buildTileToForallOp(b, loc, rootHandle, {1, 1},
                                      transform::NumThreadsSpec(), ArrayAttr{});
  Value forallLoop = forallOp->getResult(0);
  rootHandle = forallOp->getResult(1);

  // build handle to target dot op.
  Value fillAndMatmul = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
  auto matmulSplitOp = buildSplitHandleOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);
  matmul = buildFuseIntoContainingOp(b, loc, matmul, forallLoop);
  fill = buildFuseIntoContainingOp(b, loc, fill, forallLoop);

  // first level tile size for dimension m
  int64_t M0 = 288, M1 = 8;
  // first/second level tile size for dimension n
  int64_t N0 = 204, N1 = 12;
  // first/second level tile size for dimension k
  int64_t K0 = 512, K1 = 1;
  // TODO(wyzero): query cpuinfo.
  int64_t hardwareVectorSizeInBytes = 4;

  // Check if we need to pad dimension `m/n/k` if input or weight is packed
  bool mIsPadded = (M1 != 1) && (M == ShapedType::kDynamic ||
                                 (M > M0 && (M % M0 != 0 || M0 % M1 != 0)) ||
                                 (M <= M0 && M > M1 && M % M1 != 0));
  bool nIsPadded = (N1 != 1) && (N == ShapedType::kDynamic ||
                                 (N > N0 && (N % N0 != 0 || N0 % N1 != 0)) ||
                                 (N <= N0 && N > N1 && N % N1 != 0));
  bool kIsPadded = (K1 != 1) && (K == ShapedType::kDynamic ||
                                 (K > K0 && (K % K0 != 0 || K0 % K1 != 0)) ||
                                 (K <= K0 && K > K1 && K % K1 != 0));

  // Check if we need to pack the input:
  bool packInput = ((M == ShapedType::kDynamic || M >= M1) &&
                    (K == ShapedType::kDynamic || K > K0) &&
                    (N == ShapedType::kDynamic || N > N0));
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
  bool m0Skipped = (M != ShapedType::kDynamic && M <= M0);
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool k0Skipped = (K != ShapedType::kDynamic && K <= K0);
  // - the size of dimension N <= N0, then loop_n0 will be folded.
  bool n0Skipped = (N != ShapedType::kDynamic && N <= N0);
  // - the size of dimension M <= M1, then loop_m1 will be folded.
  bool m1Skipped = (M != ShapedType::kDynamic && M <= M1);
  // - the size of dimension N <= N1, then loop_n1 will be folded.
  bool n1Skipped = (N != ShapedType::kDynamic && N <= N1);
  // - the size of dimension K <= K0, then loop_k0 will be folded.
  bool k1Skipped = (K != ShapedType::kDynamic && K <= K1);

  // for very small m and n
  if (m1Skipped || n1Skipped || k1Skipped) {
    buildTileOp(b, loc, matmul, {1, 1, 1}, {0, 2, 1});
    variant = buildRunCanonicalizer(b, loc, variant);
    variant = buildDISCBufferize(b, loc, variant, false);
    b.create<transform::YieldOp>(loc);
    return success();
  }

  if (!m0Skipped) {
    auto tileM0 = buildTileOp(b, loc, rootHandle, {M0}, {0, 1});
    rootHandle = tileM0->getResult(0);
    Value loopM0 = tileM0->getResult(1);
    variant = buildRunCanonicalizer(b, loc, variant);
    matmul = buildDISCFuseIntoContainingOp(b, loc, matmul, loopM0);
    fill = buildDISCFuseIntoContainingOp(b, loc, fill, loopM0);
  }

  if (!k0Skipped) {
    auto tileK0 = buildTileOp(b, loc, matmul, {0, 0, K0}, {0, 1, 2});
    matmul = tileK0->getResult(0);
    auto outputFuseOp =
        buildReductionOutputFuseOp(b, loc, rootHandle, tileK0->getResult(1));
    rootHandle = outputFuseOp->getResult(0);
    auto inputFuseOp =
        buildReductionInputFuseOp(b, loc, fill, outputFuseOp->getResult(1));
    fill = inputFuseOp->getResult(0);
    variant = buildRunCanonicalizer(b, loc, variant);
  }

  if (!n0Skipped) {
    auto tileN0 = buildTileOp(b, loc, rootHandle, {0, N0}, {0, 1});
    rootHandle = tileN0->getResult(0);
    Value loopN0 = tileN0->getResult(1);
    variant = buildRunCanonicalizer(b, loc, variant);
    matmul = buildDISCFuseIntoContainingOp(b, loc, matmul, loopN0);
    fill = buildDISCFuseIntoContainingOp(b, loc, fill, loopN0);
  }

  auto tileM1 = buildTileOp(b, loc, matmul, {M1, 0}, {0, 1});
  matmul = tileM1->getResult(0);
  Value loopM1 = tileM1->getResult(1);
  variant = buildRunCanonicalizer(b, loc, variant);
  fill = buildDISCFuseIntoContainingOp(b, loc, fill, loopM1);

  auto tileN1 = buildTileOp(b, loc, matmul, {0, N1}, {0, 1});
  matmul = tileN1->getResult(0);
  Value loopN1 = tileN1->getResult(1);
  variant = buildRunCanonicalizer(b, loc, variant);
  fill = buildDISCFuseIntoContainingOp(b, loc, fill, loopN1);

  auto tileK1 = buildTileOp(b, loc, matmul, {0, 0, K1}, {0, 1, 2});
  matmul = tileK1->getResult(0);
  variant = buildRunCanonicalizer(b, loc, variant);
  Value weightSlice = buildGetProducerOfOperand(b, loc, matmul, 1);
  buildFoldProducerExtractSlice(b, loc, weightSlice, 2);
  // pad to match the requirement of hardware vector/tensor instruction.
  matmul = buildPadOp(b, loc, matmul, {0, 1, 2}, 3);
  SmallVector<Type> paddingTypes;
  if (!k0Skipped) paddingTypes.push_back(b.getIntegerType(1));
  fill = buildPadOp(b, loc, fill, {0, 1}, 3, paddingTypes);
  Value padForInput = buildGetProducerOfOperand(b, loc, matmul, 0);
  Value padForWeight = buildGetProducerOfOperand(b, loc, matmul, 1);

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
  bool packWeight = ((K == ShapedType::kDynamic || K > K1) &&
                     (N == ShapedType::kDynamic || N > N1));
  if (packWeight) {
    bool weightIsPadded = nIsPadded || kIsPadded;
    forallLoop = buildMatchOp(b, loc, variant, {"scf.forall"});
    SmallVector<int64_t> tileSizes;
    SmallVector<int64_t> permutation;
    if (rhsTranspose) {
      tileSizes = {N1, K1};
      permutation = {0, 2, 3, 1};
    } else {
      tileSizes = {K1, N1};
      permutation = {2, 0, 1, 3};
    }
    buildCacheRead(b, loc, padForWeight, forallLoop, {1, 1}, tileSizes,
                   weightIsPadded, permutation);
  }

  variant = buildRunCanonicalizer(b, loc, variant);
  Value multiLevelPackOps =
      buildMatchOp(b, loc, variant, {"disc_linalg_ext.multi_level_pack"});
  buildLowerMultiLevelPackToLoop(b, loc, multiLevelPackOps);
  variant = buildRunCanonicalizer(b, loc, variant);

  Value fillConditionalOps = buildMatchOp(
      b, loc, variant, {"disc_linalg_ext.conditional_generic"}, nameMap[dotOp]);
  buildVectorizeConditionalGenericOp(b, loc, fillConditionalOps);
  variant = buildRunCanonicalizer(b, loc, variant);
  Value func = buildMatchOp(b, loc, variant, {"func.func"});
  buildVectorize(b, loc, func, true);
  variant = buildRunCanonicalizer(b, loc, variant);

  variant = buildDISCBufferize(b, loc, variant, false);
  variant = buildRunCanonicalizer(b, loc, variant);

  Value conditionalOps =
      buildMatchOp(b, loc, variant, {"disc_linalg_ext.conditional_generic"});
  buildLowerConditionalGenericOp(b, loc, conditionalOps);

  variant = buildLowerVectors(b, loc, variant);
  variant = buildDecomposeVectors(b, loc, variant, hardwareVectorSizeInBytes);
  variant = buildRunCanonicalizer(b, loc, variant);
  variant = buildLowerVectors(b, loc, variant);
  b.create<transform::YieldOp>(loc);
  return success();
}
#endif  // ENABLE_AARCH64_SCHEDUELS

class CUDAMMAGEMMDefaultScheduleFactory : public ScheduleFactoryWithNoGuard {
 public:
  using ScheduleFactoryWithNoGuard::ScheduleFactoryWithNoGuard;
  bool checkFusionPatternProperties(PatternDescription&) override;
  LogicalResult assignSchedule(PatternDescription&, ModuleOp,
                               DeviceInfo) override;
};

bool CUDAMMAGEMMDefaultScheduleFactory::checkFusionPatternProperties(
    PatternDescription& pd) {
  if (!ScheduleFactory::checkFusionPatternProperties(pd)) {
    return false;
  }
  auto& fusionPattern = pd.getFusionPattern();
  auto& rootOps = fusionPattern.getRootOps();
  // Only support single output a.t.m.
  if (rootOps.size() != 1) {
    return false;
  }

  // This schedule not support epilogue fusion
  auto dominantOp = fusionPattern.getDominantOp();
  return (rootOps[0] == dominantOp) && isa<lmhlo::DotGeneralOp>(dominantOp);
}

// The schedule structure of TensorCore GEMM:
//
// parallel (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {
//   parallel (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {
//     for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {
//       // All the following loops should be fully unrolled.
//       for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {
//         for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {
//           for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {
//             for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {
//               for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {
//                 for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {
//                   vector.contract(...) // lowered to MMA intrinsic.
//                 }   // for mma_m
//               }   // for mma_n
//             }   // for mma_k
//           }   // for warp_k
//         }   // for warp_m
//       }   // for warp_n
//     }   // for cta_k
//   }   // parallel cta_m
// }   // parallel cta_n
LogicalResult CUDAMMAGEMMDefaultScheduleFactory::assignSchedule(
    PatternDescription& pd, ModuleOp m, DeviceInfo deviceInfo) {
  OpBuilder b(m);
  b.setInsertionPointToStart(&m.getBodyRegion().front());
  Location loc = m.getLoc();
  MLIRContext* ctx = m->getContext();
  auto transformOpType = transform::AnyOpType::get(ctx);
  auto seqOp = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate,
      transformOpType, [&](OpBuilder& b, Location loc, Value variantH) {});
  auto& bodyBlock = seqOp.getBody().front();
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
  auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
  if (!lhsBatchingDims.empty()) {
    return m->emitError() << "do not support batch matmul\n";
  }
  bool lhsTranspose = (lhsCntractingDims[0] == lhsTy.getRank() - 2);
  bool rhsTranspose = (rhsCntractingDims[0] == rhsTy.getRank() - 1);
  if (lhsTranspose || rhsTranspose) {
    return m->emitError() << "only support row-major matmul now\n";
  }
  int64_t M = lhsTy.getShape()[lhsTy.getRank() - 2];
  int64_t K = lhsTy.getShape()[lhsTy.getRank() - 1];
  int64_t N = rhsTy.getShape()[rhsTy.getRank() - 1];

  // build handle to target dot op.
  Value fillAndMatmul = buildMatchOp(b, loc, variant, {}, nameMap[dotOp]);
  auto matmulSplitOp = buildSplitHandleOp(b, loc, fillAndMatmul, 2);
  Value fill = matmulSplitOp->getResult(0);
  Value matmul = matmulSplitOp->getResult(1);

  // ========================== Multi-level tiling ==========================

  // Thread-block level tiling. Fixed tile size 128 x 128.
  const SmallVector<int64_t> ctaTileSizes{128, 128, 32};
  SmallVector<Attribute> blockTileMapping{
      gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimX),
      gpu::GPUBlockMappingAttr::get(ctx, gpu::Blocks::DimY)};
  auto blockTileMappingAttr = b.getArrayAttr(blockTileMapping);

  auto forallOpBlock =
      buildTileToForallOp(b, loc, matmul, {ctaTileSizes[0], ctaTileSizes[1]},
                          transform::TileSizesSpec(), blockTileMappingAttr);
  Value forallLoopBlock = forallOpBlock->getResult(0);
  Value tiledMatmulBlock = forallOpBlock->getResult(1);

  // Fuse fill op in to the forall loop.
  auto fuseIntoContainingOp =
      buildFuseIntoContainingOp(b, loc, fill, forallLoopBlock);

  // TODO: padding on block tile.

  // K iteration on block tile.
  auto splitReductionSerialOpBlock =
      buildSplitReductionSerialOp(b, loc, tiledMatmulBlock, {ctaTileSizes[2]},
                                  StringAttr::get(ctx, "cta-k-loop"));
  auto splitMatmulBlock = splitReductionSerialOpBlock->getResult(0);

  // Promote operands for shared memory buffering.
  // TODO: promote operands for register buffering.
  auto promoteDotOperandsOp =
      buildPromoteDotOperandsOp(b, loc, splitMatmulBlock, {0, 1});
  auto promotedMatmul = promoteDotOperandsOp->getResult(0);

  // TODO: software pipelining on k iteration.

  // Warp tile.
  const SmallVector<int64_t> warpTileSizes{64, 64, 32};
  SmallVector<Attribute> warpTileMapping{
      gpu::GPUWarpMappingAttr::get(ctx, gpu::Warps::DimX),
      gpu::GPUWarpMappingAttr::get(ctx, gpu::Warps::DimY)};
  auto warpTileMappingAttr = b.getArrayAttr(warpTileMapping);

  auto forallOpWarp = buildTileToForallOp(
      b, loc, promotedMatmul, {warpTileSizes[0], warpTileSizes[1]},
      transform::TileSizesSpec(), warpTileMappingAttr);
  Value forallLoopWarp = forallOpWarp->getResult(0);
  Value tiledMatmulWarp = forallOpWarp->getResult(1);

  // K iteration on warp tile.
  auto splitReductionSerialOpWarp =
      buildSplitReductionSerialOp(b, loc, tiledMatmulWarp, {warpTileSizes[2]},
                                  StringAttr::get(ctx, "warp-k-loop"));
  auto splitMatmulWarp = splitReductionSerialOpWarp->getResult(0);

  // Vector op tile.
  SmallVector<int64_t> vectorTileSizes;
  // The MMA instruction configuration for fp16.
  if (deviceInfo.cc_major >= 8) {
    vectorTileSizes = {16, 8, 16};
  } else if (deviceInfo.cc_major == 7 && deviceInfo.cc_minor == 5) {
    vectorTileSizes = {16, 8, 8};
  } else if (deviceInfo.cc_major == 7 && deviceInfo.cc_minor == 0) {
    vectorTileSizes = {8, 8, 4};
  } else {
    return m->emitError() << "unsupported GPU compute capacity\n";
  }
  auto tileOpVector =
      buildTileOp(b, loc, splitMatmulWarp, vectorTileSizes, {0, 1, 2});

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // TODO: fully unroll the vector tiled loops.

  // ============================= Vectorization =============================

  Value func4Vec = buildMatchOp(b, loc, variant, {"func.func"});
  func4Vec = buildRunCanonicalizer(b, loc, func4Vec);
  auto vectorizeOp = buildVectorize(b, loc, func4Vec, true);

  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // ============================= Bufferization =============================

  variant = buildDISCBufferize(b, loc, variant, true);
  Value funcAfterBufferize = buildMatchOp(b, loc, variant, {"func.func"});
  buildEraseDeallocOp(b, loc, funcAfterBufferize);
  Value func2ConvertTransfer = buildMatchOp(b, loc, variant, {"func.func"});
  // TODO: init with 0 in parallel.
  buildTransferWriteZeroToSCFOp(b, loc, func2ConvertTransfer);

  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // ==================== ForallOp to GPU mappings ====================

  auto blockTileMappingDictAttr =
      b.getDictionaryAttr({b.getNamedAttr("mapping", blockTileMappingAttr)});
  Value forallBlock = buildMatchOp(b, loc, variant, {"scf.forall"}, {},
                                   blockTileMappingDictAttr);
  auto parallelOp = buildForallToGPUCTAsOp(b, loc, forallBlock);

  auto warpTileMappingDictAttr =
      b.getDictionaryAttr({b.getNamedAttr("mapping", warpTileMappingAttr)});
  Value forallWarp = buildMatchOp(b, loc, variant, {"scf.forall"}, {},
                                  warpTileMappingDictAttr);
  buildForallToGPUWarpsOp(b, loc, forallWarp);

  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // ======================== Gmem to Smem conversion ========================

  Value genericOp = buildMatchOp(b, loc, variant, {"linalg.generic"});
  buildLowerGmemToSmemOp(b, loc, genericOp);

  // TODO: shared memory swizzle to avoid bank conflict.

  // ========================= Convert vector to mma =========================

  Value func4MMA = buildMatchOp(b, loc, variant, {"func.func"});
  auto vectorToMMAConversionOp = buildVectorToMMAConversionOp(b, loc, func4MMA);
  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  // ============================ Post processing ============================

  Value func4PostProcess = buildMatchOp(b, loc, variant, {"func.func"});
  buildInlineAndConvertGPUIdsOp(b, loc, func4PostProcess);

  buildLICMOp(b, loc, variant);
  buildDCEOp(b, loc, variant);
  buildCSEOp(b, loc, variant);

  b.create<transform::YieldOp>(loc);

  return success();
}

#if ENABLE_AARCH64_SCHEDUELS
DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, kDefaultScheduleFactoryPriority,
                        Aarch64GEMMDefaultScheduleFactory,
                        ArrayRef<StringRef>{kDefaultScheduleFactoryTag},
                        DeviceType::kCPU);

DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, 10,
                        Aarch64GEMMDefaultScheduleWithEpilogueFactory,
                        ArrayRef<StringRef>{"default_epilogue"},
                        DeviceType::kCPU);

DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, 100,
                        Aarch64GEMMLargeKScheduleFactory,
                        ArrayRef<StringRef>{"large_k"}, DeviceType::kCPU);

DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, 110,
                        Aarch64GEMMLargeKScheduleWithEpilogueFactory,
                        ArrayRef<StringRef>{"large_k_epilogue"},
                        DeviceType::kCPU);
#endif  // ENABLE_AARCH64_SCHEDUELS

// CUDA schedules
DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, 1000,
                        CUDAMMAGEMMDefaultScheduleFactory,
                        ArrayRef<StringRef>{"cuda_mma_default"},
                        DeviceType::kGPU);

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

DeviceType deviceTypeFromString(const std::string& str) {
  auto& map = getStringToDeviceTypeMap();
  auto it = map.find(str);
  if (it != map.end()) {
    return it->second;
  }
  llvm_unreachable("unknown device type str");
  return DeviceType::kNone;
}

PatternDescription::PatternDescription(lmhlo::FusionOp op,
                                       FusionPattern& fusionPattern,
                                       ShapeAnalysis& shapeAnalysis)
    : op_(op),
      fusionPattern_(fusionPattern),
      shapeAnalysis_(shapeAnalysis),
      tagSet_(parsefusionTagSetFromStr(getFusionTagStr(op))),
      deviceType_(placement_utils::isGpuMhlo(op) ? DeviceType::kGPU
                                                 : DeviceType::kCPU) {
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

DeviceType PatternDescription::getPatternDeviceType() const {
  return deviceType_;
}

std::string PatternDescription::getTaggedPatternStr() const {
  return patternKindToString(patternKind_) + "@" + getPatternTagStr();
}

ScheduleFactory::ScheduleFactory(int64_t id, PatternKind kind,
                                 ArrayRef<StringRef> tags,
                                 DeviceType deviceType)
    : id_(id), kind_(kind), deviceType_(deviceType) {
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

bool ScheduleFactory::checkFusionPatternProperties(PatternDescription& pd) {
  // Check the device.
  if (deviceType_ != pd.getPatternDeviceType()) {
    return false;
  }
  return true;
}

LogicalResult ScheduleFactory::buildGuardCondition(OpBuilder& b, Location loc,
                                                   PatternDescription&,
                                                   Value&) {
  return failure();
}

LogicalResult ScheduleFactory::assignSchedule(PatternDescription&, ModuleOp,
                                              DeviceInfo) {
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
      if (it->second->noGuardCondition(pd)) {
        break;
      }
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
      "<pattern-kind-0>:<tag-str-0>:<device-0>:<filename-0>;"
      "<pattern-kind-1>:<tag-str-1>:<device-1>:<filename-1>";
  SmallVector<StringRef> patternSettings;
  StringRef(transformFileName_)
      .split(patternSettings, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  int priority = kParsedFromFileScheduleFactoryStartPriority;
  for (auto& patternSetting : patternSettings) {
    SmallVector<StringRef> items;
    patternSetting.split(items, ":", /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    if (items.size() != 4) {
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
    DeviceType deviceType = deviceTypeFromString(items[2].str());
    if (deviceType == DeviceType::kNone) {
      llvm::dbgs() << "illegal transform file setting, unknown device type: "
                   << items[2] << "\n";
      return failure();
    }

    if (failed(parseTransformModuleFromFile(ctx, items[3], transformModule))) {
      llvm::dbgs()
          << "illegal transform file setting, unable to load module from: "
          << items[3] << "\n";
      return failure();
    }
    SmallVector<StringRef> tags;
    items[1].split(tags, kFusionTagSeparator, /*MaxSplit*/ -1,
                   /*KeepEmpty*/ false);
    ScheduleFactoryRegistry::get().registerScheduleFactory(
        kind, priority++,
        std::make_unique<ParsedFromFileScheduleFactory>(
            ScheduleFactoryRegistry::get().getNextUniqueId(), kind, tags,
            deviceType, transformModule.get()));
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

  return factory->assignSchedule(pd, m, getDeviceInfo());
}

}  // namespace disc_ral
}  // namespace mlir
