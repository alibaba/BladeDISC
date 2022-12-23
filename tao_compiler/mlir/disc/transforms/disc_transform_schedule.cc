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
using transform::TileOp;
using transform::TileToForeachThreadOp;
using transform::VectorizeOp;

MatchOp buildMatchOp(OpBuilder& b, Location& loc, Value target,
                     ArrayRef<StringRef> ops) {
  return b.create<MatchOp>(loc, pdl::OperationType::get(b.getContext()), target,
                           b.getStrArrayAttr(ops),
                           transform::MatchInterfaceEnumAttr{},
                           DictionaryAttr{}, TypeAttr{});
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

LogicalResult aarch64GEMMDefaultScheduleFactory(PatternDescription& pd,
                                                ModuleOp m) {
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

  // %fill = transform.structured.match ops{["linalg.fill"]} in %variant
  Value fill = buildMatchOp(b, loc, variant, {"linalg.fill"});
  // %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
  Value matmul = buildMatchOp(b, loc, variant, {"linalg.matmul"});

  // transform.structured.tile_to_foreach_thread_op %matmul num_threads [1, 1]
  auto forEachThreadOp = buildTileToForEachThreadOp(b, loc, matmul, {1, 1});
  Value forEachThreadLoop = forEachThreadOp->getResult(0);
  Value tiledMatmul = forEachThreadOp->getResult(1);

  // transform.structured.fuse_into_containing_op %fill into %0#0
  auto fuseIntoContainingOp =
      buildFuseIntoContainingOp(b, loc, fill, forEachThreadLoop);

  // first level tile and fuse matmul and fill op.
  auto fuseOp0 = buildFuseOp(b, loc, tiledMatmul, {288, 48, 0}, {0, 1, 2});

  // second level tile and fuse matmul and fill op.
  auto fuseOp1 =
      buildFuseOp(b, loc, fuseOp0->getResult(0), {6, 16, 0}, {0, 1, 2});

  // gemm reduction axis tiling
  auto tileOp =
      buildTileOp(b, loc, fuseOp1->getResult(0), {0, 0, 1}, {0, 1, 2});

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
  forEachThreadLoop = buildMatchOp(b, loc, variant, {"scf.foreach_thread"});
  auto outterLoopForN = buildGetParentForOp(b, loc, padForInput, 4);
  buildCacheRead(b, loc, padForWeight, forEachThreadLoop, {1, 1}, {1, 16}, true,
                 {2, 0, 1, 3});
  buildCacheRead(b, loc, padForInput, outterLoopForN, {1, 1}, {6, 1}, true,
                 {0, 2, 3, 1});

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

DISC_TRANSFORM_SCHEDULE(PatternKind::kGEMM, "",
                        aarch64GEMMDefaultScheduleFactory);

}  // namespace

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
    : op_(op), fusionPattern_(fusionPattern), shapeAnalysis_(shapeAnalysis) {
  // TODO(wyzero): select the pattern kind according to the `fusionPattern`.
  patternKind_ = PatternKind::kGEMM;
}

PatternKind PatternDescription::getPatternKind() const { return patternKind_; }

std::string PatternDescription::getPatternTagStr() const {
  return getFusionTagStr(op_).str();
}

std::string PatternDescription::getTaggedPatternStr() const {
  return patternKindToString(patternKind_) + "@" + getPatternTagStr();
}

/* static */ ScheduleFactoryRegistry& ScheduleFactoryRegistry::get() {
  static ScheduleFactoryRegistry instance;
  return instance;
}

bool ScheduleFactoryRegistry::registerScheduleFactory(PatternKind kind,
                                                      const std::string& tag,
                                                      ScheduleFactory factory) {
  return factoryMap_[kind].emplace(tag, factory).second;
}

ScheduleFactory ScheduleFactoryRegistry::getScheduleFactory(
    PatternKind kind, const std::string& tag) {
  auto it = factoryMap_.find(kind);
  if (it == factoryMap_.end()) return nullptr;
  auto factoryIt = it->second.find(tag);
  if (factoryIt == it->second.end()) return nullptr;
  return factoryIt->second;
}

ScheduleDispatcher::ScheduleDispatcher(const std::string& transformFileName)
    : transformFileName_(transformFileName) {}

LogicalResult ScheduleDispatcher::parseModuleFromFile(MLIRContext* ctx) {
  if (transformFileName_.empty() || !parsedModuleMap_.empty()) return success();
  std::string expectedFormatStr =
      "<pattern-kind-0>:<tag-str-0>:<filename-0>;<pattern-kind-1>:<tag-str-1>:<"
      "filename-1>";
  SmallVector<StringRef> patternSettings;
  StringRef(transformFileName_)
      .split(patternSettings, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
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

    if (failed(parseTransformModuleFromFile(
            ctx, items[2], parsedModuleMap_[kind][items[1].str()]))) {
      llvm::dbgs()
          << "illegal transform file setting, unable to load module from: "
          << items[2] << "\n";
      return failure();
    }
  }
  return success();
}

bool ScheduleDispatcher::tryToApplyScheduleFromParsedFile(
    PatternDescription& pd, ModuleOp m) {
  auto it = parsedModuleMap_.find(pd.getPatternKind());
  if (it == parsedModuleMap_.end()) return false;
  auto tagIt = it->second.find(pd.getPatternTagStr());
  if (tagIt == it->second.end()) return false;
  OpBuilder b(m);
  for (auto& op : tagIt->second.get().getBody()->getOperations()) {
    if (!isa<transform::TransformOpInterface>(&op)) continue;
    m.push_back(b.clone(op));
  }
  return true;
}

LogicalResult ScheduleDispatcher::dispatch(PatternDescription& pd, ModuleOp m) {
  if (failed(parseModuleFromFile(m.getContext()))) return failure();
  if (tryToApplyScheduleFromParsedFile(pd, m)) return success();

  auto factory = ScheduleFactoryRegistry::get().getScheduleFactory(
      pd.getPatternKind(), pd.getPatternTagStr());
  if (!factory) {
    llvm::dbgs() << "no default schedule for pattern: "
                 << pd.getTaggedPatternStr() << "\n";
    return failure();
  }

  return factory(pd, m);
}

}  // namespace disc_ral
}  // namespace mlir
