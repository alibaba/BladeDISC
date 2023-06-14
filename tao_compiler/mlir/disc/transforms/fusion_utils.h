/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_FUSION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_FUSION_UTILS_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"
#include "mlir/disc/transforms/shape_utils.h"

#define DEBUG_TYPE "disc-fusion-utils"

// This file implements some helper functions and classes used to do fusion
// & code generation.

namespace llvm {
template <>
struct DenseMapInfo<SmallVector<mlir::Value>> {
  static SmallVector<mlir::Value> getEmptyKey() {
    return SmallVector<mlir::Value>{DenseMapInfo<mlir::Value>::getEmptyKey()};
  }

  static SmallVector<mlir::Value> getTombstoneKey() {
    return SmallVector<mlir::Value>{
        DenseMapInfo<mlir::Value>::getTombstoneKey()};
  }

  static unsigned getHashValue(const SmallVector<mlir::Value>& vs) {
    unsigned hash = hash_value(vs.size());
    for (auto v : vs) hash = llvm::hash_combine(hash, v);
    return hash;
  }

  static bool isEqual(const SmallVector<mlir::Value>& lhs,
                      const SmallVector<mlir::Value>& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

namespace mlir {
namespace disc_ral {

DenseSet<Operation*> NoLoaderUser(SmallVectorImpl<Operation*>& ops);

// If `rewriter` is not null, try to erase the unused ops through it.
// Otherwise remove the unused ops directly.
// Note that if we are inside a rewriter pattern, we have to set the `rewriter`
// in order to notify the listeners of the rewriter.
void cleanUnusedLhloOps(Block* parent, PatternRewriter* rewriter = nullptr);

// returns the users of the `memref`. The users should be in the same fusion
// like `op`.
DenseSet<Operation*> getValueUsersInFusionLike(Value memref, Operation* op);

bool isOnGpu(Operation* op);

// Attributes used to annotate the fusion type, fusion name and tags.
constexpr const char* kDiscFusionTypeAttrName = "disc.fusion_type";
constexpr StringRef kFusionOpNameAttr = "disc.fusion.name";
constexpr StringRef kFusionOpTagAttr = "disc.fusion.tag";
constexpr const char* kFusionTagSeparator = "X";

// kLoop fusion template satisfies:
//   - all ops in the fusion pattern are element-wise.
//   - all the shapes of outputs of fusion pattern are same or have same number
//   of elements, and thus can fit into a same parallel loop.
//
// kInput fusion template satisfies:
//   - any op in the fusion pattern is either element-wise or a reduction.
//   - if a op is a reduction, its output cannot be consumed by other
//     ops in the same fusion pattern.
//   - all the effective shapes of outputs of fusion pattern are same.
//     - For element-wise op, its effective shape is its output shape.
//     - For reduction op, its effective shape is its operand shape.
//   - currently our downstreaming codegen engine only support 2d -> 1d tensor
//   reduction. TODO: lift this limitation.
//     - 2D row reduction: out[i] = sum({in[i][j] for all j})
//     - 2D column reduction: out[j] = sum({in[i][j] for all i})
// kStitch
//   - stitch multi kernels into a bigger kernel using scratch memory
enum FusionType {
  // Not a known fusion pattern
  kNone,
  // kLoop fusion pattern
  kLoop,
  // kInput fusion pattern and all reduce ops of the fused pattern are row
  // reduction
  kRowReduction,
  // kInput fusion pattern and all reduce ops of the fused pattern are column
  // reduction
  kColReduction,
  // kInput fusion pattern
  kInput,
  // Stitch Fusion pattern
  kStitch,
  // A schedule for concat op having many operands.
  kLargeConcat,
  // Dot fusion.
  kDot,
  // Where op fusion, maybe need a more general name?
  kWhere,
  // transform dialect based codegen fusion pattern
  kTransform,
  // sparse reduction fusion,
  kSparseReduction,
};

FusionType getFusionType(Operation* op);

// Convert a fusion type to its string representation.
StringRef fusionTypeToString(FusionType ft);

// Convert to a fusion type from its string representation.
FusionType fusionTypeFromString(StringRef ft);

// Returns true if the op is an elementwise unary lmhlo op.
// TODO: use fusibility interface
bool isElementWiseUnary(Operation* op);

// Returns true if the op is an elementwise binary lmhlo op.
// TODO: use fusibility interface
bool isElementWiseBinary(Operation* op);

// Returns true if the op is an elementwise lmhlo op.
// TODO: use fusibility interface
bool isElementWise(Operation* op);

// Returns true if this op is a rank-2 row reduction.
bool isRank2RowReduction(Operation* op);

// Returns true if this op is a row reduction.
bool isRowReduction(Operation* op);

// Returns true if this op is a rank-2 column reduction.
bool isRank2ColReduction(Operation* op);

// Returns true if this op is a rank-2 or rank-3 transpose
bool isRank2or3Transpose(Operation* op);

// Returns true if the op is supported by the downstreaming fusion codegen
// engine.
bool isFusible(Operation* op);

// Return true if enable transpose library call
bool enableTransposeLibraryCall();

// Returns data users of the value and its aliases (e.g. memref.cast).
// Here non-data users means DimOp, DeallocOp and ShapeOfOp.
SmallVector<Operation*, 4> getValueUsers(Value v);

struct TileInfo {
  // Maps axis -> tile_size along this axis.
  // select all the elements along the axis if tile_size ==
  // ShapedType::kDynamic
  DenseMap<int64_t, int64_t> tileSizes;

  // Returns false if failed to merge.
  bool merge(TileInfo& other);

  // Returns false if failed to merge.
  bool merge(int64_t axis, int64_t tileSize = ShapedType::kDynamic);

  // return true if updated.
  bool updateIfNotEqual(TileInfo& other);
};

// Basic information of fused operation set, including op-list, operands,
// results, et.al. It does not provide informations for codegen schedules, like
// dominant-op, subroot, fusion-type, et.al.
class FusionPatternBase {
 public:
  using FusionOpList = SmallVector<Operation*, 4>;
  using FusionValueList = SmallVector<Value, 4>;

  // Create a new fusion pattern from a single op.
  explicit FusionPatternBase(Operation* op);

  // Create a new fusion pattern from the ops inside the lmhlo fusion op.
  explicit FusionPatternBase(lmhlo::FusionOp op);

  // Create a new fusion pattern with the ops inside the list.
  explicit FusionPatternBase(SmallVectorImpl<Operation*>& op_list);

  // Returns the op list this fusion pattern represents.
  FusionOpList& getOpList() { return op_list_; }

  // Returns values that are consumed by the lmhlo ops inside the fusion
  // pattern.
  FusionValueList& getOperands() { return operands_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // have consumers outside the fusion pattern.
  FusionValueList& getResults() { return results_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // have consumers outside the fusion pattern.
  SmallVector<Operation*, 4>& getRootOps() { return root_ops_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // are only consumed by the lmhlo ops inside the fused pattern.
  FusionValueList& getInternalResults() { return internal_results_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // are only consumed by the lmhlo ops outside the fused pattern.
  FusionValueList& getExternalOnlyResults() { return external_only_results_; }

  // Return last writer map of ops in the fusion pattern.
  DenseMap<Value, Operation*>& getLastWriter() { return last_writer_; }

  // Returns the size of the ops this fusion pattern contains.
  int size() { return op_list_.size(); }

  // Returns the effective size (e.g. not counting const ops) of the ops this
  // fusion pattern contains.
  int effectiveSize();

  // Sorts the ops inside the fusion pattern according to the keys provided.
  void sortFusionOpListBy(DenseMap<Operation*, int>& op_to_idx);

  void sortFusionOpListWithTopologyOrder();

  // Here `value` is supposed to be a pointer to buffer.
  // Returns the defining op of `value `if no known op updates the buffer,
  // otherwise returns the last op that updates the buffer pointed by the
  // `value`.
  Operation* findLastWriter(Value value) {
    auto it = last_writer_.find(value);
    if (it != last_writer_.end()) {
      return it->second;
    }
    return value.getDefiningOp();
  }

  void updateLastWriter(Value value, Operation* op) {
    last_writer_[value] = op;
  }

  bool alreadyInRootOps(Operation* new_op) {
    for (Operation* op : root_ops_) {
      if (new_op == op) {
        return true;
      }
    }
    return false;
  }

 protected:
  // Calculates the inputs and outputs of the fusion pattern.
  void calculateOperandsAndResults();

  FusionOpList op_list_;
  FusionValueList operands_;
  FusionValueList results_;
  FusionValueList internal_results_;
  FusionValueList external_only_results_;
  SmallVector<Operation*, 4> root_ops_;
  DenseMap<Value, Operation*> last_writer_;
};

// Represents a list of lmhlo ops that are going to be fused.
// Concepts for a fusion pattern:
//   - Root op: the op whose output is the fusion-pattern's output.
//   - Sub-root op: the op whose output is to be maintained on shared-memory for
//     kStitch fusion. Currently, we only support row-reduction to be a sub-root
//     op.
//   - Regular xroot op: either a root op or a sub-root op, for whose operands
//     we successfully build tile information during kStitch fusion-pattern init
//     phase.
//   - Irregular xroot op: an root op for whose operands we fail to build tile
//     information durint kStitch fusion-pattern init phase.
//   - Skeleton op: the op who will be used to build the loop skeleton when
//     lowering a kStitch fusion to parallel loops. Currently, sub-root ops, and
//     regular xroot ops who generate external only results, are skeleton ops.
//     Other xroot ops are lowered with input-inline fusion phase.
//   Note: for an regular xroot op which is not an skeleton op, the output data
//     to be written should be coverred by its corresponding skeleton op.
//     Otherwise, this xroot are regared as irregular.
class FusionPattern : public FusionPatternBase {
 public:
  // Create a new fusion pattern from a single op.
  explicit FusionPattern(Operation* op);

  // Create a new fusion pattern from the ops inside the lmhlo fusion op.
  explicit FusionPattern(lmhlo::FusionOp op, ShapeAnalysis* shape_analysis);

  // Do not allow to build a fusion pattern with only FusionOp.
  explicit FusionPattern(lmhlo::FusionOp op) = delete;

  // Returns the dominant op of this fusion pattern.
  // For kLoop fusion, a dominant op may be any op that has external users.
  // For kInput fusion, a dominant op may be a row reduction (if exists), or
  // a column reduction op.
  Operation* getDominantOp() { return dominant_op_; }

  // Sets the dominant op to the op provided.
  void setDominantOp(Operation* op) { dominant_op_ = op; }

  // Returns the fusion kind of the fusion pattern.
  FusionType getFusionType() { return fusion_type_; }

  // Returns the fusion kind of the fusion pattern.
  StringRef getFusionTypeStr() { return fusionTypeToString(fusion_type_); }

  // Sets the fusion type to the the type provided.
  void setFusionType(FusionType type) { fusion_type_ = type; }

  // Returns true if this a fusible fusion pattern.
  bool isFusible() { return getFusionType() != FusionType::kNone; }

  // Returns true if this fusion pattern is a kLoop fusion.
  bool isKLoopFusion() { return getFusionType() == FusionType::kLoop; }

  // Returns true if this fusion pattern is a kInput fusion.
  bool isKInputFusion() {
    return (getFusionType() == FusionType::kRowReduction ||
            getFusionType() == FusionType::kColReduction);
  }

  // Returns true if the fusion type is stitch fusion.
  bool isStitchFusion() { return getFusionType() == FusionType::kStitch; }

  // Returns true if the fusion type is transform-based fusion.
  bool isTransformBasedFusion() {
    return getFusionType() == FusionType::kTransform;
  }

  // Merges two fusion patterns and returns the merged pattern. The original
  // pattern remains unmodified. The new merged pattern is uninitialized.
  FusionPattern mergeWithoutInit(FusionPattern& other);

  // Create a new fusion pattern with the given op list, without init.
  static FusionPattern createWithoutInit(SmallVectorImpl<Operation*>& op_list);

  DenseMap<Value, TileInfo>& getTilePlan() { return tile_plan_; }
  void setTilePlan(const DenseMap<Value, TileInfo>& tile_plan) {
    tile_plan_ = tile_plan;
  }

  SmallVector<Operation*, 4>& getSubRootOps() { return sub_root_ops_; }

  void setSubRootOps(const SmallVector<Operation*, 4>& sub_root_ops) {
    sub_root_ops_ = sub_root_ops;
  }

  struct SkeletonGroup {
    SmallVector<Operation*> skeletons;
    SmallVector<Operation*> root_member_list;
    // An irregular member means whose non-tiled dims are not exactly matched
    // with skeleton. This requires special designe for GPU block mapping when
    // generating the code.
    DenseSet<Operation*> irregular_root_member_set;
  };

  void findOpsOfSkeletonGroup(
      SkeletonGroup group, DenseSet<Operation*>& ops,
      DenseSet<Operation*>& shmem_cached_ops,
      const DenseMap<Operation*, SmallVector<Operation*>>& existing_group_ops,
      int row_per_block, int& shmem_usage_bits, const int shmem_limit_bits);

  int64_t getCollapsedTileDim(Value value);

  DenseSet<Operation*>& getRegularXroots() { return regular_xroots_; }

  DenseSet<Operation*>& getIrregularXroots() { return irregular_xroots_; }

 private:
  // Create a new fusion pattern with the ops inside the list.
  explicit FusionPattern(SmallVectorImpl<Operation*>& op_list);

  Operation* dominant_op_ = nullptr;
  FusionType fusion_type_ = FusionType::kNone;
  SmallVector<Operation*, 4> sub_root_ops_;
  DenseMap<Value, TileInfo> tile_plan_;
  // An xroot op is either a root or a sub-root op. Regular xroots are those
  // whose element-number of non-tileds dimes are the same with sub-root ops.
  // Otherwise an xroot is irregular.
  DenseSet<Operation*> regular_xroots_;
  DenseSet<Operation*> irregular_xroots_;
};

void dumpFusionPattern(FusionPattern& pattern);

// Returns true if the FusionPattern type is in {kWhere, kSparseReduction}.
bool isSparseFusion(FusionPattern& pattern);

// The basic approch to init fusion pattern.
bool initFusionPatternBase(ShapeAnalysis& shapeAnalysis,
                           FusionPattern& fusion_pattern);

// Get skeleton-groups in which the op orders are the same with op list in the
// given fusion pattern.
bool getOrderedSkeletonGroups(
    FusionPattern& pattern, SmallVector<FusionPattern::SkeletonGroup>& groups);

// Merge skeleton-ops that are not producer-consumer relationship together.
bool mergeSkeletonGroupsInOrder(
    FusionPattern& pattern, SmallVector<FusionPattern::SkeletonGroup>& groups,
    ShapeAnalysis* shape_analysis);

void dumpTilePlan(DenseMap<Value, TileInfo>& tilePlan);

// Represents a list of disjoint fusion patterns for a block.
using FusionPlan = std::vector<FusionPattern>;

// Returns the name of the fusion op
StringRef getFusionName(lmhlo::FusionOp op);

// Sets the name of the fusion op
void setFusionName(OpBuilder& b, lmhlo::FusionOp op, StringRef name);

// Attaches a new tag to the fusion op.
// Here different tags is mapping to different variants of the fusion op.
void addFusionTag(OpBuilder& b, lmhlo::FusionOp op, StringRef tag);

// Merge the tags of the op and `tagSet`, assign the new tag set to `op`.
void mergeFusionTag(OpBuilder& b, lmhlo::FusionOp op,
                    const std::set<std::string>& tagSet);

// Returns the tag string attached to the fusion op.
StringRef getFusionTagStr(lmhlo::FusionOp op);

// Returns the unified tag string for the whole tagSet.
std::string fusionTagSetToStr(const std::set<std::string>& tagSet);

// Returns the parsed tag set from `tagStr`.
std::set<std::string> parsefusionTagSetFromStr(StringRef tagStr);

// Returns the full name of the fusion op
// Here full name is composed of the name and tag of the fusion op.
std::string getFusionFullName(lmhlo::FusionOp op);

// Generates a signature string for the fusion pattern.
std::string generateSignatureForFusion(FusionPattern& fusionPattern);

// Returns true if both two ops are in the same fusion family.
bool inSameFusionFamily(Operation* op, Operation* other);

// Returns true if both two ops are in the same fusion op.
bool inSameFusionOp(Operation* op, Operation* other);

int64_t getFirstOperandIndex(Operation* op, Value value);

// Non valid: 'middle' col-reduction, non-2d-reduction.
bool findValidReductionOps(FusionPatternBase& target,
                           SmallVectorImpl<Operation*>& row_reductions,
                           SmallVectorImpl<Operation*>& col_reductions);

struct FusionOptions {
  // Maximum allowed number of arguments per fused kernel. Here arguments
  // include both read-only buffers and writable buffers.
  int max_num_arguments_per_kernel = 64;
};

void setGlobalFusionOptions(const FusionOptions& options);

// Here 'large' refer to having many operands.
bool isLargeConcatOp(Operation* op);

// Represents a specific fusion strategy.
// Examples are:
//  - basic fusion strategy for CPU device
//  - basic fusion strategy for GPU device
//  - stitch fusion strategy for CPU device
//  - stitch fusion strategy for GPU device
class FusionStrategy {
 public:
  FusionStrategy(const FusionOptions& options) : options_(options) {}

  virtual bool isFusible(Operation* op);
  virtual bool isFusible(FusionPattern& fusion_pattern);
  virtual bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& fusion_pattern) = 0;
  virtual bool pruneFusionPattern(ShapeAnalysis& shapeAnalysis,
                                  FusionPattern& fusion_pattern,
                                  SmallVectorImpl<Operation*>& excluded_ops);
  virtual bool tryFuseInplace(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                              FusionPattern& rhs);
  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target);
  virtual StringRef getName() { return "FusionStrategy"; }

 protected:
  FusionOptions options_;
};

using DeviceStrategyMap = DenseMap<StringRef, FusionStrategy*>;

class PlacementAwareFusionStrategy : public FusionStrategy {
 public:
  PlacementAwareFusionStrategy(const FusionOptions& options,
                               StringRef defaultDevice,
                               DeviceStrategyMap deviceStrategyMap)
      : FusionStrategy(options),
        deviceStrategyMap_(std::move(deviceStrategyMap)),
        defaultDevice_(defaultDevice) {}

  bool isFusible(Operation* op) override;
  bool isFusible(FusionPattern& fusion_pattern) override;
  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                         FusionPattern& fusion_pattern) override;
  virtual bool pruneFusionPattern(
      ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern,
      SmallVectorImpl<Operation*>& excluded_ops) override;
  virtual StringRef getName() override {
    return "PlacementAwareFusionStrategy";
  }
  DeviceStrategyMap getStrategyMap() { return deviceStrategyMap_; }

 private:
  StringRef getPlacement(Operation* op);
  StringRef getPlacement(FusionPattern& fusion_pattern);
  FusionStrategy* getStrategy(StringRef placement);
  FusionStrategy* getStrategy(Operation* op) {
    return getStrategy(getPlacement(op));
  }
  FusionStrategy* getStrategy(FusionPattern& fusion_pattern) {
    return getStrategy(getPlacement(fusion_pattern));
  }

  StringRef defaultDevice_;
  DeviceStrategyMap deviceStrategyMap_;
};

// Creates and returns a new placement-aware fusion strategy.
std::unique_ptr<FusionStrategy> makeNewPlacementAwareFusionStrategy(
    bool gpu_enabled, StringRef strategy);

// a -> b iff graph[a][b] = true;
using ValueGraph = DenseMap<Value, DenseMap<Value, bool>>;

// Represents a symbol index for parallel dimension.
struct ParallelIndex {
  // a unique id per instance
  int id;
  // step value used to create parallel for loop.
  // this value usually is one. In some cases, it may be other value.
  // Using buffer `memref<?x?x?xf32>` as an example:
  //  - tile size is: <1x?x4xf32>
  //  - parallel indices:
  //    - dim 0, step 1
  //    - dim 2, step 4
  int64_t step = 1;

  // The actual index value at compile time.
  // unknown if it's ShapedType::kDynamic, otherwise it's a constant
  // parallel index.
  int64_t value = ShapedType::kDynamic;
};

// Represents the parallel info for a buffer.
struct ParallelInfo {
  // a unique id per instance
  int id;

  // producer unique id
  int producerId;

  // Ids of all consumers of this parallel info.
  DenseSet<int> consumerIds;

  // The value this parallel info targets.
  Value value;

  // op that connects from producerId to current id.
  Operation* op = nullptr;

  // map parallel axis to the id of parallel index.
  // each id represents a unique parallel index.
  DenseMap<int, int> indices;
  SmallVector<Value> symbolIndices;
  // a unique id represents a inBound pred symbol.
  using InBoundPred = int;
  InBoundPred inBound;
  Value symbolInBound;
  // a unique id represents a isOwner pred symbol.
  using IsOwnerPred = int;
  IsOwnerPred isOwner;
  Value symbolIsOwner;

  // whether this parallel info is needed by any root.
  bool consumedByRoots = false;

  // Returns the sorted parallel axes.
  SmallVector<int> getSortedParallelAxes() {
    SmallVector<int> parallelAxes;
    for (auto&& e : indices) parallelAxes.push_back(e.first);
    llvm::sort(parallelAxes);
    return parallelAxes;
  }
};

class StitchCPUAnalysis {
 public:
  explicit StitchCPUAnalysis(FusionPattern& fusionPattern,
                             ShapeAnalysis& shapeAnalysis)
      : fusionPattern_(fusionPattern), shapeAnalysis_(shapeAnalysis) {}

  // Returns true if the fusion pattern is a valid stitch pattern.
  bool fusibilityAnalysis();

  // Do the first level (tile level) codegen for stitch fusion pattern.
  // Returns true if success, otherwise false.
  bool doCodeGeneration(OpBuilder& b, lmhlo::FusionOp fusion);

 private:
  // Builds value dominant graph. Here buffer `a` dominates buffer `b` means
  // `a` is larger than or equal to `b`.
  bool buildDominantGraph(ValueGraph& dominantGraph);

  // 1, each root is an output of the fusion pattern.
  // 2, find a minimum buffer that dominates all root buffer.
  // 3, return false if no buffer is qualified to be a dominant.
  bool doRootsAnalysis();

  // Assigns a tile sizes for each buffers.
  // Take buffer `a` memref<?x?x?xf32> as an example:
  //  Tile(a) = {0 : -1}: means axis 0 is fully selected as tile
  //  Tile(a) = {1 : -1, 2 : 4}: means axis 1 is fully selected and
  //                             tile size for axis 2 is 4.
  bool doTileAnalysis();
  bool doElemOpTileAnalysis(DenseMap<Value, TileInfo>& tilePlan, Operation* op,
                            bool& changed);
  bool doReduceOpTileAnalysis(DenseMap<Value, TileInfo>& tilePlan,
                              Operation* op, bool& changed);
  bool doBcastOpTileAnalysis(DenseMap<Value, TileInfo>& tilePlan, Operation* op,
                             bool& changed);
  bool doReshapeOpTileAnalysis(DenseMap<Value, TileInfo>& tilePlan,
                               Operation* op, bool& changed);

  // Returns a unique id per instance.
  int newSymbolId() { return nextSymbolId_++; }

  // Analyzes parallel indices of dominant value to roots & all related producer
  // buffers.
  bool doParallelAnalysis();
  // Creates a new parallel index.
  ParallelIndex& makeParallelIndex(int64_t step = 1,
                                   int64_t value = ShapedType::kDynamic);
  // Creates a new parallel info.
  ParallelInfo& makeParallelInfo(Value value, int producerId = 0,
                                 Operation* op = nullptr);
  // Propagates dominant parallel info to all roots.
  bool propagateFromDominantToRoots();
  // Back-propagation from roots to their operands.
  bool propagateFromRootsToProducers();
  // Returns true if the  parallelInfo id set is consistent.
  bool isConsistentParallelInfoSet(DenseSet<int>& idSet);
  // Debug-Only
  void dumpParallelPlan();

  // Sub-roots analysis. cache some intermediate results to avoid expensive
  // re-computation.
  bool doSubRootsAnalysis();

  // Codegen utils.
  using ValueViewStore = DenseMap<SmallVector<Value>, Value>;
  using ViewStore = DenseMap<Value, ValueViewStore>;
  Value getDominantValue() { return dominantValue_; }
  ParallelInfo& getDominantParallelInfo();
  DenseMap<int, ParallelIndex>& getParallelIndexStore() {
    return parallelIndexStore_;
  }
  bool isFusionOperand(Value v) {
    auto& operands = fusionPattern_.getOperands();
    return llvm::find(operands, v) != operands.end();
  }
  bool isFusionResult(Value v) {
    auto& results = fusionPattern_.getResults();
    return llvm::find(results, v) != results.end();
  }
  // used for emitting the outter tile-level parallel loop
  scf::ParallelOp emitTileParallelLoop(OpBuilder& b, Location loc);
  // used for emitting parallel indices
  bool emitParallelIndices(OpBuilder& b, Location loc,
                           ValueRange dominantIndex);
  bool emitElemOpParallelIndex(OpBuilder& b, Location loc, ParallelInfo& from,
                               ParallelInfo& to);
  bool emitReduceOpParallelIndex(OpBuilder& b, Location loc, ParallelInfo& from,
                                 ParallelInfo& to);
  bool emitBcastOpParallelIndex(OpBuilder& b, Location loc, ParallelInfo& from,
                                ParallelInfo& to);
  bool emitReshapeOpParallelIndex(OpBuilder& b, Location loc,
                                  ParallelInfo& from, ParallelInfo& to);
  // used for emitting in/out subviews
  bool emitInOutTiles(OpBuilder& b, Location loc, ViewStore& viewStore);
  // used for emitting sub root tile buffers
  Value emitTileBuffer(OpBuilder& b, Location loc, Value val);
  bool emitSubRootTile(OpBuilder& b, Location loc, Value val,
                       ViewStore& viewStore);
  // used for emitting sub-root calculations
  bool emitAllSubRootsAndRootsCalculation(OpBuilder& b, Location loc);
  bool emitSubRootCalculation(OpBuilder& b, Location loc, ParallelInfo& info,
                              ViewStore& viewStore,
                              SmallVectorImpl<Operation*>& clonedLmloOps);
  bool emitInputSlice(OpBuilder& b, Location loc, Value out,
                      ParallelInfo& parallelInfo,
                      SmallVectorImpl<Operation*>& clonedLmloOps);

 private:
  FusionPattern& fusionPattern_;
  ShapeAnalysis& shapeAnalysis_;

  // Used for roots dominant analysis
  Value dominantValue_;
  ValueGraph dominantGraph_;

  // Used for tile analysis
  DenseMap<Value, TileInfo> tilePlan_;

  // Used for parallel analysis
  int nextSymbolId_ = 0;
  // map parallel index id -> parallel index instance
  DenseMap<int, ParallelIndex> parallelIndexStore_;
  // map a constant value to the id of the corresponding const parallel index.
  DenseMap<int64_t, int> constParallelIndexStore_;
  // map parallel info id -> parallel info instance
  DenseMap<int, ParallelInfo> parallelInfoStore_;
  // map buffer value to the ids of its parallel info instances.
  DenseMap<Value, DenseSet<int>> parallelPlan_;

  // Used for sub-roots analysis
  DenseSet<Value> subRootsAndRootsSet_;

  // Codegen utils:
  ViewStore inOutViewStore_;
  ViewStore subRootViewStore_;
  Operation* parallelOp_;
};

template <FusionType... Types>
bool isFusionType(Operation* op) {
  SmallVector<FusionType, 2> fusionTypes({Types...});
  FusionType fusionType = FusionType::kNone;
  auto fusionTypeAttr = op->getAttrOfType<StringAttr>(kDiscFusionTypeAttrName);
  if (fusionTypeAttr) {
    fusionType = fusionTypeFromString(fusionTypeAttr.getValue());
  }
  return llvm::find(fusionTypes, fusionType) != fusionTypes.end();
}

class BaseFusionStrategy : public FusionStrategy {
 public:
  using FusionStrategy::FusionStrategy;

  using FusionStrategy::isFusible;
  bool isFusible(FusionPattern& fusion_pattern) override;
  bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
               FusionPattern& rhs, FusionPattern& target) override;
  bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                         FusionPattern& fusion_pattern) override;
  virtual StringRef getName() override { return "BaseFusionStrategy"; }

 protected:
  virtual Value getEffectiveShape(FusionPattern& target, Value value) = 0;
  virtual bool checkSameShape(FusionPattern& lhs, FusionPattern& rhs,
                              FusionPattern& target) {
    return false;
  }
};

class BaseGpuFusionStrategy : public BaseFusionStrategy {
 public:
  using BaseFusionStrategy::BaseFusionStrategy;

  bool isFusible(Operation* op) override;
  bool checkSameShape(FusionPattern& lhs, FusionPattern& rhs,
                      FusionPattern& target) {
    return lhs.isKInputFusion() && rhs.isKInputFusion();
  }
  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target) override;
  Value getEffectiveShape(FusionPattern& target, Value v) override;
  virtual StringRef getName() override { return "BaseGpuFusionStrategy"; }
};

class StitchGpuFusionStrategy : public FusionStrategy {
 public:
  StitchGpuFusionStrategy(const FusionOptions& options)
      : FusionStrategy(options) {}
  virtual bool isFusible(Operation* op) override;
  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target) override;
  virtual bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& fusion_pattern) override;
  virtual StringRef getName() override { return "StitchGpuFusionStrategy"; }

 private:
  virtual Value getEffectiveShape(FusionPattern& target, Value value);

  bool tileCoverInfoPropagateO2I(
      ShapeAnalysis& shapeAnalysis, DenseMap<Value, TileInfo>& tile_plan,
      Operation* op, SmallVector<std::pair<Value, TileInfo>, 4>& in_info,
      bool& cover);
  bool findFusionPatternTypeAndSubroot(ShapeAnalysis& shapeAnalysis,
                                       FusionPattern& fusion_pattern);
  bool tileXroots(ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern);
  bool backtraceTileAndCover(ShapeAnalysis& shapeAnalysis,
                             FusionPattern& fusion_pattern, Value value);
};

class PreDotGpuFusionStrategy : public BaseFusionStrategy {
 public:
  using BaseFusionStrategy::BaseFusionStrategy;

  virtual bool isFusible(Operation* op) override;
  Value getEffectiveShape(FusionPattern& target, Value v) override;

  virtual StringRef getName() override { return "PreDotGpuFusionStrategy"; }
};

class DotGpuFusionStrategy : public FusionStrategy {
 public:
  DotGpuFusionStrategy(const FusionOptions& options)
      : FusionStrategy(options) {}

  virtual bool isFusible(Operation* op) override;
  virtual bool isFusible(FusionPattern& fusion_pattern) override;
  virtual bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& fusion_pattern) override;
  virtual bool pruneFusionPattern(
      ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern,
      SmallVectorImpl<Operation*>& excluded_ops) override;

  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target) override;

  virtual StringRef getName() override { return "DotGpuFusionStrategy"; }

 private:
  SmallVector<Value> getEffectiveOperands(Operation* op);
};

class TransformBasedCpuFusionStrategy : public FusionStrategy {
 public:
  TransformBasedCpuFusionStrategy(const FusionOptions& options)
      : FusionStrategy(options) {}

  virtual bool isFusible(Operation* op) override;
  virtual bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& fused_pattern) override;
  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target) override;

  virtual StringRef getName() override {
    return "TransformBasedCpuFusionStrategy";
  }
};

class SparseOpCpuFusionStrategy : public FusionStrategy {
 public:
  SparseOpCpuFusionStrategy(const FusionOptions& options)
      : FusionStrategy(options) {}

  virtual bool isFusible(Operation* op) override;
  virtual bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& fused_pattern) override;
  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target) override;

  virtual StringRef getName() override { return "SparseOpCpuFusionStrategy"; }
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_FUSION_UTILS_H_
