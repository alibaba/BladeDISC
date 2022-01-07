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
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project

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

// An Attribute used to annotate the fusion type.
constexpr const char* kDiscFusionTypeAttrName = "disc.fusion_type";

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
  // TAO v1/v2 Stitch Fusion
  kStitch,
  // A schedule for concat op having many operands.
  kLargeConcat,
};

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

// Returns true if the op is supported by the downstreaming fusion codegen
// engine.
bool isFusible(Operation* op);

// Returns the number of operands that are supposed to be written.
// For some ops (e.g. lmhlo ops), some operands are the output memrefs
// Thus these operands are supposed to be updated.
int getNumResultOperands(Operation* op);

// Returns data users of the value and its aliases (e.g. memref.cast).
// Here non-data users means DimOp, DeallocOp and ShapeOfOp.
SmallVector<Operation*, 4> getValueUsers(Value v);

// Represents a list of lmhlo ops that are going to be fused.
class FusionPattern {
 public:
  using FusionOpList = SmallVector<Operation*, 4>;
  using FusionValueList = SmallVector<Value, 4>;

  // Create a new fusion pattern from a single op.
  explicit FusionPattern(Operation* op);

  // Create a new fusion pattern from the ops inside the lmhlo fusion op.
  explicit FusionPattern(lmhlo::FusionOp op);

  // Returns the op list this fusion pattern represents.
  FusionOpList& getOpList() { return op_list_; }

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

  // Merges two fusion patterns and returns the merged pattern. The original
  // pattern remains unmodified. The new merged pattern is uninitialized.
  FusionPattern mergeWithoutInit(FusionPattern& other);

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

  // Returns the size of the ops this fusion pattern contains.
  int size() { return op_list_.size(); }

  // Returns the effective size (e.g. not counting const ops) of the ops this
  // fusion pattern contains.
  int effectiveSize();

  // Sorts the ops inside the fusion pattern according to the keys provided.
  void sortFusionOpListBy(DenseMap<Operation*, int>& op_to_idx);

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

 private:
  FusionPattern(SmallVectorImpl<Operation*>& op_list);

 private:
  // Calculates the inputs and outputs of the fusion pattern.
  void calculateOperandsAndResults();

 private:
  FusionOpList op_list_;
  Operation* dominant_op_ = nullptr;
  FusionType fusion_type_ = FusionType::kNone;
  FusionValueList operands_;
  FusionValueList results_;
  FusionValueList internal_results_;
  SmallVector<Operation*, 4> root_ops_;
  DenseMap<Value, Operation*> last_writer_;
};

// Represents a list of disjoint fusion patterns for a block.
using FusionPlan = std::vector<FusionPattern>;

using llvm::EquivalenceClasses;

// Supports using EquivalenceClasses for Value
class ValueWrapper {
 public:
  explicit ValueWrapper(Value value) : value_(std::move(value)) {}

  Value getValue() const { return value_; }

  bool operator==(const ValueWrapper& rhs) const {
    return getValue() == rhs.getValue();
  }

 private:
  Value value_;
};

bool operator<(const ValueWrapper& lhs, const ValueWrapper& rhs);

// This is a simple shape constraint analysis, which is used to
// guide fusion decision (e.g. we only fuse shape-compatible ops).
//
// Currently, We only consider shape equality and same-number-elements equality
// propagation based on the shape constraint traits of elementwise ops (assuming
// that implicit shape broadcast is forbidden).
class ShapeConstraintAnalysis {
 public:
  explicit ShapeConstraintAnalysis(const SmallVectorImpl<Operation*>& op_list) {
    PropagateEquality(op_list);
  }

  // Returns true if `lhs` and `rhs` are supposed to have same shape.
  bool HasSameShape(Value lhs, Value rhs) {
    return same_shape_impl_.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs));
  }

  // Returns true if `lhs` and `rhs` are supposed to have same number of
  // elements.
  bool HasSameNumElements(Value lhs, Value rhs) {
    return same_num_elements_impl_.isEquivalent(ValueWrapper(lhs),
                                                ValueWrapper(rhs));
  }

  Value GetLeaderValueWithSameShape(Value val) const {
    if (same_shape_impl_.findLeader(ValueWrapper(val)) ==
        same_shape_impl_.member_end()) {
      return nullptr;
    }
    return same_shape_impl_.getLeaderValue(ValueWrapper(val)).getValue();
  }

 private:
  // shape equality propagation based on the shape constrains of
  // elementwise ops.
  void PropagateEquality(const SmallVectorImpl<Operation*>& op_list);

  // a UnionFind set
  EquivalenceClasses<ValueWrapper> same_shape_impl_;
  EquivalenceClasses<ValueWrapper> same_num_elements_impl_;
};

// Returns the name of the fusion op
StringRef getFusionName(lmhlo::FusionOp op);

// Sets the name of the fusion op
void setFusionName(OpBuilder& b, lmhlo::FusionOp op, StringRef name);

// Attaches a new tag to the fusion op.
// Here different tags is mapping to different variants of the fusion op.
void addFusionTag(OpBuilder& b, lmhlo::FusionOp op, StringRef tag);

// Returns the full name of the fusion op
// Here full name is composed of the name and tag of the fusion op.
std::string getFusionFullName(lmhlo::FusionOp op);

// Generates a signature string for the fusion op.
std::string generateSignatureForFusion(lmhlo::FusionOp op);

// Returns true if both two ops are in the same fusion family.
bool inSameFusionFamily(Operation* op, Operation* other);

// Returns true if both two ops are in the same fusion op.
bool inSameFusionOp(Operation* op, Operation* other);

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
  virtual bool isFusible(FusionPattern& fused_pattern);
  virtual bool initFusionPattern(ShapeConstraintAnalysis& shapeAnalysis,
                                 FusionPattern& fused_pattern) = 0;
  bool tryFuseInplace(ShapeConstraintAnalysis& shapeAnalysis,
                      FusionPattern& lhs, FusionPattern& rhs);
  virtual bool tryFuse(ShapeConstraintAnalysis& shapeAnalysis,
                       FusionPattern& lhs, FusionPattern& rhs,
                       FusionPattern& target);

 protected:
  FusionOptions options_;
};

// Creates and returns a new placement-aware fusion strategy.
std::unique_ptr<FusionStrategy> makeNewPlacementAwareFusionStrategy(
    bool gpu_enabled, StringRef strategy);

// a -> b iff graph[a][b] = true;
using ValueGraph = DenseMap<Value, DenseMap<Value, bool>>;

struct TileInfo {
  // Maps axis -> tile_size along this axis.
  // select all the elements along the axis if tile_size ==
  // ShapedType::kDynamicSize
  DenseMap<int, int> tileSizes;

  // Returns false if failed to merge.
  bool merge(TileInfo& other);

  // Returns false if failed to merge.
  bool merge(int axis, int tileSize = ShapedType::kDynamicSize);

  // return true if updated.
  bool updateIfNotEqual(TileInfo& other);
};

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
};

class StitchCPUAnalysis {
 public:
  explicit StitchCPUAnalysis(FusionPattern& fusionPattern)
      : fusionPattern_(fusionPattern) {}

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

  // Returns a unique id per instance.
  int newSymbolId() { return nextSymbolId_++; }

  // Analyzes parallel indices of dominant value to roots & all related producer
  // buffers.
  bool doParallelAnalysis();
  // Creates a new parallel index.
  ParallelIndex& makeParallelIndex(int64_t step = 1);
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

  // Used for roots dominant analysis
  Value dominantValue_;
  ValueGraph dominantGraph_;

  // Used for tile analysis
  DenseMap<Value, TileInfo> tilePlan_;

  // Used for parallel analysis
  int nextSymbolId_ = 0;
  // map parallel index id -> parallel index instance
  DenseMap<int, ParallelIndex> parallelIndexStore_;
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

bool isStitchFusion(Operation* op);

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_FUSION_UTILS_H_
