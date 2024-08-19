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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_SHAPE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_SHAPE_UTILS_H_

namespace mlir {
namespace disc_ral {

// Represents a symbolic dimension.
class SymbolDim {
 public:
  SymbolDim(int64_t dim_size = ShapedType::kDynamic) : dimSize_(dim_size) {}

  int64_t getDimSize() { return dimSize_; }

  bool isDynamic() { return getDimSize() == ShapedType::kDynamic; }

  LogicalResult Merge(SymbolDim* other);

 private:
  int64_t dimSize_;
};

// Represents a symbolic ranked shape.
class SymbolShape {
 public:
  explicit SymbolShape(SmallVector<SymbolDim*, 4> dims = {})
      : dims_(std::move(dims)) {}

  int rank() { return dims_.size(); }

  void setSymbolDims(SmallVector<SymbolDim*, 4> dims) {
    dims_ = std::move(dims);
  }

  ArrayRef<SymbolDim*> getSymbolDims() { return dims_; }

  SmallVector<int64_t, 4> getDims() {
    SmallVector<int64_t, 4> dims;
    for (SymbolDim* dim : dims_) dims.push_back(dim->getDimSize());
    return dims;
  }

  SymbolDim* getSymbolDim(int64_t dim) {
    assert(dim < dims_.size());
    return dims_[dim];
  }

  void setSymbolDim(int64_t dim, SymbolDim* symbolDim) {
    assert(dim < dims_.size());
    dims_[dim] = symbolDim;
  }

  bool operator==(const SymbolShape& other) { return other.dims_ == dims_; }

 private:
  SmallVector<SymbolDim*, 4> dims_;
};

using llvm::EquivalenceClasses;
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

struct DimValue {
  enum State : int32_t { FOLD = 0, UNFOLD = 1, INVALID = 2 } state;
  int64_t foldVal = INT64_MIN;
  Value unfoldVal = nullptr;
  DimValue() { state = INVALID; }
  explicit DimValue(Value val) {
    unfoldVal = val;
    state = (val == nullptr) ? INVALID : UNFOLD;
  }
  explicit DimValue(int64_t val) {
    foldVal = val;
    state = (val >= 0) ? FOLD : INVALID;
  }
  bool isValid() const { return state != INVALID; }
  bool isFold() const { return state == FOLD; }
  bool isUnfold() const { return state == UNFOLD; }
  void dump() const {
    if (isFold()) {
      llvm::errs() << "Fold value: " << foldVal << "\n";
    } else if (isUnfold()) {
      llvm::errs() << "Unfold value: ";
      auto& non_const = const_cast<Value&>(unfoldVal);
      non_const.dump();
    } else {
      llvm::errs() << "Invalid DimValue\n";
    }
  }
  inline bool operator==(const DimValue& dimVal) const {
    if (state == dimVal.state) {
      return (isFold() && foldVal == dimVal.foldVal) ||
             (isUnfold() && unfoldVal == dimVal.unfoldVal);
    }
    return false;
  }
  inline bool operator!=(const DimValue& dimVal) const {
    return !(*this == dimVal);
  }
  inline bool operator<(const DimValue& dimVal) const {
    if (state != dimVal.state) {
      return state < dimVal.state;
    } else if (foldVal != dimVal.foldVal) {
      return foldVal < dimVal.foldVal;
    } else {
      return unfoldVal.getAsOpaquePointer() <
             dimVal.unfoldVal.getAsOpaquePointer();
    }
  }
  std::size_t hash() const {
    std::size_t h = llvm::hash_value(state);
    h = llvm::hash_combine(h, llvm::hash_value(foldVal));
    h = llvm::hash_combine(h, mlir::hash_value(unfoldVal));
    return h;
  }
};

struct DimValueHash {
  std::size_t operator()(const DimValue& dimVal) const { return dimVal.hash(); }
};

struct DimValueContainerHash {
  std::size_t operator()(const std::vector<DimValue>& ds) const {
    std::size_t hash = llvm::hash_value(ds.size());
    for (const auto& d : ds) {
      hash = llvm::hash_combine(hash, d.hash());
    }
    return hash;
  }
};

// A helper class to query and manipulate shape constraint IR on buffer level.
class ShapeAnalysis {
 public:
  virtual ~ShapeAnalysis() = default;

  // Returns true if the two value have the same symbolic shape.
  virtual bool isShapeEqual(Value lhs, Value rhs) = 0;

  // Suppose:
  //  lhsDimIdxs = {ld0, ld1, ...}
  //  rhsDimIdxs = {rd0, rd1, ...}
  // Returns true if `lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  // rhs.shape[rd0] * rhs.shape[rd1] * ...`
  virtual bool isProductEqual(Value lhs, ArrayRef<int> lhsDimIdxs, Value rhs,
                              ArrayRef<int> rhsDimIdxs) = 0;

  // Returns true if:
  //  lhs.shape[lhsFrom] * ... lhs.shape[lhsTo-1] ==
  //  rhs.shape[rhsFrom] * ... rhs.shape[rhsTo-1]
  virtual bool isProductEqual(Value lhs, int lhsFrom, int lhsTo, Value rhs,
                              int rhsFrom, int rhsTo);

  // Returns true if the two value have the same number elements.
  virtual bool isSameNumElements(Value lhs, Value rhs);
};

// A subclass to impement `ShapeAnalysis` on buffer level.
// The implementation is based on shape constraint ir.
class ShapeConstraintIRAnalysis : public ShapeAnalysis {
 public:
  // Build shape related analysis on the provided `op`.
  // This generally can be divided into two steps:
  // 1, load exsiting shape constraint ir (e.g. symbolic dim ops)
  // 2, build mapping between memref values and symbolic dim ops.
  explicit ShapeConstraintIRAnalysis(Operation* op);
  // auto-save updated shape constriant ir when destroying.
  ~ShapeConstraintIRAnalysis();

  // Returns the `SymbolicDimMgr` this object holds.
  SymbolicDimMgr& symbolicDimMgr() { return mgr_; }
  const SymbolicDimMgr& symbolicDimMgr() const { return mgr_; }

  // Returns true if the two value have the same symbolic shape.
  bool isShapeEqual(Value lhs, Value rhs) override;

  // Suppose:
  //  lhsDimIdxs = {ld0, ld1, ...}
  //  rhsDimIdxs = {rd0, rd1, ...}
  // Returns true if `lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  // rhs.shape[rd0] * rhs.shape[rd1] * ...`
  bool isProductEqual(Value lhs, ArrayRef<int> lhsDimIdxs, Value rhs,
                      ArrayRef<int> rhsDimIdxs) override;

 private:
  // The operation this analysis runs on.
  Operation* op_;

  // The `SymbolicDimMgr` this analysis holds.
  SymbolicDimMgr mgr_;

  // Map a ranked memref value to an array of symbolicDims, each represents one
  // dimension size of the memref value.
  DenseMap<Value, SmallVector<disc_shape::SymbolicDimOp>> memrefValue2SymDims_;
};

// Shape analysis for propagating and analyzing known shape information in
// compilation time in a given operation.
class ShapeAnalysisDeprecated : public ShapeAnalysis {
 public:
  explicit ShapeAnalysisDeprecated(Operation* op) : op_(op) {}

  LogicalResult run();

  Operation* getOperation() { return op_; }

  Type getRefinedType(Value value);

  SymbolShape* getShape(Value value);
  DimValue getDimValue(Value operand, int64_t dim);
  bool isDimEqual(Value lhs, int64_t lhsDim, Value rhs, int64_t rhsDim);
  bool isShapeEqual(Value lhs, Value rhs) override;
  bool isShapeValueEqual(Value lhs, Value rhs);

  // Insert tie_shape ops to explicit tie dimension equality in the IR level.
  // Used for shape simplification pass.
  LogicalResult buildTieShapeOps();

  // Returns true if `lhs` and `rhs` are supposed to have same number of
  // elements.
  bool isSameNumElements(Value lhs, Value rhs) override;

  bool isProductEqual(Value lhs, ArrayRef<int> lhsDimIdxs, Value rhs,
                      ArrayRef<int> rhsDimIdxs) override {
    return false;
  }

  // Extract continuous equal dims between `lhs` and `rhs`, with the
  // consideration of dim-linearize (e.g., caused by reshape). Note that when
  // there are multiple mapping possibilities, only the first case is extracted
  // and retured. For example, for [a, a] -> [a], it mappes the first `a` of lhs
  // to rhs.
  // TODO: deal with multiple mappings with context information.
  bool extractContinuousDimEqualInfo(
      Value lhs, Value rhs,
      SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>&
          equal);

  // Build equal shape information for the given values. The equal information
  // is maintained in a self-maintained map with `fusion` as the key.
  void buildEqualShapesInFusion(const Operation* fusion,
                                const DenseSet<Value>& values_in_fusion);

  // Deprecated. Get the leader value with same shape for `val` in `op_`.
  Value GetLeaderValueWithSameShapeGlobal(Value val) const;
  // Get the leader value with same shape for `val` in  `fusion`.
  Value GetLeaderValueWithSameShapeInFusion(const Operation* fusion,
                                            Value val) const;

 private:
  LogicalResult buildShapeMap();
  LogicalResult buildSymbolShape(Value value);
  LogicalResult buildRegionShapeMap(Region* region);
  LogicalResult buildBlockShapeMap(Block* block);
  LogicalResult buildOperationShapeMap(Operation* op);

  LogicalResult buildOperationShapeValueMap(Operation* op);

  LogicalResult buildRegionDimValueMap(Region* region);
  LogicalResult buildBlockDimValueMap(Block* block);
  LogicalResult buildDimValueMap(Value operand, int64_t dim, DimValue dimValue);

  LogicalResult applyOpConstraint(Operation* op);
  LogicalResult applyTensorOpConstraint(Operation* op);
  LogicalResult applyMhloOpConstraint(Operation* op);
  LogicalResult applyLmhloOpConstraint(Operation* op);
  // Build dim equality according to dim-values.
  LogicalResult applyDimEqualWithDimValue();
  void postProcessingDimValues();

  // Build dim decompose relationship.
  LogicalResult buildRegionMulDecompose(Region* region);
  LogicalResult buildBlockMulDecompose(Block* block);

  SymbolDim* getRootDim(SymbolDim* symbolDim);
  SymbolDim* getDim(Value value, int64_t dim);
  DimValue getRootDimValue(DimValue dimValue);
  DimValue getDimValue(SymbolDim* symbolDim);

  LogicalResult mapDimEqual(SymbolDim* lhs, SymbolDim* rhs);
  LogicalResult mapDimEqual(Value lhs, int64_t lhsDim, Value rhs,
                            int64_t rhsDim);
  LogicalResult mapDimValueEqual(DimValue lhs, DimValue rhs);
  LogicalResult mapShapeEqual(SymbolShape* lhs, SymbolShape* rhs);
  LogicalResult mapShapeEqual(Value lhs, Value rhs);

  bool isDimValueEqual(DimValue lhs, DimValue rhs);
  SmallVector<std::vector<DimValue>> getDimDecompose(Value value,
                                                     int64_t index);

  using SymbolShapeVisitor =
      std::function<LogicalResult(OpBuilder&, Location&, Value value)>;
  using Symbol2InstancesDominantType =
      DenseMap<SymbolDim*, DenseMap<Value, Value>>;
  LogicalResult visitSymbolShapes(SymbolShapeVisitor visitor);
  LogicalResult buildSymbolDimInstances(
      DenseMap<SymbolDim*, SmallVector<Value>>& symbolDim2Instances,
      DenseMap<Value, SmallVector<Value>>& shapedValue2Dims);
  LogicalResult buildSymbolDimInstancesDominantMap(
      DenseMap<SymbolDim*, SmallVector<Value>>& instanceMap,
      DenseMap<SymbolDim*, DenseMap<Value, Value>>& dominantMap);
  void dumpSymbol2InstancesDominant(
      Symbol2InstancesDominantType symbol2InstancesDominant);

  bool getConstIntValue(Operation* op, int64_t& result);

 private:
  bool initialized = false;
  Operation* op_;
  SmallVector<std::unique_ptr<SymbolDim>, 4> dimVec_;
  DenseMap<SymbolDim*, int64_t> dimMap_;
  DenseMap<Value, SymbolShape> shapeMap_;

  EquivalenceClasses<DimValue> dimValueEquivalence_;
  DenseMap<SymbolDim*, DimValue> dimSymbol2DimValue_;
  // Suppose `val` can be delinearized to {a, b} and {a, c, d}, we will form
  // {val, {{a, b}, {a, c, d}}} in `dimValueRootDelinearize_`.
  std::unordered_map<DimValue, SmallVector<std::vector<DimValue>>, DimValueHash>
      dimValueMulDecompose_;

  // The tensor of shape value (e.g., the last operand of mhlo.dynamic_reshape.)
  EquivalenceClasses<ValueWrapper> shapeValueEqual_;
  DenseMap<Value, Value> value2ShapeValue_;

  // The following is in order to be compatible with the existing codegen logic.
  EquivalenceClasses<ValueWrapper> valueWithEqualShape_;
  EquivalenceClasses<ValueWrapper> valueWithSameElements_;

  // { fusion_op, equal value classes }
  DenseMap<const Operation*, EquivalenceClasses<ValueWrapper>>
      valueWithEqualShapeInFusion_;
};

// This is a simple shape equality analysis given a list of operations.
//
// Currently, We only consider shape equality and same-number-elements equality
// propagation based on the shape constraint traits of elementwise ops (assuming
// that implicit shape broadcast is forbidden).
// Deprecated. Will be replaced with `ShapeAnalysis` in the future.
class OpListShapeAnalysis {
 public:
  explicit OpListShapeAnalysis(const SmallVectorImpl<Operation*>& op_list) {
    PropagateEquality(op_list);
  }

  // Returns true if `lhs` and `rhs` are supposed to have same shape.
  bool isShapeEqual(Value lhs, Value rhs) {
    return same_shape_impl_.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs));
  }

  // Returns true if `lhs` and `rhs` are supposed to have same number of
  // elements.
  bool isSameNumElements(Value lhs, Value rhs) {
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

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_SHAPE_UTILS_H_