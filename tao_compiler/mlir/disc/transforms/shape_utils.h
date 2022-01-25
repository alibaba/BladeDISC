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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_SHAPE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_SHAPE_UTILS_H_

namespace mlir {
namespace disc_ral {

// Represents a symbolic dimension.
class SymbolDim {
 public:
  SymbolDim(int64_t dim_size = ShapedType::kDynamicSize) : dimSize_(dim_size) {}

  int64_t getDimSize() { return dimSize_; }

  bool isDynamic() { return getDimSize() == ShapedType::kDynamicSize; }

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

// Shape analysis for propagating and analyzing known shape information in
// compilation time in a given operation.
class ShapeAnalysis {
 public:
  explicit ShapeAnalysis(Operation* op) : op_(op) {}

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
    void dump() const {
      if (state == FOLD) {
        llvm::errs() << "Fold value: " << foldVal << "\n";
      } else if (state == UNFOLD) {
        llvm::errs() << "Unfold value: ";
        auto& non_const = const_cast<Value&>(unfoldVal);
        non_const.dump();
      } else {
        llvm::errs() << "Invalid DimValue\n";
      }
    }
    inline bool operator==(const DimValue& dimVal) const {
      if (state == dimVal.state) {
        return (state == FOLD && foldVal == dimVal.foldVal) ||
               (state == UNFOLD && unfoldVal == dimVal.unfoldVal);
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
    std::size_t operator()(const DimValue& dimVal) const {
      return dimVal.hash();
    }
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

  LogicalResult run();

  Operation* getOperation() { return op_; }

  Type getRefinedType(Value value);

  bool isDimEqual(Value lhs, int64_t lhsDim, Value rhs, int64_t rhsDim);
  bool isShapeEqual(Value lhs, Value rhs);
  bool isShapeValueEqual(Value lhs, Value rhs);

  // Insert tie_shape ops to explicit tie dimension equality in the IR level.
  // Used for shape simplification pass.
  LogicalResult buildTieShapeOps();

  // Returns true if `lhs` and `rhs` are supposed to have same number of
  // elements.
  bool HasSameNumElements(Value lhs, Value rhs);

  // Extract continuous equal dims between `lhs` and `rhs`, with the
  // consideration of dim-linearize (e.g., caused by reshape).
  bool extractContDimEqualInfo(
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
  DimValue getDimValue(Value operand, int64_t dim);
  SymbolShape* getShape(Value value);

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

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_SHAPE_UTILS_H_