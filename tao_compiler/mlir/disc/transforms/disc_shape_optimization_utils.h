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

#include <memory>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#ifndef TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_

namespace mlir {
namespace disc_ral {

// Returns a unique id for each call.
int64_t getNextSymbolicDimUniqueId();

// Represents a symbolic dimension.
class SymbolicDim {
 public:
  int64_t uniqueId() const;

  int64_t getDimSize() const { return dimSize_; }

  void setDimSize(int64_t val) { dimSize_ = val; }

  bool isDynamic() const { return getDimSize() == ShapedType::kDynamicSize; }

  LogicalResult Merge(SymbolicDim* other);

 private:
  int64_t uniqueId_ = getNextSymbolicDimUniqueId();
  int64_t dimSize_ = ShapedType::kDynamicSize;
};

// Return the symbolicDim ref attribute if there is an attached disc
// shape-constraint specific attribute filed. Return nullptr if there isn't an
// attached symbolic dim ref attributes.
llvm::Optional<SmallVector<FlatSymbolRefAttr>> getRankedValueSymbolicDimRefs(
    Value value);

class SymbolicDimMgr {
 public:
  explicit SymbolicDimMgr(ModuleOp m);

  LogicalResult load();

  SymbolicDim* newSymbolicDim();

  // Returns a symbolicDim which have static dim size == `val`.
  SymbolicDim* newConstantSymbolicDim(int64_t val);

  SmallVector<SymbolicDim*> getOrCreateSymbolicDimsForRankedValue(Value value);

  // All symbolic-equal dims form a group.
  // Returns the root SymbolicDim of the symbolic-equal symbolic dim group that
  // this SymbolicDim belongs to.
  SymbolicDim* getRootSymbolicDim(SymbolicDim* symbol);

  // Returns true if lhs and rhs are known to be equal.
  bool isSymbolicDimEqual(SymbolicDim* lhs, SymbolicDim* rhs);

  // Marks lhs and rhs have same size and try to merge lhs & rhs static known
  // info. Returns failure if failed to merge lhs & rhs.
  LogicalResult mapSymbolicDimEqual(SymbolicDim* lhs, SymbolicDim* rhs);

  //   SymbolicDim* getSymbolicDimUsingRef(const FlatSymbolRefAttr& ref);

  LogicalResult save();

 private:
  //   DenseMap<std::string, SymbolicDim*> symbolRef2symbolicDim_;
  SmallVector<std::unique_ptr<SymbolicDim>, 4> symbolicDimStorage_;

  // map a symbolic dim -> its root SymbolicDim
  // Here root symbolic dim means the representative member in the
  // symbolic-equal symbolic dim set that this symbolic dim belongs to.
  DenseMap<SymbolicDim*, SymbolicDim*> symbolDimUnionSet_;

  // map a concret constant value to a symbolic dim instance that represents the
  // constant.
  DenseMap<int64_t, SymbolicDim*> constantSymbolicDimMap_;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_