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

// Represents a symbolic dimension.
class SymbolicDim {
 public:
  int64_t uniqueId() const;

  int64_t getDimSize() const { return dimSize_; }

  bool isDynamic() const { return getDimSize() == ShapedType::kDynamicSize; }

  LogicalResult Merge(SymbolicDim* other);

 private:
  int64_t dimSize_;
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

  SmallVector<SymbolicDim*> getOrCreateSymbolicDimsForRankedValue(Value value);

  // All symbolic-equal dims form a group.
  // Returns the root SymbolicDim of the symbolic-equal symbolic dim group that
  // this SymbolicDim belongs to.
  SymbolicDim* getRootSymbolicDim(SymbolicDim* symbol);

  LogicalResult mapSymbolicDimEqual(SymbolicDim* lhs, SymbolicDim* rhs);

  //   SymbolicDim* getSymbolicDimUsingRef(const FlatSymbolRefAttr& ref);

  LogicalResult save();

 private:
  //   DenseMap<std::string, SymbolicDim*> symbolRef2symbolicDim_;
  SmallVector<std::unique_ptr<SymbolicDim>, 4> symbolicDimStorage_;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_UTILS_H_