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

#ifndef TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_H_
#define TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_H_

namespace mlir {
namespace disc_ral {

// Represents a symbolic dimension.
class SymbolicDim {
 public:
  int64_t uniqueId() const;

 private:
};

class SymbolicDimMgr {
 public:
  explicit SymbolicDimMgr(ModuleOp m);

  LogicalResult load();

  LogicalResult save();

 private:
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_DISC_TRANSFORMS_DISC_SHAPE_OPTIMIZATION_H_