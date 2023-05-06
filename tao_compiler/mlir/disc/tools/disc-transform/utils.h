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

#ifndef DISC_TOOLS_DISC_TRANSFORM_UTILS_H_
#define DISC_TOOLS_DISC_TRANSFORM_UTILS_H_

#include <string>
#include <unordered_map>

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace disc_ral {

// An attribute name used to represent the unique name assigned to each linalg
// ops. This tag can be used to distinguish different linalg ops when using
// transform IR.
extern const char* kDISCLinalgTransformName;

class TransformNameAssigner {
 public:
  TransformNameAssigner() = default;
  TransformNameAssigner(ArrayRef<Operation*> ops);

  // Returns the name assigned to the new `op`. The `op` must not be assigned
  // name by this assigner before.
  std::string nameNewOperation(Operation* op);

  // Returns the <op,name> map. The map records the names for all the ops that
  // have been passed to `nameNewOperation`.
  const std::unordered_map<Operation*, std::string>& getNameMap();

 private:
  // Assigned a unique id for the ops having the same op name.
  std::unordered_map<std::string, int> nameCounterMap_;
  std::unordered_map<Operation*, std::string> nameMap_;
};

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation* createLinalgCopyOp(OpBuilder& b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {});

/// Load transform dialect IR from the given file.
LogicalResult parseTransformModuleFromFile(
    MLIRContext* context, llvm::StringRef transformFileName,
    OwningOpRef<ModuleOp>& transformModule);

// Appends transform dependent dialects.
void addTransformDialectDependentDialects(DialectRegistry& registry);

/// Return true if the given memref has workgroup memory space.
bool hasSharedMemoryAddressSpace(MemRefType memrefType);

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TOOLS_DISC_TRANSFORM_UTILS_H_
