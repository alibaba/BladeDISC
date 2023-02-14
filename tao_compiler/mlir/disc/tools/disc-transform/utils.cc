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

#include "mlir/disc/tools/disc-transform/utils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#define DEBUG_TYPE "disc-transform-utils"

namespace mlir {
namespace disc_ral {

const char* kDISCLinalgTransformName = "disc.transform.name";

TransformNameAssigner::TransformNameAssigner(ArrayRef<Operation*> ops) {
  for (auto op : ops) nameNewOperation(op);
}

std::string TransformNameAssigner::nameNewOperation(Operation* op) {
  SmallString<128> fullName;
  auto opName = op->getName().stripDialect().str();
  fullName.append(opName);
  if (auto cnt = nameCounterMap_[opName]++) {
    fullName.append(("_" + Twine(cnt)).str());
  }

  return nameMap_[op] = fullName.str();
}

const std::unordered_map<Operation*, std::string>&
TransformNameAssigner::getNameMap() {
  return nameMap_;
}

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation* createLinalgCopyOp(OpBuilder& b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes) {
  auto memrefTypeFrom = from.getType().dyn_cast<MemRefType>();
  auto memrefTypeTo = to.getType().dyn_cast<MemRefType>();
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::makeArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder& b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
}

/// Load transform dialect IR from the given file.
LogicalResult parseTransformModuleFromFile(
    MLIRContext* context, llvm::StringRef transformFileName,
    OwningOpRef<ModuleOp>& transformModule) {
  // Parse transformFileName content into a ModuleOp.
  std::string errorMessage;
  auto memoryBuffer = openInputFile(transformFileName, &errorMessage);
  if (!memoryBuffer) {
    llvm::errs() << "failed to parse transform file: " << transformFileName
                 << "\n";
    return failure();
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  transformModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  return success();
}

}  // namespace disc_ral
}  // namespace mlir
