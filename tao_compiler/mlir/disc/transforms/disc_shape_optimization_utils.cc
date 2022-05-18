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

// This file implements the logic to do some shape optimizations on tensor
// level.

#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

#undef LLVM_DEBUG

#define LLVM_DEBUG(x) (x)

namespace mlir {
namespace disc_ral {

using ::mlir::func::FuncOp;

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) : m_(m) {
  // TODO
}

LogicalResult SymbolicDimMgr::load() {
  m_.walk([&](disc_shape::SymbolicDimOp op) {
    symbolDimUnionSet_[op] = op;
    symbolNameSet_.insert(op.getName().str());
  });
  return success();
}

std::string SymbolicDimMgr::getNextName() {
  std::string name;
  do {
    name = (llvm::Twine("S") + llvm::Twine(nextSymbolicOpIdx_++)).str();
  } while (!symbolNameSet_.insert(name).second);
  return name;
}

SymbolicDimOp SymbolicDimMgr::newSymbolicDim() {
  OpBuilder b(m_);
  auto symbol = b.create<SymbolicDimOp>(m_.getLoc(), getNextName());
  symbolDimUnionSet_[symbol] = symbol;
  m_.push_back(symbol);
  return symbol;
}

SymbolicDimOp SymbolicDimMgr::newConstantSymbolicDim(int64_t val) {
  auto it = constantSymbolicDimMap_.find(val);
  if (it == constantSymbolicDimMap_.end()) {
    it = constantSymbolicDimMap_.insert(std::make_pair(val, newSymbolicDim()))
             .first;
    it->second.setDimSize(val);
  }
  return getRootSymbolicDim(it->second);
}

SymbolicDimOp SymbolicDimMgr::getRootSymbolicDim(SymbolicDimOp symbol) {
  SymbolicDimOp current = symbol;
  while (symbolDimUnionSet_[current] != current)
    current = symbolDimUnionSet_[current];
  return current;
}

bool SymbolicDimMgr::isSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs) {
  SymbolicDimOp lhsRoot = getRootSymbolicDim(lhs);
  SymbolicDimOp rhsRoot = getRootSymbolicDim(rhs);
  return lhsRoot == rhsRoot;
}

bool SymbolicDimMgr::compareSymbolicDimOpNames(StringRef lhs, StringRef rhs) {
  // Not known encoding schema, fallback branch
  if (lhs[0] != 'S' || rhs[0] != 'S') return lhs < rhs;

  int64_t lhsIdx, rhsIdx;
  if (lhs.substr(1).getAsInteger(10, lhsIdx) ||
      rhs.substr(1).getAsInteger(10, rhsIdx))
    return lhs < rhs;
  return lhsIdx < rhsIdx;
}

LogicalResult SymbolicDimMgr::mapSymbolicDimEqual(SymbolicDimOp lhs,
                                                  SymbolicDimOp rhs) {
  SymbolicDimOp lhsRoot = getRootSymbolicDim(lhs);
  SymbolicDimOp rhsRoot = getRootSymbolicDim(rhs);

  if (lhsRoot != rhsRoot) {
    if (compareSymbolicDimOpNames(lhsRoot.getName(), rhsRoot.getName())) {
      if (failed(lhsRoot.Merge(rhsRoot))) return failure();
      symbolDimUnionSet_[rhsRoot] = lhsRoot;
    } else {
      if (failed(rhsRoot.Merge(lhsRoot))) return failure();
      symbolDimUnionSet_[lhsRoot] = rhsRoot;
    }
  }
  return success();
}

bool isSymbolicDimProductEqual(const SmallVectorImpl<SymbolicDimOp>& lhs,
                               const SmallVectorImpl<SymbolicDimOp>& rhs) {
  // TODO
  return false;
}

LogicalResult mapSymbolicDimProductEqual(
    const SmallVectorImpl<SymbolicDimOp>& lhs,
    const SmallVectorImpl<SymbolicDimOp>& rhs) {
  // TODO
  return success();
}

LogicalResult SymbolicDimMgr::save() {
  // replace all uses of a symbolic dim op with its root symbolic dim op
  if (failed(walkRankedTensorValue(m_, [&](Value value, RankedTensorType ty,
                                           ArrayAttr attrs) {
        SmallVector<Attribute> newAttrs;
        for (Attribute attr : attrs) {
          auto sym =
              m_.lookupSymbol<SymbolicDimOp>(attr.cast<FlatSymbolRefAttr>());
          assert(sym);
          SymbolicDimOp root = getRootSymbolicDim(sym);
          FlatSymbolRefAttr rootSymbol = FlatSymbolRefAttr::get(root);
          newAttrs.push_back(rootSymbol);
        }
        auto symbolicShapeAttr = ArrayAttr::get(value.getContext(), newAttrs);
        auto newTy = RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                           symbolicShapeAttr);
        value.setType(newTy);
        return success();
      }))) {
    return failure();
  }

  // Update function type
  if (failed(updateFunctionType(m_))) return failure();

  // collect symbolic dim ops that are referred by other ops/types.
  DenseSet<SymbolicDimOp> usedSymbolicOps;
  SmallVector<std::string> usedSymbolNames;
  if (failed(walkRankedTensorValue(m_, [&](Value value, RankedTensorType ty,
                                           ArrayAttr attrs) {
        SmallVector<Attribute> newAttrs;
        for (Attribute attr : attrs) {
          auto sym =
              m_.lookupSymbol<SymbolicDimOp>(attr.cast<FlatSymbolRefAttr>());
          assert(sym);
          if (usedSymbolicOps.insert(sym).second)
            usedSymbolNames.push_back(sym.getName().str());
        }
        return success();
      }))) {
    return failure();
  }

  // remove symbolic dim ops that are known not used by any other ops/types.
  for (auto& p : symbolDimUnionSet_) {
    if (!usedSymbolicOps.count(p.first)) p.first->erase();
  }

  // canonicalize the name of symbolic dim ops
  llvm::sort(usedSymbolNames,
             [&](const std::string& lhs, const std::string& rhs) {
               return compareSymbolicDimOpNames(lhs, rhs);
             });
  std::unordered_map<std::string, std::string> nameMapping;
  for (const auto& en : llvm::enumerate(usedSymbolNames)) {
    nameMapping[en.value()] =
        (llvm::Twine("S") + llvm::Twine(en.index())).str();
  }
  for (SymbolicDimOp op : usedSymbolicOps)
    op.setName(nameMapping[op.getName().str()]);

  // replace the name of a symbolic dim op to its new name.
  if (failed(walkRankedTensorValue(m_, [&](Value value, RankedTensorType ty,
                                           ArrayAttr attrs) {
        SmallVector<Attribute> newAttrs;
        for (Attribute attr : attrs) {
          auto sym = m_.lookupSymbol<SymbolicDimOp>(
              nameMapping[attr.cast<FlatSymbolRefAttr>().getValue().str()]);
          assert(sym);
          FlatSymbolRefAttr newAttr = FlatSymbolRefAttr::get(sym);
          newAttrs.push_back(newAttr);
        }
        auto symbolicShapeAttr = ArrayAttr::get(value.getContext(), newAttrs);
        auto newTy = RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                           symbolicShapeAttr);
        value.setType(newTy);
        return success();
      }))) {
    return failure();
  }

  // Update function type
  if (failed(updateFunctionType(m_))) return failure();

  return success();
}

SmallVector<SymbolicDimOp>
SymbolicDimMgr::getOrCreateSymbolicDimsForRankedValue(Value value) {
  // TODO: load existing symbols from the attribute attached on the tensor type
  SmallVector<SymbolicDimOp> symbols;
  auto ty = value.getType().cast<RankedTensorType>();
  if (auto attrs = value.getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .dyn_cast_or_null<ArrayAttr>()) {
    for (Attribute attr : attrs) {
      auto sym = m_.lookupSymbol<SymbolicDimOp>(attr.cast<FlatSymbolRefAttr>());
      assert(sym);
      symbols.push_back(sym);
    }
    assert(ty.getRank() == symbols.size());
  } else {
    for (int64_t dim : ty.getShape()) {
      symbols.push_back(dim == ShapedType::kDynamicSize
                            ? newSymbolicDim()
                            : newConstantSymbolicDim(dim));
    }
  }

  return symbols;
}

llvm::Optional<SmallVector<FlatSymbolRefAttr>> getRankedValueSymbolicDimRefs(
    Value value) {
  auto ty = value.getType().dyn_cast<RankedTensorType>();
  if (!ty) return {};
  auto attrs = ty.getEncoding().dyn_cast_or_null<ArrayAttr>();
  if (!attrs) return {};
  if (attrs.size() != ty.getRank()) return {};
  SmallVector<FlatSymbolRefAttr> symbols;
  for (const auto& attr : attrs) {
    auto symbol = attr.dyn_cast<FlatSymbolRefAttr>();
    if (!symbol) return {};
    symbols.push_back(symbol);
  }
  return symbols;
}

// Updates the function types according to the types of entry block arguments
// and the types of operands of the return op of the func op. This function
// suppose that there is only one block inside the function region.
LogicalResult updateFunctionType(FuncOp func) {
  if (func.getBody().getBlocks().size() == 0) return success();
  if (func.getBody().getBlocks().size() > 1)
    return func->emitError() << "not support multi-block function\n";
  SmallVector<Type, 4> refinedInputTypes;
  for (Value arg : func.getBody().front().getArguments()) {
    refinedInputTypes.push_back(arg.getType());
  }

  // 2, collect output types
  SmallVector<Type, 4> refinedOutputTypes;
  Operation& op = func.getBody().front().getOperations().back();
  for (Value operand : op.getOperands()) {
    refinedOutputTypes.push_back(operand.getType());
  }

  // 3, refine function type to new type
  auto newFuncTy = FunctionType::get(func.getContext(), refinedInputTypes,
                                     refinedOutputTypes);
  func.setType(newFuncTy);
  return success();
}

// Walk each ranked tensor type values inside op.
LogicalResult walkRankedTensorValue(Operation* op, Visitor visitor) {
  if (op->walk([&](Operation* op) {
          for (Value value : llvm::to_vector(op->getResults())) {
            auto ty = value.getType().dyn_cast<RankedTensorType>();
            if (!ty) continue;
            auto attrs = ty.getEncoding().dyn_cast_or_null<ArrayAttr>();
            if (!attrs) continue;
            if (failed(visitor(value, ty, attrs)))
              return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }).wasInterrupted()) {
    return failure();
  }

  if (op->walk([&](Block* block) {
          for (Value value : llvm::to_vector((block->getArguments()))) {
            auto ty = value.getType().dyn_cast<RankedTensorType>();
            if (!ty) continue;
            auto attrs = ty.getEncoding().dyn_cast_or_null<ArrayAttr>();
            if (!attrs) continue;
            if (failed(visitor(value, ty, attrs)))
              return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }).wasInterrupted()) {
    return failure();
  }
  return success();
}

LogicalResult updateFunctionType(Operation* op) {
  if (op->walk([&](FuncOp func) {
          if (failed(updateFunctionType(func))) return WalkResult::interrupt();
          return WalkResult::advance();
        }).wasInterrupted())
    return failure();
  return success();
}

}  // namespace disc_ral
}  // namespace mlir
