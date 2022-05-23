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

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

using ::mlir::func::FuncOp;

// Gives a consistent order of a list op SymbolicDim Ops
bool compareSymbolicDimOpNames(StringRef lhs, StringRef rhs) {
  // S -> unknown dimension size at compile time
  // C -> constant dimension size at compile time
  // Not known encoding schema, fallback branch
  if (lhs.size() < 1 || lhs[0] != 'S' && lhs[0] != 'C') return lhs < rhs;
  if (rhs.size() < 1 || rhs[0] != 'S' && rhs[0] != 'C') return lhs < rhs;

  int64_t lhsIdx, rhsIdx;
  if (lhs.substr(1).getAsInteger(10, lhsIdx) ||
      rhs.substr(1).getAsInteger(10, rhsIdx))
    return lhs < rhs;
  return (lhs[0] < rhs[0]) || (lhs[0] == rhs[0] && lhsIdx < rhsIdx);
}

// Gives a consistent order of a list op SymbolicDimProducts
bool compareSymbolicDimProduct(const SymbolicDimProduct& lhs,
                               const SymbolicDimProduct& rhs) {
  if (lhs.symbols.size() < rhs.symbols.size()) return true;
  if (lhs.symbols.size() == rhs.symbols.size()) {
    for (const auto& z : llvm::zip(lhs.symbols, rhs.symbols)) {
      StringRef lhsName = const_cast<SymbolicDimOp&>(std::get<0>(z)).getName();
      StringRef rhsName = const_cast<SymbolicDimOp&>(std::get<1>(z)).getName();
      if (compareSymbolicDimOpNames(lhsName, rhsName)) return true;
      if (lhsName != rhsName) return false;
    }
  }
  return false;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const SymbolicDimProduct& product) {
  os << "SymbolicDimProduct[\n\tfactor: " << product.factor << ",\n";
  for (auto& s : product.symbols) os << "\tsymbol: " << s << "\n";
  os << "]\n";
  return os;
}

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) : m_(m) {}

LogicalResult SymbolicDimMgr::load() {
  m_.walk([&](disc_shape::SymbolicDimOp op) {
    symbolDimUnionSet_[op] = op;
    symbolNameSet_.insert(op.getName().str());
  });
  return loadShapeConstraintGraph();
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
    it->second.setName((llvm::Twine("C") + llvm::Twine(val)).str());
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

// Suppose m and n are non-negative integers.
int64_t gcd(int64_t m, int64_t n) {
  if (!m) return n;
  if (!n) return m;
  return (m < n) ? gcd(m, n % m) : gcd(m % n, n);
}

SymbolicDimProduct SymbolicDimMgr::simplifySymbolicDimProduct(
    const SymbolicDimProduct& x) {
  SmallVector<SymbolicDimOp> copied;
  // TODO(disc): expand op if it can be factorized into other symbol dims.
  for (SymbolicDimOp op : x.symbols) copied.push_back(getRootSymbolicDim(op));

  llvm::sort(copied, [&](SymbolicDimOp lhs, SymbolicDimOp rhs) {
    return compareSymbolicDimOpNames(lhs.getName(), rhs.getName());
  });
  SymbolicDimProduct newX;
  newX.factor = x.factor;
  for (SymbolicDimOp op : copied) {
    if (!op.isDynamic()) {
      newX.factor *= op.getDimSize();
    } else {
      newX.symbols.push_back(op);
    }
  }
  return newX;
}

std::pair<SymbolicDimProduct, SymbolicDimProduct>
SymbolicDimMgr::simplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y) {
  // First do some basic clean up (e.g. folding const symbolic dim op into the
  // fator field)
  auto lhs = simplifySymbolicDimProduct(x);
  auto rhs = simplifySymbolicDimProduct(y);

  SymbolicDimProduct newLhs, newRhs;
  int64_t gcdFactor = gcd(std::abs(lhs.factor), std::abs(rhs.factor));
  // 0 * lhs_symbols = 0 * rhs_symbols, no more information, just return empty
  // newLhs & newRhs
  if (!gcdFactor) return std::make_pair(std::move(newLhs), std::move(newRhs));

  // Canonicalization factor form: always let the smaller factor being positive
  // number.
  if (std::abs(lhs.factor) < std::abs(rhs.factor)) {
    if (lhs.factor < 0) gcdFactor = -gcdFactor;
  } else {
    if (rhs.factor < 0) gcdFactor = -gcdFactor;
  }

  newLhs.factor = lhs.factor / gcdFactor;
  newRhs.factor = rhs.factor / gcdFactor;

  DenseSet<SymbolicDimOp> rhsSymbolSet(rhs.symbols.begin(), rhs.symbols.end());
  DenseSet<SymbolicDimOp> nonZeroCommonSymbolSet;
  for (SymbolicDimOp op : lhs.symbols) {
    if (rhsSymbolSet.count(op) && op.knownNonSizeZero()) {
      nonZeroCommonSymbolSet.insert(op);
    } else {
      newLhs.symbols.push_back(op);
    }
  }
  for (SymbolicDimOp op : rhs.symbols) {
    if (nonZeroCommonSymbolSet.count(op)) continue;
    newRhs.symbols.push_back(op);
  }

  if (!newLhs.factor) newLhs.symbols.clear();
  if (!newRhs.factor) newRhs.symbols.clear();

  return std::make_pair(std::move(newLhs), std::move(newRhs));
}

llvm::Optional<SymbolicDimProduct> SymbolicDimMgr::symbolicDimProductDivide(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  LLVM_DEBUG(llvm::dbgs() << "Try to check if x % y == 0?\nx = " << lhs
                          << "y = " << rhs << "\n");
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);
  LLVM_DEBUG(llvm::dbgs() << "Try to check if x % y == 0? after simplify:\nx = "
                          << newLhs << "y = " << newRhs << "\n");

  // early return if any is zero.
  if (newLhs.factor == 0 || newRhs.factor == 0) return {};
  // early return if the const factor is divisible.
  if (newLhs.factor % newRhs.factor != 0) return {};
  if (newLhs.symbols.size() < newRhs.symbols.size()) return {};

  SymbolicDimProduct result;
  result.factor = newLhs.factor / newRhs.factor;

  DenseMap<SymbolicDimOp, int> symProcMap;
  for (SymbolicDimOp sym : newRhs.symbols) ++symProcMap[sym];

  for (SymbolicDimOp sym : newLhs.symbols) {
    auto it = symProcMap.find(sym);
    if (it == symProcMap.end()) {
      result.symbols.push_back(sym);
      continue;
    }
    if (--it->second == 0) {
      symProcMap.erase(it);
      continue;
    }
  }

  if (!symProcMap.empty()) return {};
  LLVM_DEBUG(llvm::dbgs() << "x % y == 0\nx = " << newLhs << "y = " << newRhs
                          << "x / y = " << result << "\n");
  return result;
}

// Try to check if:
//   lhs = common_factors * lhs'
//   rhs = common_factors * rhs'
//   and we already know that product(lhs') == product(rhs')
bool SymbolicDimMgr::isMultipleOfKnownSymbolicDimProductEqualPair(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  for (auto& pairOutter : productEqualityMap_) {
    SymbolicDimProduct& x = pairOutter.first;
    auto factorX = symbolicDimProductDivide(lhs, x);
    if (!factorX) continue;
    for (auto& pairInner : pairOutter.second) {
      if (!pairInner.second) continue;
      SymbolicDimProduct& y = pairInner.first;
      auto factorY = symbolicDimProductDivide(rhs, y);
      if (!factorY || factorX != factorY) continue;
      return true;
    }
  }

  return false;
}

bool SymbolicDimMgr::isSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                               const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (newLhs == newRhs) return true;

  LogicalResult status = updateProductEqualityMap();
  assert(!failed(status));
  return isMultipleOfKnownSymbolicDimProductEqualPair(newLhs, newRhs);
}

LogicalResult SymbolicDimMgr::mapSymbolicDimProductEqual(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  LLVM_DEBUG(llvm::dbgs() << "Try to map product equal: x = " << lhs
                          << "\ny = " << rhs << "\n");
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);
  LLVM_DEBUG(llvm::dbgs() << "Try to map product equal after simplify: x = "
                          << newLhs << "\ny = " << newRhs << "\n");

  // early return for identity case.
  if (newLhs == newRhs) return success();

  if (newLhs.factor == newRhs.factor && newLhs.symbols.size() == 1 &&
      newRhs.symbols.size() == 1) {
    return mapSymbolicDimEqual(newLhs.symbols[0], newRhs.symbols[0]);
  }

  productEqualityMap_[newLhs][newRhs] = productEqualityMap_[newRhs][newLhs] =
      true;

  return updateProductEqualityMap();
}

LogicalResult SymbolicDimMgr::updateProductEqualityMap() {
  SymbolicDimProductMap newMap;
  DenseSet<SymbolicDimProduct> productSet;
  for (auto& pairOutter : productEqualityMap_) {
    SymbolicDimProduct& x = pairOutter.first;
    for (auto& pairInner : pairOutter.second) {
      if (!pairInner.second) continue;
      SymbolicDimProduct& y = pairInner.first;
      SymbolicDimProduct newX, newY;
      std::tie(newX, newY) = simplifySymbolicDimProductPair(x, y);
      if (newX == newY) continue;
      newMap[newX][newY] = newMap[newY][newX] = true;
      productSet.insert(newX);
      productSet.insert(newY);
    }
  }

  bool changed;
  do {
    changed = false;
    for (auto& x : productSet)
      for (auto& y : productSet)
        for (auto& z : productSet) {
          if (x != z && newMap[x][y] && newMap[y][z] && !newMap[x][z]) {
            newMap[x][z] = newMap[z][x] = true;
            changed = true;
          }
        }
  } while (changed);

  productEqualityMap_ = std::move(newMap);

  for (auto& x : productSet)
    for (auto& y : productSet) {
      if (!productEqualityMap_[x][y]) continue;
      productEqualityMap_[x][y] = productEqualityMap_[y][x] = false;
      if (!isMultipleOfKnownSymbolicDimProductEqualPair(x, y)) {
        productEqualityMap_[x][y] = productEqualityMap_[y][x] = true;
      }
    }

  DenseSet<SymbolicDimProduct> toRemove;
  for (auto& x : productSet) {
    if (llvm::all_of(productSet, [&](const SymbolicDimProduct& y) {
          return !productEqualityMap_[x][y];
        })) {
      toRemove.insert(x);
    }
  }
  for (auto& x : toRemove) {
    productEqualityMap_.erase(x);
    LLVM_DEBUG(llvm::dbgs() << "When updateProductEqualityMap try to remove "
                               "redundant symbolicDimProduct "
                            << x << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "After updateProductEqualityMap:\n");
  for (auto& pairOutter : productEqualityMap_) {
    SymbolicDimProduct& x = pairOutter.first;
    for (auto& pairInner : pairOutter.second) {
      if (!pairInner.second) continue;
      SymbolicDimProduct& y = pairInner.first;
      LLVM_DEBUG(llvm::dbgs() << "Pair: x = " << x << "\ny = " << y << "\n");
    }
  }
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
  int numNonConstDims = 0;
  std::unordered_map<std::string, std::string> nameMapping;
  for (const auto& en : llvm::enumerate(usedSymbolNames)) {
    if (en.value().size() > 0 && en.value()[0] == 'C') {
      nameMapping[en.value()] = en.value();
    } else {
      nameMapping[en.value()] =
          (llvm::Twine("S") + llvm::Twine(numNonConstDims++)).str();
    }
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

  // Save shape constraint graph function
  return saveShapeConstraintGraph();
}

/* static */ StringRef SymbolicDimMgr::getShapeConstraintGraphFunctionName() {
  return "shape_constraint_graph";
}

LogicalResult SymbolicDimMgr::saveShapeConstraintGraph() {
  if (failed(updateProductEqualityMap()))
    return m_->emitError() << "fail to update prodcut euqal map\n";

  // first try to remove the old shape constraint graph
  StringRef funcName = getShapeConstraintGraphFunctionName();
  if (auto func = m_.lookupSymbol<FuncOp>(funcName)) func->erase();

  OpBuilder moduleBuilder(m_);
  auto funcTy = FunctionType::get(m_.getContext(), {}, {});
  FuncOp func = moduleBuilder.create<FuncOp>(m_.getLoc(), funcName, funcTy);
  Block* block = func.addEntryBlock();
  OpBuilder b(block, block->begin());
  auto returnOp = b.create<func::ReturnOp>(func.getLoc());
  m_.push_back(func);
  Location loc = func.getLoc();
  b.setInsertionPoint(returnOp);

  // save product equal predicate
  // TODO(disc): return null if the symbolicOps not exists.
  auto build_dim_op = [&](const SymbolicDimProduct& prod) {
    SmallVector<Value> values;
    if (prod.factor != 1) {
      values.push_back(b.create<arith::ConstantIndexOp>(loc, prod.factor));
    }
    for (SymbolicDimOp sym : prod.symbols) {
      values.push_back(b.create<disc_shape::DimOp>(
          loc, b.getIndexType(), FlatSymbolRefAttr::get(sym)));
    }
    return values;
  };
  SmallVector<SymbolicDimProduct> sortedProductVec;
  for (auto& p : productEqualityMap_) sortedProductVec.push_back(p.first);
  llvm::sort(sortedProductVec, compareSymbolicDimProduct);
  for (auto& x : sortedProductVec) {
    for (auto& y : sortedProductVec) {
      if (!compareSymbolicDimProduct(x, y)) continue;
      if (!productEqualityMap_[x][y]) continue;
      auto lhsOperands = build_dim_op(x);
      auto rhsOperands = build_dim_op(y);
      b.create<disc_shape::TieProductEqualOp>(loc, llvm::None, lhsOperands,
                                              rhsOperands);
    }
  }
  return success();
}

LogicalResult SymbolicDimMgr::loadShapeConstraintGraph() {
  StringRef funcName = getShapeConstraintGraphFunctionName();
  auto func = m_.lookupSymbol<FuncOp>(funcName);
  if (!func) return success();

  auto build_sym_product = [&](ValueRange range,
                               SymbolicDimProduct& product) -> LogicalResult {
    for (Value v : range) {
      auto definingOp = v.getDefiningOp();
      if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(definingOp)) {
        product.factor *= constOp->getAttrOfType<IntegerAttr>("value").getInt();
        continue;
      } else if (auto dimOp = dyn_cast_or_null<disc_shape::DimOp>(definingOp)) {
        auto sym = m_.lookupSymbol<SymbolicDimOp>(dimOp.name());
        if (!sym) return dimOp->emitError() << "fail to find symbolic dim op\n";
        product.symbols.push_back(sym);
        continue;
      }
      return failure();
    }
    return success();
  };
  if (func.walk([&](disc_shape::TieProductEqualOp op) {
            SymbolicDimProduct lhs, rhs;
            if (failed(build_sym_product(op.lhs(), lhs)) ||
                failed(build_sym_product(op.rhs(), rhs)) ||
                failed(mapSymbolicDimProductEqual(lhs, rhs)))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return func->emitError() << "fail to load product equal constraint\n";
  }
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

SymbolicDimExpr::SymbolicDimExpr(Value sym)
    : expr(getAffineSymbolExpr(0, sym.getContext())), symbols(1, sym) {}

SymbolicDimExpr::SymbolicDimExpr(int64_t val, MLIRContext* context)
    : expr(getAffineConstantExpr(val, context)) {}

llvm::Optional<int64_t> SymbolicDimExpr::getConstValue() {
  if (auto cstExpr = expr.dyn_cast<AffineConstantExpr>())
    return cstExpr.getValue();
  return {};
}

template <typename Combiner>
/* static */ SymbolicDimExpr SymbolicDimExpr::buildBinaryExpr(
    const SymbolicDimExpr& lhs, const SymbolicDimExpr& rhs,
    Combiner&& combiner) {
  SymbolicDimExpr expr;
  expr.symbols.append(lhs.symbols);
  expr.symbols.append(rhs.symbols);
  expr.expr = combiner(
      lhs.expr, rhs.expr.shiftSymbols(rhs.symbols.size(), lhs.symbols.size()));
  return expr;
}

/* static */ SymbolicDimExpr SymbolicDimExpr::buildMulExpr(
    const SymbolicDimExpr& lhs, const SymbolicDimExpr& rhs) {
  return buildBinaryExpr(lhs, rhs,
                         [](AffineExpr a, AffineExpr b) { return a + b; });
}

/* static */ bool SymbolicDimExpr::isEqual(const SymbolicDimExpr& lhs,
                                           const SymbolicDimExpr& rhs) {
  // TODO(disc): canonicalize the order of symbols of lhs and rhs first
  // TODO(disc): simplify expr of lhs and rhs first
  return lhs.symbols == rhs.symbols && lhs.expr == rhs.expr;
}

}  // namespace disc_ral
}  // namespace mlir
