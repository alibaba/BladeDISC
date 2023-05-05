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

#include "mlir/disc/transforms/disc_shape_optimization_utils.h"

#include <algorithm>
#include <chrono>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

#define DEBUG_TYPE "disc-shape-optimization-utils"
// #define DISC_DEBUG(x) LLVM_DEBUG(x)
#define DISC_DEBUG(x) (x)

#define DEBUG_TIMER(msg)                                                     \
  DISC_DEBUG(end = std::chrono::steady_clock::now());                        \
  DISC_DEBUG(                                                                \
      llvm::dbgs() << msg << " takes: "                                      \
                   << std::chrono::duration_cast<std::chrono::microseconds>( \
                          end - begin)                                       \
                          .count()                                           \
                   << " us\n");                                              \
  DISC_DEBUG(begin = std::chrono::steady_clock::now());

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

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) : m_(m), symbolTable_(m_) {}

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

SymbolicDimOp SymbolicDimMgr::newSymbolicDim(StringRef name) {
  OpBuilder b(m_);
  auto symbol =
      b.create<SymbolicDimOp>(m_.getLoc(), name.empty() ? getNextName() : name);
  symbolDimUnionSet_[symbol] = symbol;
  symbolTable_.insert(symbol);
  return symbol;
}

SymbolicDimOp SymbolicDimMgr::newConstantSymbolicDim(int64_t val) {
  auto it = constantSymbolicDimMap_.find(val);
  if (it == constantSymbolicDimMap_.end()) {
    auto name = (llvm::Twine("C") + llvm::Twine(val)).str();
    it = constantSymbolicDimMap_
             .insert(std::make_pair(val, newSymbolicDim(name)))
             .first;
    it->second.setDimSize(val);
  }
  return getRootSymbolicDim(it->second);
}

SymbolicDimOp SymbolicDimMgr::getRootSymbolicDim(SymbolicDimOp symbol) {
  SymbolicDimOp current = symbol;
  SmallVector<SymbolicDimOp> path;
  while (symbolDimUnionSet_[current] != current) {
    path.push_back(current);
    current = symbolDimUnionSet_[current];
  }
  for (SymbolicDimOp sym : path) symbolDimUnionSet_[sym] = current;
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
    productEqualityMapUpdated_ = false;
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
  copied.reserve(x.symbols.size());
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

  DenseMap<SymbolicDimOp, int> lhsSymbolMap;
  DenseMap<SymbolicDimOp, int> rhsSymbolMap;
  for (SymbolicDimOp op : lhs.symbols) ++lhsSymbolMap[op];
  for (SymbolicDimOp op : rhs.symbols) ++rhsSymbolMap[op];

  for (SymbolicDimOp op : lhs.symbols) {
    auto it = rhsSymbolMap.find(op);
    if (it != rhsSymbolMap.end() && op.getKnownNonSizeZero()) {
      if (--it->second == 0) rhsSymbolMap.erase(it);
      continue;
    }
    newLhs.symbols.push_back(op);
  }

  for (SymbolicDimOp op : rhs.symbols) {
    auto it = lhsSymbolMap.find(op);
    if (it != lhsSymbolMap.end() && op.getKnownNonSizeZero()) {
      if (--it->second == 0) lhsSymbolMap.erase(it);
      continue;
    }
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
  } else if (newLhs.symbols.size() == 0 && newRhs.symbols.size() == 1 &&
             newRhs.factor == 1) {
    return mapSymbolicDimEqual(newConstantSymbolicDim(newLhs.factor),
                               newRhs.symbols[0]);
  } else if (newRhs.symbols.size() == 0 && newLhs.symbols.size() == 1 &&
             newLhs.factor == 1) {
    return mapSymbolicDimEqual(newConstantSymbolicDim(newRhs.factor),
                               newLhs.symbols[0]);
  }

  productEqualityMap_[newLhs][newRhs] = productEqualityMap_[newRhs][newLhs] =
      true;

  productEqualityMapUpdated_ = false;
  return success();
}

LogicalResult SymbolicDimMgr::updateProductEqualityMap() {
  // early return if nothing is updated.
  if (productEqualityMapUpdated_) return success();

  std::chrono::steady_clock::time_point begin, end;
  DISC_DEBUG(begin = std::chrono::steady_clock::now());
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
  DEBUG_TIMER(
      "SymbolicDimMgr::updateProductEqualityMap "
      "simplifySymbolicDimProductPair");

  // hash function of SymbolicDimProduct is expensive, thus we map it to integer
  // domain first.
  DenseMap<SymbolicDimProduct*, size_t> symProd2Idx;
  SmallVector<SymbolicDimProduct*> idx2SymProd(productSet.size());
  SmallVector<size_t> idx2root(productSet.size());
  for (auto& x : productSet) {
    size_t idx = symProd2Idx.size();
    symProd2Idx[&x] = idx;
    idx2SymProd[idx] = &x;
    idx2root[idx] = idx;
  }

  auto getRootIdx = [&](size_t root) {
    SmallVector<size_t> path;
    while (idx2root[root] != root) {
      path.push_back(root);
      root = idx2root[root];
    }
    for (size_t idx : path) idx2root[idx] = root;
    return root;
  };

  for (size_t x = 0; x < symProd2Idx.size(); ++x) {
    auto& xProd = *idx2SymProd[x];
    auto& rowMap = newMap[xProd];
    size_t xRoot = getRootIdx(x);
    for (size_t y = x; y < symProd2Idx.size(); ++y) {
      auto& yProd = *idx2SymProd[y];
      if (!rowMap[yProd]) continue;
      idx2root[getRootIdx(y)] = xRoot;
    }
  }

  for (size_t x = 0; x < symProd2Idx.size(); ++x)
    for (size_t y = x; y < symProd2Idx.size(); ++y) {
      if (getRootIdx(x) != getRootIdx(y)) continue;
      auto& xSymProd = *idx2SymProd[x];
      auto& ySymProd = *idx2SymProd[y];

      newMap[xSymProd][ySymProd] = newMap[ySymProd][xSymProd] = true;
    }

  DISC_DEBUG(llvm::dbgs() << "productSet.size() = " << productSet.size()
                          << "\n");
  DEBUG_TIMER("SymbolicDimMgr::updateProductEqualityMap propagate graph");

  productEqualityMap_ = std::move(newMap);

  for (auto& x : productSet)
    for (auto& y : productSet) {
      if (!productEqualityMap_[x][y]) continue;
      productEqualityMap_[x][y] = productEqualityMap_[y][x] = false;
      if (!isMultipleOfKnownSymbolicDimProductEqualPair(x, y)) {
        productEqualityMap_[x][y] = productEqualityMap_[y][x] = true;
      }
    }
  DEBUG_TIMER("SymbolicDimMgr::updateProductEqualityMap remove multiply");

  DenseSet<SymbolicDimProduct> toRemove;
  for (auto& x : productSet) {
    if (llvm::all_of(productSet, [&](const SymbolicDimProduct& y) {
          return !productEqualityMap_[x][y];
        })) {
      toRemove.insert(x);
    }
  }
  DEBUG_TIMER("SymbolicDimMgr::updateProductEqualityMap build toRemove ");
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
  DEBUG_TIMER("SymbolicDimMgr::updateProductEqualityMap apply toRemove ");
  productEqualityMapUpdated_ = true;
  return success();
}

LogicalResult SymbolicDimMgr::save() {
  std::chrono::steady_clock::time_point begin, end;
  DISC_DEBUG(begin = std::chrono::steady_clock::now());
  // replace all uses of a symbolic dim op with its root symbolic dim op
  using Name2SymbolFn = std::function<SymbolicDimOp(StringRef)>;
  auto updateAttrs = [&](ArrayAttr attrs, Name2SymbolFn fn) {
    SmallVector<Attribute> newAttrs;
    for (Attribute attr : attrs) {
      auto sym = fn(attr.cast<FlatSymbolRefAttr>().getValue());
      assert(sym);
      SymbolicDimOp root = getRootSymbolicDim(sym);
      FlatSymbolRefAttr rootSymbol = FlatSymbolRefAttr::get(root);
      newAttrs.push_back(rootSymbol);
    }
    return ArrayAttr::get(m_->getContext(), newAttrs);
  };
  // update attributes attached in values.
  if (failed(walkRankedTensorValue(
          m_, [&](Value value, RankedTensorType ty, ArrayAttr attrs) {
            auto symbolicShapeAttr = updateAttrs(attrs, [&](StringRef name) {
              return symbolTable_.lookup<SymbolicDimOp>(name);
            });
            auto newTy = RankedTensorType::get(
                ty.getShape(), ty.getElementType(), symbolicShapeAttr);
            value.setType(newTy);
            return success();
          }))) {
    return failure();
  }
  DEBUG_TIMER("SymbolicDimMgr::save walkRankedTensorValue");

  // update attributes attached in operations.
  m_.walk([&](Operation* op) {
    auto attrs =
        op->getAttrOfType<ArrayAttr>(SymbolicDimOp::getSymbolicDimAttrName());
    if (!attrs) return;
    auto symbolicShapeAttr = updateAttrs(attrs, [&](StringRef name) {
      return symbolTable_.lookup<SymbolicDimOp>(name);
    });
    op->setAttr(SymbolicDimOp::getSymbolicDimAttrName(), symbolicShapeAttr);
  });
  DEBUG_TIMER("SymbolicDimMgr::save update attributes");
  // update shape production equality
  if (failed(updateProductEqualityMap()))
    return m_->emitError() << "fail to update prodcut euqal map\n";
  DEBUG_TIMER("SymbolicDimMgr::save updateProductEqualityMap");

  // Update function type
  if (failed(updateFunctionType(m_))) return failure();
  DEBUG_TIMER("SymbolicDimMgr::save updateFunctionType");

  // collect symbolic dim ops that are referred by other ops/types.
  DenseSet<SymbolicDimOp> usedSymbolicOps;
  SmallVector<std::string> usedSymbolNames;
  // collect uses in values.
  auto collectUsedSymbols = [&](ArrayAttr attrs) {
    for (Attribute attr : attrs) {
      auto sym = symbolTable_.lookup<SymbolicDimOp>(
          attr.cast<FlatSymbolRefAttr>().getValue());
      assert(sym);
      if (usedSymbolicOps.insert(sym).second)
        usedSymbolNames.push_back(sym.getName().str());
    }
  };
  if (failed(walkRankedTensorValue(
          m_, [&](Value value, RankedTensorType ty, ArrayAttr attrs) {
            collectUsedSymbols(attrs);
            return success();
          }))) {
    return failure();
  }
  // collect uses in operations.
  m_.walk([&](Operation* op) {
    auto attrs =
        op->getAttrOfType<ArrayAttr>(SymbolicDimOp::getSymbolicDimAttrName());
    if (!attrs) return;
    collectUsedSymbols(attrs);
  });
  DEBUG_TIMER("SymbolicDimMgr::save collect symbolicDim ops");

  // remove symbolic dim ops that are known not used by any other ops/types.
  for (auto& p : symbolDimUnionSet_) {
    if (!usedSymbolicOps.count(p.first)) p.first->erase();
  }
  DEBUG_TIMER("SymbolicDimMgr::save remove symbolicDim ops");

  // remove unused shape production equality
  SmallVector<SymbolicDimProduct> candidates;
  for (auto& outter : productEqualityMap_) {
    if (llvm::any_of(outter.first.symbols, [&](SymbolicDimOp sym) {
          return usedSymbolicOps.count(sym) == 0;
        }))
      candidates.push_back(outter.first);
  }
  for (auto& prod : candidates) productEqualityMap_.erase(prod);
  DEBUG_TIMER("SymbolicDimMgr::save remove unused production");

  for (auto& outter : productEqualityMap_) {
    SmallVector<SymbolicDimProduct> candidates;
    for (auto& inner : outter.second) {
      if (llvm::any_of(inner.first.symbols, [&](SymbolicDimOp sym) {
            return usedSymbolicOps.count(sym) == 0;
          }))
        candidates.push_back(outter.first);
    }
    for (auto& prod : candidates) outter.second.erase(prod);
  }
  DEBUG_TIMER("SymbolicDimMgr::save remove unused production #2");

  // canonicalize the name of symbolic dim ops
  llvm::sort(usedSymbolNames,
             [&](const std::string& lhs, const std::string& rhs) {
               return compareSymbolicDimOpNames(lhs, rhs);
             });
  int numNonConstDims = 0;
  std::unordered_map<std::string, std::string> nameMapping;
  for (const auto& name : usedSymbolNames) {
    if (name.size() > 0 && name[0] == 'C') {
      nameMapping[name] = name;
    } else {
      nameMapping[name] =
          (llvm::Twine("S") + llvm::Twine(numNonConstDims++)).str();
    }
  }
  std::unordered_map<std::string, SymbolicDimOp> name2Symbol;
  for (SymbolicDimOp op : usedSymbolicOps) {
    auto name = op.getName().str();
    op.setName(nameMapping[name]);
    name2Symbol[name] = op;
  }
  DEBUG_TIMER("SymbolicDimMgr::save canonicalize the name");

  // replace the name of a symbolic dim op to its new name.
  // update attributes attached to operations.
  m_.walk([&](Operation* op) {
    auto attrs =
        op->getAttrOfType<ArrayAttr>(SymbolicDimOp::getSymbolicDimAttrName());
    if (!attrs) return;
    auto symbolicShapeAttr = updateAttrs(
        attrs, [&](StringRef name) { return name2Symbol[name.str()]; });
    op->setAttr(SymbolicDimOp::getSymbolicDimAttrName(), symbolicShapeAttr);
  });
  // update attributes attached to values.
  if (failed(walkRankedTensorValue(
          m_, [&](Value value, RankedTensorType ty, ArrayAttr attrs) {
            auto symbolicShapeAttr = updateAttrs(
                attrs, [&](StringRef name) { return name2Symbol[name.str()]; });
            auto newTy = RankedTensorType::get(
                ty.getShape(), ty.getElementType(), symbolicShapeAttr);
            value.setType(newTy);
            return success();
          }))) {
    return failure();
  }
  DEBUG_TIMER("SymbolicDimMgr::save replace the name");

  // Update function type
  if (failed(updateFunctionType(m_))) return failure();

  DEBUG_TIMER("SymbolicDimMgr::save updateFunctionType");

  // Save shape constraint graph function
  return saveShapeConstraintGraph();
}

/* static */ StringRef SymbolicDimMgr::getShapeConstraintGraphFunctionName() {
  return "shape_constraint_graph";
}

LogicalResult SymbolicDimMgr::saveShapeConstraintGraph() {
  // first try to remove the old shape constraint graph
  StringRef funcName = getShapeConstraintGraphFunctionName();
  if (auto func = symbolTable_.lookup<FuncOp>(funcName)) func->erase();

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
  auto build_operands = [&](const SymbolicDimProduct& prod) {
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
      auto lhsOperands = build_operands(x);
      auto rhsOperands = build_operands(y);
      b.create<disc_shape::TieProductEqualOp>(loc, llvm::None, lhsOperands,
                                              rhsOperands);
    }
  }
  return success();
}

LogicalResult SymbolicDimMgr::loadShapeConstraintGraph() {
  StringRef funcName = getShapeConstraintGraphFunctionName();
  auto func = symbolTable_.lookup<FuncOp>(funcName);
  if (!func) return success();

  auto build_sym_product = [&](ValueRange range,
                               SymbolicDimProduct& product) -> LogicalResult {
    for (Value v : range) {
      auto definingOp = v.getDefiningOp();
      if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(definingOp)) {
        product.factor *= constOp->getAttrOfType<IntegerAttr>("value").getInt();
        continue;
      } else if (auto dimOp = dyn_cast_or_null<disc_shape::DimOp>(definingOp)) {
        auto sym = symbolTable_.lookup<SymbolicDimOp>(dimOp.getName());
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
            if (failed(build_sym_product(op.getLhs(), lhs)) ||
                failed(build_sym_product(op.getRhs(), rhs)) ||
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
      auto sym = symbolTable_.lookup<SymbolicDimOp>(
          attr.cast<FlatSymbolRefAttr>().getValue());
      assert(sym);
      symbols.push_back(sym);
    }
    assert(ty.getRank() == symbols.size());
  } else {
    for (int64_t dim : ty.getShape()) {
      symbols.push_back(dim == ShapedType::kDynamic
                            ? newSymbolicDim()
                            : newConstantSymbolicDim(dim));
    }
  }

  return symbols;
}

SymbolicDimOp SymbolicDimMgr::cloneSymbol(SymbolicDimOp symbol) {
  // early return for static shape symbol
  if (!symbol.isDynamic()) return symbol;
  SymbolicDimOp newSymbol = newSymbolicDim();

  // copied all attributes expect the name.
  for (const NamedAttribute& attr : symbol->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "sym_name") continue;
    newSymbol->setAttr(name, attr.getValue());
  }
  return newSymbol;
}

LogicalResult SymbolicDimMgr::cloneSymbolGroup(
    const DenseSet<SymbolicDimOp>& symbols,
    DenseMap<SymbolicDimOp, SymbolicDimOp>& mapping) {
  DenseMap<SymbolicDimOp, SmallVector<SymbolicDimOp>> root2Symbol;
  for (SymbolicDimOp symbol : symbols) {
    SymbolicDimOp newSymbol = cloneSymbol(symbol);
    mapping[symbol] = newSymbol;
    SymbolicDimOp root = getRootSymbolicDim(symbol);
    root2Symbol[root].push_back(newSymbol);
  }

  // copy dim equality predicate.
  for (auto& it : root2Symbol) {
    if (it.second.size() <= 1) continue;
    for (SymbolicDimOp symbol : it.second)
      if (failed(mapSymbolicDimEqual(symbol, it.second[0]))) return failure();
  }

  // copy dim product equality predicate
  SmallVector<std::pair<SymbolicDimProduct, SymbolicDimProduct>> candidates;
  for (auto& outter : productEqualityMap_) {
    if (llvm::any_of(outter.first.symbols, [&](SymbolicDimOp sym) {
          return symbols.count(sym) == 0;
        }))
      continue;
    for (auto& inner : outter.second) {
      if (!inner.second) continue;
      if (llvm::any_of(inner.first.symbols, [&](SymbolicDimOp sym) {
            return symbols.count(sym) == 0;
          }))
        continue;
      auto copySymbolicDimProduct = [&](SymbolicDimProduct& prod) {
        SymbolicDimProduct newProd;
        newProd.factor = prod.factor;
        for (SymbolicDimOp sym : prod.symbols) {
          newProd.symbols.push_back(mapping[sym]);
        }
        return newProd;
      };
      candidates.emplace_back(copySymbolicDimProduct(outter.first),
                              copySymbolicDimProduct(inner.first));
    }
  }

  for (auto& it : candidates) {
    productEqualityMap_[it.first][it.second] = true;
    productEqualityMap_[it.second][it.first] = true;
  }

  return success();
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

llvm::Optional<SmallVector<FlatSymbolRefAttr>> getMemRefValueSymbolicDimRefs(
    Value value) {
  auto ty = value.getType().dyn_cast<MemRefType>();
  Operation* op = value.getDefiningOp();
  if (!ty || !op) return {};
  auto attrs = op->getAttrOfType<ArrayAttr>(
      disc_shape::SymbolicDimOp::getSymbolicDimAttrName());
  if (!attrs || attrs.size() != ty.getRank()) return {};

  SmallVector<FlatSymbolRefAttr> symbols;
  symbols.reserve(attrs.size());
  for (const auto& attr : attrs) {
    auto symbol = attr.dyn_cast<FlatSymbolRefAttr>();
    if (!symbol) return {};
    symbols.push_back(symbol);
  }
  return symbols;
}

ArrayAttr makeSymbolicDimOpRefArrayAttr(
    const SmallVector<SymbolicDimOp>& symbols) {
  assert(!symbols.empty());
  SmallVector<Attribute> attrs;
  attrs.reserve(symbols.size());
  for (auto& sym : symbols) {
    attrs.push_back(FlatSymbolRefAttr::get(sym));
  }
  return ArrayAttr::get(symbols[0]->getContext(), attrs);
}

void attachSymbolicDimOpRefArrayAttrOnOperation(
    Operation* op, const SmallVector<SymbolicDimOp>& symbols) {
  assert(op != nullptr);
  auto symbolicShapeAttr = makeSymbolicDimOpRefArrayAttr(symbols);
  op->setAttr(SymbolicDimOp::getSymbolicDimAttrName(), symbolicShapeAttr);
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
                         [](AffineExpr a, AffineExpr b) { return a * b; });
}

/* static */ bool SymbolicDimExpr::isEqual(const SymbolicDimExpr& lhs,
                                           const SymbolicDimExpr& rhs) {
  // TODO(disc): canonicalize the order of symbols of lhs and rhs first
  // TODO(disc): simplify expr of lhs and rhs first
  return lhs.symbols == rhs.symbols && lhs.expr == rhs.expr;
}

SliceOpShapeHelper::SliceOpShapeHelper(Operation* op) : op(op) {
  // mhlo.real_dynamic_slice has four operands.
  // lmhlo.real_dynamic_slice has five oeprands, the last one is the output
  // operand.
  assert(op->getNumOperands() >= 4);
  auto ty = op->getOperand(0).getType().cast<ShapedType>();
  startIndices = SmallVector<int64_t>(ty.getRank(), ShapeValueState::kUnknown);
  limitIndices = SmallVector<int64_t>(ty.getRank(), ShapeValueState::kUnknown);
  strides = SmallVector<int64_t>(ty.getRank(), ShapeValueState::kUnknown);

  auto attr = op->getAttrOfType<DictionaryAttr>(getAttrName());
  if (!attr) return;
  auto startAttr = attr.getAs<DenseIntElementsAttr>("start_indices");
  auto limitAttr = attr.getAs<DenseIntElementsAttr>("limit_indices");
  auto strideAttr = attr.getAs<DenseIntElementsAttr>("strides");
  assert(startAttr.getNumElements() == ty.getRank());
  assert(limitAttr.getNumElements() == ty.getRank());
  assert(strideAttr.getNumElements() == ty.getRank());

  for (int i = 0; i < ty.getRank(); ++i) {
    mergeStartIndex(i, startAttr.getValues<int64_t>()[i]);
    mergeLimitIndex(i, limitAttr.getValues<int64_t>()[i]);
    mergeStride(i, strideAttr.getValues<int64_t>()[i]);
  }
}

bool SliceOpShapeHelper::isFullySlicedAxis(int axis) {
  assert(axis < static_cast<int>(startIndices.size()));
  return startIndices[axis] == 0 && strides[axis] == 1 &&
         limitIndices[axis] == ShapeValueState::kLimitIsDimSize;
}

LogicalResult SliceOpShapeHelper::markAsFullySlicedAxis(int axis) {
  assert(axis < static_cast<int>(startIndices.size()));

  if (failed(mergeStartIndex(axis, 0)))
    return op->emitError() << "fail to update start index for axis " << axis
                           << "\n";

  if (failed(mergeLimitIndex(axis, ShapeValueState::kLimitIsDimSize)))
    return op->emitError() << "fail to update limit index for axis " << axis
                           << "\n";

  if (failed(mergeStride(axis, 1)))
    return op->emitError() << "fail to update stride index for axis " << axis
                           << "\n";

  return success();
}

LogicalResult SliceOpShapeHelper::mergeStartIndex(int axis, int64_t value) {
  assert(axis < static_cast<int>(startIndices.size()));

  if (startIndices[axis] != value && value != ShapeValueState::kUnknown &&
      startIndices[axis] != ShapeValueState::kUnknown)
    return failure();

  if (startIndices[axis] == ShapeValueState::kUnknown)
    startIndices[axis] = value;
  return success();
}

LogicalResult SliceOpShapeHelper::mergeLimitIndex(int axis, int64_t value) {
  assert(axis < static_cast<int>(limitIndices.size()));

  if (limitIndices[axis] == ShapeValueState::kLimitIsDimSize ||
      value == ShapeValueState::kLimitIsDimSize) {
    limitIndices[axis] = ShapeValueState::kLimitIsDimSize;
    return success();
  }

  if (limitIndices[axis] != value && value != ShapeValueState::kUnknown &&
      limitIndices[axis] != ShapeValueState::kUnknown)
    return failure();

  if (limitIndices[axis] == ShapeValueState::kUnknown)
    limitIndices[axis] = value;
  return success();
}

LogicalResult SliceOpShapeHelper::mergeStride(int axis, int64_t value) {
  assert(axis < static_cast<int>(strides.size()));

  if (strides[axis] != value && value != ShapeValueState::kUnknown &&
      strides[axis] != ShapeValueState::kUnknown)
    return failure();

  if (strides[axis] == ShapeValueState::kUnknown) strides[axis] = value;
  return success();
}

LogicalResult SliceOpShapeHelper::save() {
  OpBuilder b(op);
  auto startAttr = GetI64ElementsAttr(startIndices, &b);
  auto limitAttr = GetI64ElementsAttr(limitIndices, &b);
  auto strideAttr = GetI64ElementsAttr(strides, &b);

  SmallVector<mlir::NamedAttribute, 4> elemAttrs;
  elemAttrs.emplace_back(b.getNamedAttr("start_indices", startAttr));
  elemAttrs.emplace_back(b.getNamedAttr("limit_indices", limitAttr));
  elemAttrs.emplace_back(b.getNamedAttr("strides", strideAttr));

  op->setAttr(getAttrName(), b.getDictionaryAttr(elemAttrs));
  return success();
}

PadOpShapeHelper::PadOpShapeHelper(Operation* op) : op(op) {
  // mhlo.dynamic_pad has five operands.
  // lmhlo.dynamic_pad has six oeprands, the last one is the output operand.
  assert(op->getNumOperands() >= 5);
  auto ty = op->getOperand(0).getType().cast<ShapedType>();
  edgePaddingLows =
      SmallVector<int64_t>(ty.getRank(), ShapeValueState::kUnknown);
  edgePaddingHighs =
      SmallVector<int64_t>(ty.getRank(), ShapeValueState::kUnknown);
  interiorPaddings =
      SmallVector<int64_t>(ty.getRank(), ShapeValueState::kUnknown);

  auto attr = op->getAttrOfType<DictionaryAttr>(getAttrName());
  if (!attr) return;
  auto lowAttr = attr.getAs<DenseIntElementsAttr>("edge_padding_low");
  auto highAttr = attr.getAs<DenseIntElementsAttr>("edge_padding_high");
  auto interiorAttr = attr.getAs<DenseIntElementsAttr>("interior_padding");
  assert(lowAttr.getNumElements() == ty.getRank());
  assert(highAttr.getNumElements() == ty.getRank());
  assert(interiorAttr.getNumElements() == ty.getRank());

  for (int i = 0; i < ty.getRank(); ++i) {
    mergeEdgePaddingLow(i, lowAttr.getValues<int64_t>()[i]);
    mergeEdgePaddingHigh(i, highAttr.getValues<int64_t>()[i]);
    mergeInteriorPadding(i, interiorAttr.getValues<int64_t>()[i]);
  }
}

bool PadOpShapeHelper::isNotPaddedAxis(int axis) {
  assert(axis < static_cast<int>(edgePaddingLows.size()));
  assert(axis < static_cast<int>(edgePaddingHighs.size()));
  assert(axis < static_cast<int>(interiorPaddings.size()));
  return edgePaddingLows[axis] == 0 && edgePaddingHighs[axis] == 0 &&
         interiorPaddings[axis] == 0;
}

LogicalResult PadOpShapeHelper::markAsNotPaddedAxis(int axis) {
  assert(axis < static_cast<int>(edgePaddingLows.size()));
  assert(axis < static_cast<int>(edgePaddingHighs.size()));
  assert(axis < static_cast<int>(interiorPaddings.size()));

  if (failed(mergeEdgePaddingLow(axis, 0)))
    return op->emitError() << "fail to update low index for axis " << axis
                           << "\n";

  if (failed(mergeEdgePaddingHigh(axis, 0)))
    return op->emitError() << "fail to update high index for axis " << axis
                           << "\n";

  if (failed(mergeInteriorPadding(axis, 0)))
    return op->emitError() << "fail to update interior index for axis " << axis
                           << "\n";

  return success();
}

LogicalResult PadOpShapeHelper::mergeEdgePaddingLow(int axis, int64_t value) {
  assert(axis < static_cast<int>(edgePaddingLows.size()));

  if (edgePaddingLows[axis] != value && value != ShapeValueState::kUnknown &&
      edgePaddingLows[axis] != ShapeValueState::kUnknown)
    return failure();

  if (edgePaddingLows[axis] == ShapeValueState::kUnknown)
    edgePaddingLows[axis] = value;
  return success();
}

LogicalResult PadOpShapeHelper::mergeEdgePaddingHigh(int axis, int64_t value) {
  assert(axis < static_cast<int>(edgePaddingHighs.size()));

  if (edgePaddingHighs[axis] != value && value != ShapeValueState::kUnknown &&
      edgePaddingHighs[axis] != ShapeValueState::kUnknown)
    return failure();

  if (edgePaddingHighs[axis] == ShapeValueState::kUnknown)
    edgePaddingHighs[axis] = value;
  return success();
}

LogicalResult PadOpShapeHelper::mergeInteriorPadding(int axis, int64_t value) {
  assert(axis < static_cast<int>(interiorPaddings.size()));

  if (interiorPaddings[axis] != value && value != ShapeValueState::kUnknown &&
      interiorPaddings[axis] != ShapeValueState::kUnknown)
    return failure();

  if (interiorPaddings[axis] == ShapeValueState::kUnknown)
    interiorPaddings[axis] = value;
  return success();
}

LogicalResult PadOpShapeHelper::save() {
  OpBuilder b(op);
  auto lowAttr = GetI64ElementsAttr(edgePaddingLows, &b);
  auto highAttr = GetI64ElementsAttr(edgePaddingHighs, &b);
  auto interiorAttr = GetI64ElementsAttr(interiorPaddings, &b);

  SmallVector<mlir::NamedAttribute, 4> elemAttrs;
  elemAttrs.emplace_back(b.getNamedAttr("edge_padding_low", lowAttr));
  elemAttrs.emplace_back(b.getNamedAttr("edge_padding_high", highAttr));
  elemAttrs.emplace_back(b.getNamedAttr("interior_padding", interiorAttr));

  op->setAttr(getAttrName(), b.getDictionaryAttr(elemAttrs));
  return success();
}

llvm::Optional<SmallVector<SymbolicDimOp>> getMemRefValueSymbolicDims(
    SymbolicDimMgr& mgr, Value value) {
  auto dimAttrs = getMemRefValueSymbolicDimRefs(value);
  if (!dimAttrs) return llvm::None;

  SmallVector<SymbolicDimOp> syms;
  for (auto& attr : *dimAttrs) {
    syms.push_back(mgr.getRootSymbolicDim(
        mgr.symbolTable().lookup<SymbolicDimOp>(attr.getValue())));
  }
  return syms;
}

}  // namespace disc_ral
}  // namespace mlir
