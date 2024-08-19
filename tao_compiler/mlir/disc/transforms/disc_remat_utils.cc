// Copyright 2024 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/disc/transforms/disc_remat_utils.h"

#include <cmath>

#include "absl/container/flat_hash_map.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
namespace mlir {
namespace disc_ral {
bool IsHostBuffer(Value value) {
  auto memRefType = value.getType().cast<MemRefType>();
  return memRefType.getMemorySpace() == 0;
}

LogicalResult DiscBufferLivingRange::Analysis() {
  buffer_list_.clear();
  living_buffers_.clear();
  int64_t position;
  buffer_map_.clear();
  auto reccordOpWithPosition = [&](Value value, Operation* op) {
    buffer_map_[value].push_back(std::make_pair(op, position++));
  };
  main_.walk([&](Operation* op) {
    // Traverse the function's blocks and operations.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      auto buffer = allocOp.getResult();
      if (IsHostBuffer(allocOp.getResult())) {
        return;
      }
      buffer_list_.push_back(buffer);
      reccordOpWithPosition(buffer, op);
    } else if (isa<memref::DeallocOp>(op)) {
      auto buffer = op->getOperand(0);
      if (IsHostBuffer(buffer)) {
        return;
      }
      reccordOpWithPosition(buffer, op);
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto operand : returnOp.getOperands()) {
        reccordOpWithPosition(operand, op);
      }
    } else if (isa<lmhlo_disc::H2DOp>(op) || isa<lmhlo_disc::D2HOp>(op) ||
               isa<scf::YieldOp>(op) || isa<memref::DimOp>(op)) {
      return;
    } else {
      for (Value operand : op->getOperands()) {
        if (buffer_map_.count(operand)) reccordOpWithPosition(operand, op);
      }
    }
  });
  for (auto iter : buffer_map_) {
    auto buffer = iter.first;
    for (size_t i = 1; i < iter.second.size(); i++) {
      auto start = iter.second[i - 1].first;
      auto startPosition = iter.second[i - 1].second;

      auto end = iter.second[i].first;
      auto endPosition = iter.second[i].second;
      LivingBuffer livingBuffer(start, startPosition, end, endPosition, buffer);
      living_buffers_.push_back(livingBuffer);
    }
  }
  return success();
}
std::vector<OpWithPosition> DiscBufferLivingRange::GetUsersOrderByPosition(
    Value buffer) {
  std::vector<OpWithPosition> ops;
  if (buffer_map_.find(buffer) == buffer_map_.end()) {
    return ops;
  }
  return buffer_map_[buffer];
}

//////////////// Symbolic Memory Profiler Utils //////////////////
bool hasDynamicDimension(Value buffer) {
  auto memrefType = buffer.getType().cast<MemRefType>();
  for (auto dim : memrefType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      return true;
    }
  }
  return false;
}
int64_t getElementSize(Type elementType) {
  if (elementType.isF32()) {
    return sizeof(float);
  } else if (elementType.isF16()) {
    return sizeof(uint16_t);
  } else if (elementType.isBF16()) {
    return sizeof(uint16_t);
  } else if (elementType.isF64()) {
    return sizeof(double);
  } else if (elementType.isInteger(1)) {
    return sizeof(bool);
  } else if (elementType.isInteger(8)) {
    return sizeof(int8_t);
  } else if (elementType.isInteger(16)) {
    return sizeof(int16_t);
  } else if (elementType.isInteger(32)) {
    return sizeof(int32_t);
  } else if (elementType.isInteger(64)) {
    return sizeof(int64_t);
  } else if (elementType.isIndex()) {
    return sizeof(int32_t);
  } else {
    llvm::dbgs() << elementType << "\n";
    // Add more types as needed
    llvm::errs() << "Unsupported element type\n";
    return -1;
  }
}

MemoryUsage& operator+=(MemoryUsage& lhs, const SymbolicDimProduct& rhs) {
  MemoryUsage result;
  bool mergeSymbols = false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].symbols == rhs.symbols) {
      mergeSymbols = true;
      lhs[i].factor += rhs.factor;
    }
  }
  if (!mergeSymbols) {
    lhs.push_back(rhs);
  }
  return lhs;
}
MemoryUsage& operator-=(MemoryUsage& lhs, const SymbolicDimProduct& rhs) {
  bool mergeSymbols = false;
  size_t removeIndex = -1;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].symbols == rhs.symbols) {
      mergeSymbols = true;
      lhs[i].factor -= rhs.factor;
      if (lhs[i].factor == 0) {
        removeIndex = i;
        break;
      }
    }
  }
  if (removeIndex != -1) {
    lhs.erase(lhs.begin() + removeIndex);
  }
  if (!mergeSymbols) {
    auto newRhs = rhs;
    newRhs.factor *= -1;
    lhs.push_back(newRhs);
  }
  return lhs;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const MemoryUsage& memoryUsage) {
  for (size_t i = 0; i < memoryUsage.size(); ++i) {
    // print prod
    os << memoryUsage[i].factor;
    if (memoryUsage[i].symbols.size() > 0) os << "*";
    for (size_t j = 0; j < memoryUsage[i].symbols.size(); ++j) {
      auto dimOp = memoryUsage[i].symbols[j];
      if (j != memoryUsage[i].symbols.size() - 1) {
        os << dimOp.getName() << "*";
      } else {
        os << dimOp.getName();
      }
    }
    if (i != memoryUsage.size() - 1 && memoryUsage[i].factor > 0) {
      os << " + ";
    }
  }
  return os;
}
// TODO(yancey): just a PoC implementation, need to implement this function
// by traveling the shape operators
SymbolicDimProduct SimplifySymbolicDims(SymbolicDimProduct proudct,
                                        SymbolicDimOp s0) {
  auto factor = proudct.factor;
  SmallVector<SymbolicDimOp, 6> symbols;
  for (auto symbol : proudct.symbols) {
    StringRef symbolName = const_cast<SymbolicDimOp&>(symbol).getName();
    if (symbolName.startswith("S")) {
      if (symbolName == "S0") {
        factor *= 1;
      } else if (symbolName == "S1") {
        factor *= 4;
      } else if (symbolName == "S2") {
        factor *= 32001;
      } else if (symbolName == "S3") {
        factor *= 128;
      } else {
        llvm::errs() << "unsupported symbol name\n";
      }
      symbols.push_back(s0);
    } else {
      symbols.push_back(symbol);
    }
  }
  return SymbolicDimProduct{symbols, factor};
}
int64_t getMemRefSize(Value value) {
  auto memRefType = value.getType().cast<MemRefType>();
  int64_t elementSize = getElementSize(memRefType.getElementType());
  if (elementSize < 0) {
    return -1;  // Unsupported type
  }

  int64_t numElements = 1;
  for (int64_t dim : memRefType.getShape()) {
    numElements *= dim;
  }
  return numElements * elementSize;
}
SymbolicDimProduct getSymbolicMemrefBytes(Value buffer, SymbolicDimMgr* mgr) {
  auto s0 = mgr->findSymbolicDimOp("S0");
  auto ty = buffer.getType().cast<MemRefType>();
  auto elementBytes = getElementSize(ty.getElementType());
  if (hasDynamicDimension(buffer)) {
    if (auto recvOp = dyn_cast<disc_ral::RecvInputOp>(buffer.getDefiningOp())) {
      // get first user of the buffer
      if (auto rcastOp =
              dyn_cast<memref::ReinterpretCastOp>(*buffer.getUsers().begin())) {
        buffer = rcastOp.getResult();
      }
    }
    auto symDims = getMemRefValueSymbolicDims(*mgr, buffer);
    SymbolicDimProduct prod{symDims.value()};
    prod = SimplifySymbolicDims(prod, s0.value());
    prod.factor *= elementBytes;
    return mgr->simplifySymbolicDimProduct(prod);
  }
  SymbolicDimProduct prod{};
  int64_t dimProduct = 1;
  for (int64_t dim : ty.getShape()) {
    dimProduct *= dim;
  }
  prod.factor = dimProduct * elementBytes;
  return prod;
}
bool isHostBuffer(Value value) {
  auto memRefType = value.getType().cast<MemRefType>();
  return memRefType.getMemorySpace() == 0;
}
Value maybeInReloadBlock(Value buffer) {
  if (auto ifOp = dyn_cast<scf::IfOp>(buffer.getDefiningOp()->getParentOp())) {
    return ifOp.getResult(0);
  }
  return buffer;
}
/////////////////////// SymbolicMemoryProfiler ///////////////////////
bool SymbolicMemoryProfiler::isTempBuffer(Value value) {
  Operation* prevFusionOp = nullptr;
  for (auto user : value.getUsers()) {
    if (isa<memref::DeallocOp>(user)) continue;
    if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(user->getParentOp())) {
      if (!prevFusionOp) prevFusionOp = fusionOp;
      if (prevFusionOp != fusionOp) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

int64_t getConcretValuewithFakeValue(MemoryUsage memoryUsages,
                                     int64_t cstValue) {
  int64_t result = 0;
  for (auto prod : memoryUsages) {
    int64_t factor = prod.factor;
    if (prod.symbols.size() > 0)
      factor *= std::pow(cstValue, prod.symbols.size());
    result += factor;
  }
  return result;
}
float bytesToMB(int64_t bytes) { return bytes * 1.0 / (1024.0 * 1024.0); }
MemoryUsage findPeakMemoryWithFakeValue(
    std::vector<MemoryUsage> memoryUsageList, int64_t fakeValue) {
  size_t instIndex = 0;
  int64_t maxMemoryUsage = 0;
  for (size_t i = 0; i < memoryUsageList.size(); ++i) {
    int64_t memoryUsage =
        getConcretValuewithFakeValue(memoryUsageList[i], fakeValue);
    if (memoryUsage > maxMemoryUsage) {
      maxMemoryUsage = memoryUsage;
      instIndex = i;
    }
  }
  llvm::dbgs() << "fakeValue: " << fakeValue << " maxMemory: " << maxMemoryUsage
               << " bytes"
               << " instIndex: " << instIndex
               << " expr: " << memoryUsageList[instIndex] << "\n";
  return memoryUsageList[instIndex];
}
std::vector<int64_t> ConcretMemoryUsageSimulator(int64_t concretValue) {
  /*
  SymbolicDimProductSum memoryUsage;
  SmallVector<SymbolicDimProductSum> memoryUsageList;
  std::unordered_set<Value, ValueHash> set, skipBuffers;
  mlir::OpBuilder b(main);
  llvm::dbgs() << "memory usage with seqlen=" << cstS0 << "\n";
  int rematBuffers = 0;
  main.walk([&](Operation* op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (allocOp.getOperation()->getAttr(
              b.getStringAttr("disc.remat.dummy-buffer"))) {
        return;
      }
      // alloc operator maybe inside a remat block
      auto buffer = allocOp.getResult();
      auto reloadBuffer = maybeInReloadBlock(buffer);
      if (auto rematBlock = getRematBlock(allocOp.getOperation())) {
        return;
      }
      if (shouldSkipBufferInPeakMemoryEstimator(buffer)) {
        skipBuffers.insert(reloadBuffer);
        return;
      }
      auto bufferBytes = getSymMemrefSize(buffer, mgr);
      memoryUsage = symbolicDimProductSumAdd(memoryUsage, bufferBytes);
      memoryUsageList.push_back(memoryUsage);
      set.insert(reloadBuffer);
      llvm::dbgs() << bytesToMB(getConcretValuewithCst(memoryUsage, cstS0))
                   << "\n";

    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      auto buffer = deallocOp.getOperand();
      if (skipBuffers.count(buffer)) {
        return;
      }
      if (auto rematBlock = getRematBlock(deallocOp.getOperation())) {
        return;
      }
      auto bufferBytes = getSymMemrefSize(buffer, mgr);
      memoryUsage = symbolicDimProductSub(memoryUsage, bufferBytes);
      memoryUsageList.push_back(memoryUsage);
      set.erase(buffer);
      llvm::dbgs() << bytesToMB(getConcretValuewithCst(memoryUsage, cstS0))
                   << "\n";
    } else if (auto d2hOp = dyn_cast<lmhlo_disc::D2HOp>(op)) {
      auto buffer = d2hOp.getOperand(0);
      auto rematBlock = getRematBlock(d2hOp.getOperation());
      auto rematBuffer = rematBlock->getResult(0);
      int64_t minS0 =
          rematBlock->getAttrOfType<IntegerAttr>(kRematMinSymDim).getInt();
      if (cstS0 > minS0) {
        auto bufferBytes = getSymMemrefSize(buffer, mgr);
        memoryUsage = symbolicDimProductSub(memoryUsage, bufferBytes);
        llvm::dbgs() << bytesToMB(getConcretValuewithCst(memoryUsage, cstS0))
                     << "\n";
      }
    } else if (auto h2dOp = dyn_cast<lmhlo_disc::H2DOp>(op)) {
      auto buffer = h2dOp.getOperand(1);
      auto rematBlock = getRematBlock(h2dOp.getOperation());
      auto rematBuffer = rematBlock->getResult(0);
      int64_t minS0 =
          rematBlock->getAttrOfType<IntegerAttr>(kRematMinSymDim).getInt();
      if (cstS0 > minS0) {
        auto rematBuffer = rematBlock->getResult(0);
        auto bufferBytes = getSymMemrefSize(buffer, mgr);
        memoryUsage = symbolicDimProductSumAdd(memoryUsage, bufferBytes);
        llvm::dbgs() << bytesToMB(getConcretValuewithCst(memoryUsage, cstS0))
                     << "\n";
        rematBuffers++;
      }
    }
  });
  */
}

LogicalResult SymbolicMemoryProfiler::Analysis() {
  mlir::OpBuilder b(main_);
  std::unique_ptr<ShapeAnalysis> shapeAnalysisPtr;
  shapeAnalysisPtr.reset(new ShapeConstraintIRAnalysis(main_));
  auto shapeIRAnalysis =
      dynamic_cast<ShapeConstraintIRAnalysis*>(shapeAnalysisPtr.get());
  if (!shapeIRAnalysis) {
    llvm::errs() << "shape analysis failed\n";
    return failure();
  }
  mgr_ = &shapeIRAnalysis->symbolicDimMgr();
  memory_usage_list_.clear();

  MemoryUsage currentUsage;
  std::unordered_set<Value, ValueHash> skipBuffers;
  main_.walk([&](Operation* op) {
    if (auto recvOp = dyn_cast<disc_ral::RecvInputOp>(op)) {
      auto buffer = recvOp.getResult();
      auto bufferBytes = getSymbolicMemrefBytes(buffer, mgr_);
      currentUsage += bufferBytes;
      memory_usage_list_.push_back(currentUsage);
    } else if (auto sendOp = dyn_cast<disc_ral::SendOutputOp>(op)) {
      auto buffer = sendOp.getOperand(2);
      auto bufferBytes = getSymbolicMemrefBytes(buffer, mgr_);
      currentUsage -= bufferBytes;
      memory_usage_list_.push_back(currentUsage);
    } else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      // TODO(yancey): dummy attr is a workaround implement for remat,
      // better to have a custom operator
      if (allocOp.getOperation()->getAttr(
              b.getStringAttr("disc.remat.dummy-buffer"))) {
        return;
      }
      // alloc operator maybe inside a remat block
      auto buffer = allocOp.getResult();
      // skip the buffer if it is a temp buffer of fusion op
      auto reloadBuffer = maybeInReloadBlock(buffer);
      if (isHostBuffer(buffer) || isTempBuffer(buffer)) {
        skipBuffers.insert(reloadBuffer);
        return;
      }

      auto bufferBytes = getSymbolicMemrefBytes(buffer, mgr_);
      currentUsage += bufferBytes;
      memory_usage_list_.push_back(currentUsage);
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      auto buffer = deallocOp.getOperand();
      if (skipBuffers.count(buffer)) {
        return;
      }
      auto bufferBytes = getSymbolicMemrefBytes(buffer, mgr_);
      currentUsage -= bufferBytes;
      memory_usage_list_.push_back(currentUsage);
    }
  });
  // NOTE: search peak memory expr with fake symbolic value, please
  // note, it's a fuzzy search algorithm, the result may not be accurate
  // but it's good enough for most cases
  peak_memory_ = findPeakMemoryWithFakeValue(memory_usage_list_, 2048);
  return success();
}

}  // namespace disc_ral
}  // namespace mlir