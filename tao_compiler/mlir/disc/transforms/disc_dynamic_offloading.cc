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

#include <unordered_map>

#include "absl/strings/str_split.h"
#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_remat_utils.h"
#include "mlir/disc/transforms/shape_utils.h"

namespace mlir {
namespace disc_ral {
constexpr StringRef kRematBlockTypeAttr = "disc.remat.type";
constexpr StringRef kRematBufferAttr = "disc.remat.is_dummy_buffer";
constexpr StringRef kRematMinSymDim = "disc.remat.min_symbolic_dim";

struct DiscOffloadingPass : public DiscOffloadingPassBase<DiscOffloadingPass> {
  DiscOffloadingPass()
      : DiscOffloadingPassBase<DiscOffloadingPass>::DiscOffloadingPassBase() {}
  void getDependentDialects(DialectRegistry& registry) const override {
    DiscOffloadingPassBase<DiscOffloadingPass>::getDependentDialects(registry);
    registry.insert<shape::ShapeDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<lmhlo_disc::LmhloDiscDialect>();
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override;
  void InsertRematBlock(mlir::OpBuilder& b, LivingBuffer& livingBuffer,
                        Value rematCond, std::vector<OpWithPosition>& ops,
                        int64_t minSymValue);
};
Location getFusionLocation(OpBuilder& b, Operation* op) {
  if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(op->getParentOp())) {
    b.setInsertionPointAfter(fusionOp);
    return fusionOp->getLoc();
  }
  b.setInsertionPointAfter(op);
  return op->getLoc();
}

using FuncOp = mlir::func::FuncOp;

SymbolicDimProduct getSymbolicMemRefSize(Value value, SymbolicDimMgr* mgr,
                                         MLIRContext* ctx) {
  auto memRefType = value.getType().cast<MemRefType>();
  // get symbolic dims of the memref
  // auto symbolics = getSymbolicDims(value);
  auto symbolics = mgr->getOrCreateSymbolicDimsForRankedValue(value);
  SymbolicDimProduct prod{symbolics};
  return prod;
}
/*
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
*/

bool shouldSkipBufferInPeakMemoryEstimator(Value value) {
  if (IsHostBuffer(value)) return true;

  // skip buffer if it is a temp buffer which only used inside of a fusion op
  // alloc = memref.alloc
  // lmhlo.fusion() {
  //    op1(buffer0, buffer1, alloc)
  //    op2(alloc, buffer2, buffer3)
  // }
  // dealloc = memref.dealloc alloc
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

bool IsDynamicShapeBuffer(Value buffer) {
  auto memrefType = buffer.getType().cast<MemRefType>();
  for (auto dim : memrefType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      return true;
    }
  }
  return false;
}
Value cloneBuffer(OpBuilder& b, Location loc, Value buffer) {
  MemRefType type = refBuffer.getType().cast<MemRefType>();
  SmallVector<Value, 4> dynShape;
  for (size_t i = 0; i < type.getRank(); i++) {
    if (type.getShape()[i] == ShapedType::kDynamic) {
      dynShape.push_back(b.create<memref::DimOp>(loc, refBuffer, i));
    }
  }
  auto allocOp = b.create<memref::AllocOp>(loc, type, dynShape);
  StringRef attrName = SymbolicDimOp::getSymbolicDimAttrName();
  if (refBuffer.getDefiningOp()->hasAttr(attrName)) {
    allocOp.getOperation()->setAttr(
        attrName, refBuffer.getDefiningOp()->getAttr(attrName));
  }
  return allocOp.getResult();
}

// InsertRematBlock create a remat block for the living buffer:
// reload and offload blocks are always pair in graph:
//
// if remat_cond:
//   return offload(buffer)
// else:
//   return dummy_buffer
// ......
// if remat_cond:
//   return reload(buffer)
// else:
//   return buffer
void DiscOffloadingPass::InsertRematBlock(mlir::OpBuilder& b,
                                          LivingBuffer& livingBuffer,
                                          Value rematCond,
                                          std::vector<OpWithPosition>& ops,
                                          int64_t minSymValue) {
  // TODO(yancey): we need a custom operator to handle the dummy buffer to avoid
  // call alloc or dealloc operator
  auto buffer = livingBuffer.buffer;
  auto startOp = livingBuffer.start;
  auto endOp = livingBuffer.end;
  StringRef attrName = SymbolicDimOp::getSymbolicDimAttrName();
  auto deviceMemrefType = buffer.getType().cast<MemRefType>();
  auto hostMemrefType = MemRefType::get(
      deviceMemrefType.getShape(), deviceMemrefType.getElementType(), {}, 0);
  // insert offload block
  auto offloadIfOp =
      b.create<scf::IfOp>(startOp->getLoc(),
                          /*resultTypes*/ hostMemrefType, rematCond,
                          /*hasElseRegion*/ true);
  offloadIfOp.getOperation()->setAttr(
      attrName, buffer.getDefiningOp()->getAttr(attrName));
  offloadIfOp.getOperation()->setAttr(kRematBlockTypeAttr,
                                      b.getStringAttr("offload"));
  offloadIfOp.getOperation()->setAttr(kRematMinSymDim,
                                      b.getI64IntegerAttr(minSymValue));
  offloadIfOp.getThenRegion().front().clear();
  b.setInsertionPointToEnd(&offloadIfOp.getThenRegion().front());
  auto hostBuffer = cloneBuffer(b, endOp->getLoc(), buffer);
  hostBuffer.setType(MemRefType::get(deviceMemrefType.getShape(),
                                     deviceMemrefType.getElementType(), {}, 0));
  b.create<lmhlo_disc::D2HOp>(endOp->getLoc(), buffer, hostBuffer);
  b.create<memref::DeallocOp>(endOp->getLoc(), buffer);
  b.create<scf::YieldOp>(endOp->getLoc(), hostBuffer);

  offloadIfOp.getElseRegion().front().clear();
  b.setInsertionPointToStart(&offloadIfOp.getElseRegion().front());
  auto dummyHostBuffer = cloneBuffer(b, endOp->getLoc(), buffer);
  dummyHostBuffer.getDefiningOp()->setAttr(kRematBufferAttr,
                                           b.getBoolAttr(true));
  dummyHostBuffer.setType(MemRefType::get(
      deviceMemrefType.getShape(), deviceMemrefType.getElementType(), {}, 0));
  b.create<scf::YieldOp>(endOp->getLoc(), dummyHostBuffer);
  b.setInsertionPointAfter(offloadIfOp);

  if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(endOp->getParentOp())) {
    b.setInsertionPoint(fusionOp);
  } else {
    b.setInsertionPoint(endOp);
  }

  // insert reload block
  scf::IfOp reloadIfOp =
      b.create<scf::IfOp>(endOp->getLoc(),
                          /*resultTypes*/ deviceMemrefType, rematCond,
                          /*hasElseRegion*/ true);
  if (buffer.getDefiningOp()->hasAttr(attrName)) {
    reloadIfOp.getOperation()->setAttr(
        attrName, buffer.getDefiningOp()->getAttr(attrName));
  }
  reloadIfOp.getOperation()->setAttr(kRematBlockTypeAttr,
                                     b.getStringAttr("reload"));
  reloadIfOp.getOperation()->setAttr(kRematMinSymDim,
                                     b.getI64IntegerAttr(minSymValue));
  reloadIfOp.getThenRegion().front().clear();
  b.setInsertionPointToStart(&reloadIfOp.getThenRegion().front());

  auto deviceBuffer = cloneBuffer(b, endOp->getLoc(), buffer);
  deviceBuffer.setType(deviceMemrefType);
  auto h2dOp = b.create<lmhlo_disc::H2DOp>(
      endOp->getLoc(), offloadIfOp.getResult(0), deviceBuffer);
  b.create<scf::YieldOp>(endOp->getLoc(), deviceBuffer);
  reloadIfOp.getElseRegion().front().clear();
  b.setInsertionPointToStart(&reloadIfOp.getElseRegion().front());
  auto dummyDeviceBuffer = cloneBuffer(b, endOp->getLoc(), buffer);
  dummyDeviceBuffer.setType(deviceMemrefType);
  dummyDeviceBuffer.getDefiningOp()->setAttr(kRematBufferAttr,
                                             b.getBoolAttr(true));
  b.create<scf::YieldOp>(endOp->getLoc(), buffer);
  for (auto pair : ops) {
    auto op = pair.first;
    auto position = pair.second;
    if (position > livingBuffer.end_position) {
      for (size_t i = 0; i < op->getNumOperands(); i++) {
        if (op->getOperand(i) == buffer) {
          op->setOperand(i, reloadIfOp.getResult(0));
        }
      }
    }
  }
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const SymbolicDimProduct& prod) {
  // print prod
  os << prod.factor;
  if (prod.symbols.size() > 0) os << "*";
  for (size_t j = 0; j < prod.symbols.size(); ++j) {
    auto dimOp = prod.symbols[j];
    if (j != prod.symbols.size() - 1) {
      os << dimOp.getName() << "*";
    } else {
      os << dimOp.getName();
    }
  }
  return os;
}
std::tuple<bool, double, double> solveQuadratic(int64_t a, int64_t b,
                                                int64_t c) {
  if (a == 0) {
    throw std::invalid_argument(
        "Coefficient A cannot be zero in a quadratic equation.");
  }

  double discriminant = b * b - 4 * a * c;
  // no solution if b^2 - 4ac < 0
  if (discriminant < 0) {
    return std::make_tuple(false, 0.0, 0.0);
  }

  double sqrtDiscriminant = std::sqrt(discriminant);

  // compute x1, x2
  int64_t x1 = (-b + sqrtDiscriminant) / (2 * a);
  int64_t x2 = (-b - sqrtDiscriminant) / (2 * a);
  return std::make_tuple(true, x1, x2);
}
int64_t findMinSymbolicDimValue(MemoryUsage memoryPeakExpr,
                                int64_t memoryLimitation) {
  int64_t a, b, c = 0;
  // memoryPeakExpr = a * S0^2 + b * S0 + c
  // memoryPeakExpr >= memoryLimitation
  // a * S0^2 + b * S0 + c - memoryLimitation >= 0
  for (auto prod : memoryPeakExpr) {
    if (prod.symbols.size() == 0) {
      c = prod.factor;
    } else if (prod.symbols.size() == 1) {
      b = prod.factor;
    } else if (prod.symbols.size() == 2) {
      a = prod.factor;
    }
  }
  c -= memoryLimitation;
  if (a == 0) {
    return -c / b;
  }
  auto [hasSolution, x0, x1] = solveQuadratic(a, b, c);
  if (!hasSolution) {
    throw std::invalid_argument("No solution for the quadratic equation.");
  }
  return std::max(x0, x1);
}

bool inRematOffloadBlock(Value value) {
  if (auto ifOp = dyn_cast<scf::IfOp>(value.getDefiningOp()->getParentOp())) {
    auto blockType =
        ifOp.getOperation()->getAttrOfType<StringAttr>(kRematBlockTypeAttr);
    if (blockType && blockType.getValue() == "offload") {
      return true;
    }
  }
  return false;
}

std::optional<Value> findSymbolicDimValue(FuncOp& main,
                                          const std::string& key) {
  SmallVector<Value, 4> dynInputs;
  std::unordered_map<std::string, Value> symValueMap;
  main.walk([&](Operation* op) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    if (!allocOp) {
      return;
    }
    if (op->getNumOperands() == 0) {
      return;
    }
    auto attrs =
        op->getAttrOfType<ArrayAttr>(SymbolicDimOp::getSymbolicDimAttrName());
    int symbDimIndex = 0;
    for (auto attr : attrs) {
      auto name = attr.cast<FlatSymbolRefAttr>().getValue();
      if (name.startswith("S")) {
        if (symValueMap.count(name.str()) == 0) {
          symValueMap[name.str()] = op->getOperand(symbDimIndex++);
        }
      }
    }
    return;
  });
  if (symValueMap.count(key) == 0) {
    return std::nullopt;
  }
  return symValueMap[key];
}
Value InsertRematCond(mlir::OpBuilder& b, Location loc, Value s0,
                      int64_t minS0) {
  auto offloadCond = b.create<arith::CmpIOp>(
      s0.getLoc(), arith::CmpIPredicate::sgt, s0,
      b.create<arith::ConstantIndexOp>(s0.getLoc(), minS0));
  return offloadCond;
}
std::vector<LivingBuffer> FilterBuffers(
    std::vector<LivingBuffer> livingBuffers) {
  std::vector<LivingBuffer> result;
  for (auto lb : livingBuffers) {
    // TODO(yancey): just for experiment, let's remove this condition in the
    // future
    if (!IsDynamicShapeBuffer(lb.buffer)) {
      continue;
    }
    // filter buffer if already in remat block
    if (isa<scf::IfOp>((lb.start->getParentOp()))) continue;
    result.push_back(lb);
  }
  return result;
}
void SortBuffersByPrioriy(std::vector<LivingBuffer>& livingBuffers) {
  std::sort(livingBuffers.begin(), livingBuffers.end(),
            [](const LivingBuffer& a, const LivingBuffer& b) {
              return a.living_range > b.living_range;
            });
}
std::optional<LivingBuffer> PickHighestPriorityLivingBuffer(
    const std::vector<LivingBuffer>& livingBuffers) {
  // step1: filter living buffers which can not reduce the peak memory or too
  // small buffer
  auto buffers = FilterBuffers(livingBuffers);
  if (buffers.size() == 0) {
    return std::nullopt;
  }
  // step2: sort living buffers by priority, e.g. living range value
  SortBuffersByPrioriy(buffers);
  return buffers[0];
}
void DiscOffloadingPass::runOnOperation() {
  FuncOp main = getOperation();
  if (main.getName() == SymbolicDimMgr::getShapeConstraintGraphFunctionName())
    return;
  mlir::OpBuilder b(main);
  const int64_t memoryLimitation = 21474836480;  // 30GB
  llvm::dbgs() << "memory limitation: " << memoryLimitation << "\n";
  bool changed = true;
  int maxIteration = 200;
  std::unique_ptr<SymbolicMemoryProfiler> profiler(
      new SymbolicMemoryProfiler(main));
  std::unique_ptr<DiscBufferLivingRange> bufferLivingRange(
      new DiscBufferLivingRange(main));
  // mapping symbolic dim(S0) in shape constrint graph to SSA value
  // TODO(yancey): please note, we only support only one symbolic dim now,
  // let's find a way to enhancement this.
  auto symS0Value = findSymbolicDimValue(main, "S0");
  if (!symS0Value) {
    llvm::errs() << "failed to find S0 value\n";
    return;
  }
  while (changed && maxIteration--) {
    if (failed(profiler->Analysis())) {
      llvm::errs() << "failed to analysis\n";
      return;
    }
    if (failed(bufferLivingRange->Analysis())) {
      llvm::errs() << "failed to analysis buffer living range\n";
      return;
    }
    changed = false;
    auto memoryPeakExpr = profiler->GetPeakMemory();
    int64_t minS0 = findMinSymbolicDimValue(memoryPeakExpr, memoryLimitation);

    auto livingBuffer =
        PickHighestPriorityLivingBuffer(bufferLivingRange->GetLivingBuffers());
    if (livingBuffer.has_value()) {
      auto buffer = livingBuffer.value();
      auto users = bufferLivingRange->GetUsersOrderByPosition(buffer.buffer);
      auto loc = getFusionLocation(b, buffer.start);
      auto rematCond = InsertRematCond(b, loc, symS0Value.value(), minS0);
      InsertRematBlock(b, buffer, rematCond, users, minS0);
      changed = true;
    }
  }
}
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createDiscOffloadingPass() {
  return std::make_unique<DiscOffloadingPass>();
}

}  // namespace disc_ral
}  // namespace mlir