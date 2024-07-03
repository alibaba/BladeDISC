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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/disc/transforms/shape_utils.h"

namespace mlir {
namespace disc_ral {

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
  }
  void runOnOperation() override;
  void InsertOffloadingOp(mlir::OpBuilder& rewriter, Operation* prevOp,
                          Operation* op, Value buffer,
                          std::vector<Operation*> consumers);
};

struct LiveRange {
  Value buffer;
  Operation* startOp = nullptr;
  Operation* endOp = nullptr;
  std::vector<std::pair<Operation*, int64_t>> ranges;
  int64_t start_position;
  int64_t end_position;
};

struct LiveRangeHash {
  std::size_t operator()(const Value& operand) const {
    std::size_t hash = mlir::hash_value(operand);
    return hash;
  }
};
using BufferLiveRanges =
    std::unordered_map<mlir::Value, LiveRange, LiveRangeHash>;
using BufferList = std::vector<std::pair<mlir::Value, LiveRange>>;
using BufferMap = std::unordered_map<mlir::Value, LiveRange>;
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
  } else {
    llvm::dbgs() << elementType << "\n";
    // Add more types as needed
    llvm::errs() << "Unsupported element type\n";
    return -1;
  }
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

float bytesToMB(int64_t bytes) { return bytes * 1.0 / (1024.0 * 1024.0); }

bool isHostBuffer(Value value) {
  auto memRefType = value.getType().cast<MemRefType>();
  return memRefType.getMemorySpace() == 0;
}

int64_t memoryPeakEvalution(ModuleOp main, bool printDetail = false) {
  int64_t usageMemory = 0;
  int64_t peakMemory = 0;
  main.walk([&](Operation* op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      // skip if host buffer
      auto buffer = allocOp.getResult();
      if (isHostBuffer(buffer)) {
        return;
      }
      usageMemory += getMemRefSize(allocOp.getResult());
      peakMemory = std::max(peakMemory, usageMemory);
      if (printDetail) {
        llvm::dbgs() << "memory usage: "
                     << llvm::format("%0.2f", bytesToMB(usageMemory))
                     << " MB\n";
      }
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      usageMemory -= getMemRefSize(deallocOp.getOperand());
      if (printDetail) {
        llvm::dbgs() << "memory usage: "
                     << llvm::format("%0.2f", bytesToMB(usageMemory))
                     << " MB\n";
      }
    }
  });
  return peakMemory;
}

struct LivingBuffers {
  Value buffer;
  Operation* startOp = nullptr;
  Operation* endOp = nullptr;
  std::vector<std::pair<Operation*, int64_t>> consumers;
  int64_t start_position;
};

class BufferLiveRange {
 public:
  explicit BufferLiveRange(ModuleOp main) : main_(main) {}
  void AllocateBuffer(Value value, Operation* op, int64_t position);
  void Analysis();
  std::vector<Value> getBufferList() { return buffer_list_; }
  LivingBuffers getLivingBuffers(Value value) {
    if (living_buffer_map_.count(value) == 0) {
      return LivingBuffers();
    }
    return living_buffer_map_[value];
  }

 private:
  ModuleOp main_;
  std::vector<Value> buffer_list_;
  std::unordered_map<Value, LivingBuffers, LiveRangeHash> living_buffer_map_;
};
void BufferLiveRange::AllocateBuffer(Value value, Operation* op,
                                     int64_t position) {
  LivingBuffers livingBuffers;
  livingBuffers.startOp = op;
  livingBuffers.start_position = position;
  livingBuffers.buffer = value;
  living_buffer_map_[value] = livingBuffers;
  buffer_list_.push_back(value);
}
void BufferLiveRange::Analysis() {
  buffer_list_.clear();
  living_buffer_map_.clear();
  int64_t position;
  main_.walk([&](Operation* op) {
    // Traverse the function's blocks and operations.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      auto buffer = allocOp.getResult();
      if (isHostBuffer(allocOp.getResult())) {
        return;
      }
      AllocateBuffer(buffer, op, position);
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      living_buffer_map_[deallocOp.getOperand()].endOp = deallocOp;
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto operand : returnOp.getOperands()) {
        if (living_buffer_map_.count(operand)) {
          living_buffer_map_[operand].endOp = returnOp;
        }
      }
    } else if (isa<lmhlo_disc::H2DOp>(op) || isa<lmhlo_disc::D2HOp>(op)) {
      return;
    } else {
      for (Value operand : op->getOperands()) {
        if (living_buffer_map_.count(operand)) {
          living_buffer_map_[operand].consumers.push_back(std::make_pair(
              op, position - living_buffer_map_[operand].start_position));
        }
      }
    }
    position++;
  });
}

BufferLiveRanges getBufferLiveRanges(ModuleOp main) {
  BufferLiveRanges bufferLiveRanges;
  int64_t position = 0;
  main.walk([&](Operation* op) {
    // Traverse the function's blocks and operations.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (isHostBuffer(allocOp.getResult())) {
        return;
      }
      LiveRange range;
      range.startOp = op;
      range.start_position = position;
      auto buffer = allocOp.getResult();
      bufferLiveRanges[buffer] = range;
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      bufferLiveRanges[deallocOp.getOperand()].endOp = deallocOp;
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto operand : returnOp.getOperands()) {
        if (bufferLiveRanges.count(operand)) {
          bufferLiveRanges[operand].endOp = returnOp;
        }
      }
    } else if (isa<lmhlo_disc::H2DOp>(op) || isa<lmhlo_disc::D2HOp>(op)) {
      return;
    } else {
      for (Value operand : op->getOperands()) {
        if (bufferLiveRanges.count(operand)) {
          bufferLiveRanges[operand].ranges.push_back(std::make_pair(
              op, position - bufferLiveRanges[operand].start_position));
        }
      }
    }
    position++;
  });
  return bufferLiveRanges;
}
void DiscOffloadingPass::InsertOffloadingOp(mlir::OpBuilder& rewriter,
                                            Operation* prevOp, Operation* op,
                                            Value buffer,
                                            std::vector<Operation*> consumers) {
  auto memrefType = buffer.getType().cast<MemRefType>();
  // insert offloading an loading inst
  if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(prevOp->getParentOp())) {
    rewriter.setInsertionPointAfter(fusionOp);
  } else {
    rewriter.setInsertionPointAfter(prevOp);
  }
  // offloading to host
  auto hostBuffer =
      rewriter
          .create<memref::AllocOp>(prevOp->getLoc(),
                                   buffer.getType().cast<MemRefType>())
          .getResult();
  // delete gpu.address_global attribute on host buffer
  hostBuffer.setType(MemRefType::get(memrefType.getShape(),
                                     memrefType.getElementType(), {}, 0));
  rewriter.create<lmhlo_disc::D2HOp>(op->getLoc(), buffer, hostBuffer);
  rewriter.create<memref::DeallocOp>(op->getLoc(), buffer);

  if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(op->getParentOp())) {
    rewriter.setInsertionPoint(fusionOp);
  } else {
    rewriter.setInsertionPoint(op);
  }
  // loading to device
  auto deviceBuffer = rewriter
                          .create<memref::AllocOp>(
                              op->getLoc(), buffer.getType().cast<MemRefType>())
                          .getResult();
  rewriter.create<lmhlo_disc::H2DOp>(op->getLoc(), hostBuffer, deviceBuffer);
  // update operand value to new device buffer
  for (size_t i = 0; i < consumers.size(); i++) {
    auto consumer = consumers[i];
    for (size_t j = 0; j < consumer->getNumOperands(); j++) {
      if (consumer->getOperand(j) == buffer) {
        consumer->setOperand(j, deviceBuffer);
      }
    }
  }
}
void DiscOffloadingPass::runOnOperation() {
  auto main = getOperation();
  auto context = main->getContext();
  mlir::OpBuilder rewriter(context);
  //  1. find all buffer live-range
  bool changed = true;
  int iteration = 200;
  memoryPeakEvalution(main, true);
  llvm::dbgs() << "======\n";
  BufferLiveRange bufferLiveRange(main);
  while (!changed || iteration--) {
    changed = false;
    int64_t memoryPeak = memoryPeakEvalution(main);
    bufferLiveRange.Analysis();

    for (auto buffer : bufferLiveRange.getBufferList()) {
      auto livingBuffers = bufferLiveRange.getLivingBuffers(buffer);
      for (size_t i = 1; i < livingBuffers.consumers.size(); ++i) {
        // TODO(yancey.yx) using a common cost function to decide the offloading
        if (livingBuffers.consumers[i].second -
                livingBuffers.consumers[i - 1].second >
            1000) {
          auto prevOp = livingBuffers.consumers[i - 1].first;
          auto op = livingBuffers.consumers[i].first;

          // collect users of the buffer after offloading op
          std::vector<Operation*> consumers;
          for (size_t j = i; j < livingBuffers.consumers.size(); j++) {
            consumers.push_back(livingBuffers.consumers[j].first);
          }
          consumers.push_back(livingBuffers.endOp);

          InsertOffloadingOp(rewriter, prevOp, op, buffer, consumers);
          llvm::dbgs() << "cutting live range: "
                       << livingBuffers.consumers[i].second
                       << " to 0, after offloading buffer: " << buffer
                       << " reduce peak memory from "
                       << memoryPeak * 1.0 / (1024.0 * 1024.0 * 1024.0)
                       << " GB to "
                       << memoryPeakEvalution(main) * 1.0 /
                              (1024.0 * 1024.0 * 1024.0)
                       << " GB\n";
          changed = true;
          break;
        }
      }
      if (changed) break;
    }
  }
  llvm::dbgs() << "======\n";
  memoryPeakEvalution(main, true);
  main.dump();
}
std::unique_ptr<OperationPass<ModuleOp>> createDiscOffloadingPass() {
  return std::make_unique<DiscOffloadingPass>();
}

}  // namespace disc_ral
}  // namespace mlir