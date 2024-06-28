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
};

struct LiveRange {
  Operation* startOp;
  Operation* endOp;
  std::vector<std::pair<Operation*, int64_t>> ranges;
  int64_t start_position;
};
struct LiveRangeHash {
  std::size_t operator()(const Value& operand) const {
    std::size_t hash = mlir::hash_value(operand);
    return hash;
  }
};
using BufferLiveRanges =
    std::unordered_map<mlir::Value, LiveRange, LiveRangeHash>;
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

Operation* mayFusionOp(Operation* op) {
  if (auto fusionOp = dyn_cast_or_null<lmhlo::FusionOp>(op->getParentOp())) {
    return fusionOp;
  }
  return op;
}
float bytesToGB(int64_t bytes) { return bytes * 1.0 / (1024.0 * 1024.0); }
int64_t memoryPeakEvalution(ModuleOp main, bool printDetail = false) {
  int64_t peakMemory = 0;
  main.walk([&](Operation* op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      // skip if host buffer
      for (auto user : allocOp.getResult().getUsers()) {
        if (isa<lmhlo_disc::D2HOp>(user)) {
          return;
        }
      }
      peakMemory += getMemRefSize(allocOp.getResult());
      if (printDetail) {
        llvm::dbgs() << "memory usage: "
                     << llvm::format("%0.2f", bytesToGB(peakMemory)) << " MB\n";
      }
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      peakMemory -= getMemRefSize(deallocOp.getOperand());
      if (printDetail) {
        llvm::dbgs() << "memory usage: "
                     << llvm::format("%0.2f", bytesToGB(peakMemory)) << " MB\n";
      }
    }
  });
  return peakMemory;
}
BufferLiveRanges getBufferLiveRanges(ModuleOp main) {
  BufferLiveRanges bufferLiveRanges;
  int64_t position = 0;
  main.walk([&](Operation* op) {
    // Traverse the function's blocks and operations.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      LiveRange range;
      range.startOp = op;
      range.start_position = position;
      bufferLiveRanges[allocOp.getResult()] = range;
    } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
      bufferLiveRanges[deallocOp.getOperand()].endOp = op;
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

void DiscOffloadingPass::runOnOperation() {
  auto main = getOperation();
  auto context = main->getContext();
  mlir::OpBuilder rewriter(context);
  // BufferLiveRange bufferLiveRange(main);
  // bufferLiveRange.Analysis();
  //  1. find all buffer live-range
  // auto bufferLiveRanges = getBufferLiveRanges(main);
  bool changed = true;
  int iteration = 300;
  memoryPeakEvalution(main, true);
  llvm::dbgs() << "=================\n";
  while (iteration--) {
    changed = false;
    int64_t memoryPeak = memoryPeakEvalution(main);
    auto bufferLiveRanges = getBufferLiveRanges(main);
    // llvm::dbgs() << "memoryPeak: "
    //              << memoryPeak * 1.0 / (1024.0 * 1024.0 * 1024.0) << " GB
    //              \n";
    for (auto& pair : bufferLiveRanges) {
      auto buffer = pair.first;
      auto liveRange = pair.second;
      for (size_t i = 1; i < liveRange.ranges.size(); i++) {
        // TODO(yancey.yx) using a common cost function to decide the offloading
        if (liveRange.ranges[i].second > 1000) {
          changed = true;
          auto prevOp = liveRange.ranges[i - 1].first;
          auto op = liveRange.ranges[i].first;
          if (auto fusionOp =
                  dyn_cast<lmhlo::FusionOp>(prevOp->getParentOp())) {
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
          rewriter.create<lmhlo_disc::D2HOp>(op->getLoc(), buffer, hostBuffer);
          rewriter.create<memref::DeallocOp>(op->getLoc(), buffer);

          if (auto fusionOp = dyn_cast<lmhlo::FusionOp>(op->getParentOp())) {
            rewriter.setInsertionPoint(fusionOp);
          } else {
            rewriter.setInsertionPoint(op);
          }
          // loading to device
          auto deviceBuffer =
              rewriter
                  .create<memref::AllocOp>(op->getLoc(),
                                           buffer.getType().cast<MemRefType>())
                  .getResult();
          rewriter.create<lmhlo_disc::H2DOp>(op->getLoc(), hostBuffer,
                                             deviceBuffer);
          for (size_t j = i; j < liveRange.ranges.size(); j++) {
            auto consumer = liveRange.ranges[i].first;
            for (size_t k = 0; k < op->getNumOperands(); k++) {
              if (consumer->getOperand(k) == buffer) {
                consumer->setOperand(k, deviceBuffer);
              }
            }
          }
          llvm::dbgs() << "cutting live range: " << liveRange.ranges[i].second
                       << " to 0, after offloading buffer: " << buffer
                       << " reduce peak memory from "
                       << memoryPeak * 1.0 / (1024.0 * 1024.0 * 1024.0)
                       << " GB to "
                       << memoryPeakEvalution(main) * 1.0 /
                              (1024.0 * 1024.0 * 1024.0)
                       << " GB\n";
          break;
        }
      }
      if (changed) break;
    }
    memoryPeakEvalution(main, true);
  }
}
std::unique_ptr<OperationPass<ModuleOp>> createDiscOffloadingPass() {
  return std::make_unique<DiscOffloadingPass>();
}

}  // namespace disc_ral
}  // namespace mlir