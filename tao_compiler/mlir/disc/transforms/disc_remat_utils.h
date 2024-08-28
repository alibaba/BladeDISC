/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "mlir/disc/transforms/shape_utils.h"
namespace mlir {
namespace disc_ral {

struct ValueHash {
  std::size_t operator()(const Value& operand) const {
    std::size_t hash = mlir::hash_value(operand);
    return hash;
  }
};
using OpWithPosition = std::pair<Operation*, int64_t>;

bool IsHostBuffer(Value value);
// return sizeof(dtype)
int64_t getElementSize(Type elementType);
// TODO(yancey): just a PoC implementation, need to implement this function
// by traveling the shape operators
SymbolicDimProduct SimplifySymbolicDims(SymbolicDimProduct proudct,
                                        SymbolicDimOp s0);

struct LivingBuffer {
  LivingBuffer(Operation* start, int64_t start_position, Operation* end,
               int64_t end_position, Value buffer)
      : start(start),
        start_position(start_position),
        end(end),
        end_position(end_position),
        buffer(buffer) {
    living_range = end_position - start_position;
  }

  Operation* start;
  int64_t start_position;
  Operation* end;
  int64_t end_position;
  Value buffer;
  int64_t living_range;
};

class DiscBufferLivingRange {
 public:
  explicit DiscBufferLivingRange(mlir::func::FuncOp main) : main_(main) {}

  LogicalResult Analysis();

  std::vector<LivingBuffer> GetLivingBuffers() { return living_buffers_; }
  // get all users of the buffer, ordered by position
  std::vector<OpWithPosition> GetUsersOrderByPosition(Value buffer);

 private:
  mlir::func::FuncOp main_;
  std::vector<LivingBuffer> living_buffers_;
  // mapping buffer to operator and position
  std::unordered_map<Value, std::vector<std::pair<Operation*, int64_t>>,
                     ValueHash>
      buffer_map_;
  std::vector<Value> buffer_list_;
};

using MemoryUsage = llvm::SmallVector<SymbolicDimProduct>;

MemoryUsage& operator+=(MemoryUsage& lhs, const SymbolicDimProduct& rhs);
MemoryUsage& operator-=(MemoryUsage& lhs, const SymbolicDimProduct& rhs);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const MemoryUsage& memoryUsage);

// SymbolicMemoryProfiler is used to analyze the memory usage of a
// mlir function, it will return the peak memory usage and the memory usage
class SymbolicMemoryProfiler {
 public:
  explicit SymbolicMemoryProfiler(mlir::func::FuncOp& main,
                                  ShapeConstraintIRAnalysis& shapeAnalysis)
      : main_(main), shapeAnalysis_(shapeAnalysis) {}
  LogicalResult Analysis();
  MemoryUsage GetPeakMemory() { return peak_memory_; }
  std::vector<MemoryUsage> GetMemoryUsageList() { return memory_usage_list_; }

  std::vector<int64_t> ConcretMemoryUsageSimulator(int64_t concretValue);

 private:
  // return true if it is a temp buffer which only used inside of a fusion op,
  // this buffer would be removed after codegen, an example pattern:
  //
  // alloc = memref.alloc
  // lmhlo.fusion() {
  //    op1(buffer0, buffer1, alloc)
  //    op2(alloc, buffer2, buffer3)
  // }
  // dealloc = memref.dealloc alloc
  bool isTempBuffer(Value value);
  MemoryUsage searchPeakMemory(SmallVector<MemoryUsage>& memoryUsageList);

  SymbolicDimMgr* mgr_;
  mlir::func::FuncOp main_;
  MemoryUsage peak_memory_;
  std::vector<MemoryUsage> memory_usage_list_;
  ShapeConstraintIRAnalysis& shapeAnalysis_;
};

}  // namespace disc_ral
}  // namespace mlir